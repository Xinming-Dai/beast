#!/usr/bin/env python3
"""Compact vertical stack of PSTHs of the top-BPS neurons for patch-embedding ridge encoders.

Each neuron is one small axis-free panel: the ground-truth PSTH in black and the PSTH predicted by
one or two models in the colors of ``plot_helpers.py`` (with two models, the first is drawn
thicker underneath and the second on top). Neurons are ranked by the BPS of the last model on the
held-out trials of one fold. The BPS of every stacked neuron is written to a txt file next to the
figure, as ``1.33 vs 1.55`` in the order of ``--model``. The PSTH computation and normalization are the ones of
``top_bps_psth_compare_models_patch.py``.

BPS alone can rank neurons whose predicted PSTH is flat, because it rewards a prediction that only
follows the trial-to-trial gain of a neuron. With ``--ranking``, the neurons are taken instead from
the ranking saved by ``neuron_ranking_and_filtering/rank_neurons_bps_corr_patch.py`` for this
session, last model, K and fold, keeping only neurons whose predicted PSTH correlates with the
ground truth (``--min-psth-corr``) and is modulated over time (``--min-modulation``);
``--no-filter`` ranks all neurons by BPS. The txt file then also lists the two filter metrics of
the last model.

```bash
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_corr_patch.py
python scripts/neural_analysis/neural_encoding_patch_embedding/psth_stack_patch.py \
  --session 78 --model dino full --k 6 --fold 0 --num-neurons 4 --no-show --ranking
```
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    MODELS,
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    SESSION_EIDS,
    load_ridge_predictions,
    load_session_spikes,
    model_color,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.top_bps_psth_compare_models_patch import (  # noqa: E402
    GROUND_TRUTH_COLOR,
    _normalize_traces,
    _psth,
    parse_trials_idx_arg,
    trials_idx_to_bounds,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.rank_neurons_bps_corr_patch import (  # noqa: E402
    RANKING_DIR,
    ranking_path,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (  # noqa: E402
    add_ranking_args,
    load_ranking,
    select_top_neurons,
    thresholds_from_args,
)
from scripts.neural_analysis.plot_helpers import (  # noqa: E402
    _default_output_path_helper,
    bps_per_neuron,
)


def plot_psth_stack(
    gt: np.ndarray,
    preds: list[np.ndarray],
    neuron_indices: list[int],
    *,
    colors: list[str],
    normalize: bool = True,
    figsize: tuple[float, float] = (1.0, 1.6),
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Plot the ground-truth and predicted PSTHs of each neuron as a vertical stack.

    Args:
        gt: ground-truth spikes, shape (trials, time, neurons).
        preds: predicted rates of each model, same shape as `gt`, at most two.
        neuron_indices: neurons from top to bottom.
        colors: color of the predicted PSTH of each model.
        normalize: min-max normalize each neuron over all its traces, so all panels share a scale.
        figsize: figure size in inches.
        save_path: output image path.
        also_save_pdf: also write a PDF next to `save_path`.
        show: open an interactive window.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: if there are not one or two models, or `colors` does not match `preds`.
    """
    if not 1 <= len(preds) <= 2 or len(colors) != len(preds):
        raise ValueError(f'expected one or two models with one color each, got {len(preds)}')
    # the first of two models is thicker and sits underneath, so both stay visible
    widths = [2.2, 1.2] if len(preds) == 2 else [1.2]
    fig, axes = plt.subplots(len(neuron_indices), 1, figsize=figsize, dpi=300, squeeze=False)
    time_s = np.arange(gt.shape[1])
    for ax, neuron_idx in zip(axes[:, 0], neuron_indices, strict=True):
        gt_trace = _psth(gt[:, :, neuron_idx])
        traces = [gt_trace] + [_psth(pred[:, :, neuron_idx]) for pred in preds]
        if normalize:
            traces = _normalize_traces(traces)
        ax.plot(time_s, traces[0], color=GROUND_TRUTH_COLOR, linewidth=0.8)
        for order, (trace, color, width) in enumerate(zip(traces[1:], colors, widths)):
            ax.plot(time_s, trace, color=color, linewidth=width, zorder=3 + order)
        ax.set_xlim(time_s[0], time_s[-1])
        ax.axis('off')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.12)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, facecolor='white')
        if also_save_pdf:
            fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.02)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def format_bps_text(
    neuron_indices: list[int],
    bps_by_model: list[np.ndarray],
    models: list[str],
    *,
    session: str,
    k: int,
    fold: int,
    psth_corr: np.ndarray | None = None,
    modulation: np.ndarray | None = None,
) -> str:
    """Text with the BPS of every stacked neuron, one line per neuron from top to bottom.

    Args:
        neuron_indices: stacked neurons from top to bottom.
        bps_by_model: per-neuron BPS of each model, shape (neurons,) each.
        models: model names, in the order of `bps_by_model`.
        session: session name.
        k: number of PCA components.
        fold: held-out fold.
        psth_corr: per-neuron PSTH correlation of the last model, shape (neurons,). Appended to
            each line together with `modulation`.
        modulation: per-neuron PSTH modulation of the last model, shape (neurons,).

    Returns:
        The text, with BPS rounded to two decimals and joined by ` vs `.
    """
    names = ' vs '.join(MODEL_LABELS[model] for model in models)
    lines = [f'BPS ({names}), session {session}, K={k}, fold {fold}', '']
    for neuron_idx in neuron_indices:
        values = ' vs '.join(f'{bps[neuron_idx]:.2f}' for bps in bps_by_model)
        line = f'neuron {neuron_idx}: {values}'
        if psth_corr is not None and modulation is not None:
            line += (
                f' | {MODEL_LABELS[models[-1]]} psth r={psth_corr[neuron_idx]:.2f},'
                f' mod={modulation[neuron_idx]:.2f}'
            )
        lines.append(line)
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write one PSTH stack of the top-BPS neurons of one or two models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='78', choices=SESSION_EIDS)
    parser.add_argument(
        '--model',
        nargs='+',
        default=['full'],
        choices=MODELS,
        help='one or two models, e.g. --model full dino',
    )
    parser.add_argument('--k', type=int, default=6, choices=PCA_KS)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num-neurons', type=int, default=4, help='top-BPS neurons to stack')
    parser.add_argument(
        '--neuron-indices',
        type=int,
        nargs='+',
        default=None,
        help='neurons from top to bottom (default: top-BPS neurons of the last --model)',
    )
    parser.add_argument(
        '--trial-idx',
        type=parse_trials_idx_arg,
        default=None,
        metavar='N | K,N',
        help='0-based trials: N alone = first N trials; two values = indices K..N inclusive',
    )
    add_ranking_args(parser)
    parser.add_argument('--no-normalize', action='store_true', help='plot raw PSTH values')
    parser.add_argument('--width', type=float, default=1.0, help='figure width in inches')
    parser.add_argument('--height', type=float, default=1.6, help='figure height in inches')
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output', type=Path, default=None)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    spikes = load_session_spikes(SESSION_EIDS[args.session], args.neural_dir)
    if len(args.model) > 2:
        parser.error('--model takes one or two models')
    # the held-out trials, and so the ground truth, are the same for every model
    gt, preds = None, []
    for model in args.model:
        _, gt, pred = load_ridge_predictions(args.session, model, args.k, args.fold, spikes)
        preds.append(pred)
    t0, t1 = trials_idx_to_bounds(gt.shape[0], args.trial_idx)
    gt, preds = gt[t0:t1], [pred[t0:t1] for pred in preds]

    neuron_indices = args.neuron_indices
    psth_corr, modulation = None, None
    if neuron_indices is None and args.ranking:
        path = ranking_path(
            args.session,
            args.model[-1],
            args.k,
            args.fold,
            args.ranking_dir or RANKING_DIR,
        )
        ranking = load_ranking(
            path,
            {'session': args.session, 'model': args.model[-1], 'k': args.k, 'fold': args.fold},
            gt.shape[2],
        )
        min_psth_corr, min_modulation = thresholds_from_args(args)
        neuron_indices = select_top_neurons(
            ranking,
            args.num_neurons,
            min_psth_corr=min_psth_corr,
            min_modulation=min_modulation,
        )
        if not neuron_indices:
            parser.error(
                f'no neuron of {path} has psth r >= {min_psth_corr} and modulation >= '
                f'{min_modulation}, lower the thresholds or pass --no-filter'
            )
        by_neuron = ranking.set_index('neuron_idx').sort_index()
        psth_corr = by_neuron['psth_corr'].to_numpy()
        modulation = by_neuron['modulation'].to_numpy()
        print(f'top neurons of {args.model[-1]} from {path}: {neuron_indices}')
    elif neuron_indices is None:
        bps = bps_per_neuron(gt, preds[-1])
        ranked = np.argsort(np.nan_to_num(bps, nan=-np.inf))[::-1]
        neuron_indices = [int(i) for i in ranked[: args.num_neurons]]
        print(f'top-BPS neurons of {args.model[-1]}: {neuron_indices}')

    save_path = args.output or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_psth_stack',
        f'psth_stack_{args.session}_{"_".join(args.model)}_k{args.k}_fold{args.fold}.png',
    )
    bps_text = format_bps_text(
        neuron_indices,
        [bps_per_neuron(gt, pred) for pred in preds],
        args.model,
        session=args.session,
        k=args.k,
        fold=args.fold,
        psth_corr=psth_corr,
        modulation=modulation,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.with_suffix('.txt').write_text(bps_text)
    print(bps_text)
    plot_psth_stack(
        gt,
        preds,
        neuron_indices,
        colors=[model_color(model) for model in args.model],
        normalize=not args.no_normalize,
        figsize=(args.width, args.height),
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )
    print(save_path)
    print(save_path.with_suffix('.txt'))


if __name__ == '__main__':
    main()
