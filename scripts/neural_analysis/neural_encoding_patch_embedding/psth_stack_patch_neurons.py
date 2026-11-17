#!/usr/bin/env python3
"""PSTH stack of hand-picked neurons of one session.

Counterpart of `psth_stack_patch_selected_vs_unselected.py` for neurons chosen by hand: pass the
session and the neuron indices and the stack is drawn in the order given, with the same
:func:`plot_psth_stack` and thin lines. Nothing is ranked or filtered, so no ranking file is read.

It writes `psth_stack_<session>_<models>_k<K>_fold<F>_neurons_<ids>.png` with a `.txt` next to it
that lists the csv BPS (mean over the five folds, from `neuron_selection_handoff/`) and the
`wins_vs_dino` of the stacked neurons. The PSTHs are those of one fold.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/psth_stack_patch_neurons.py \
  --session 72 --neuron-idx 61 65 162 119 --model dino full --k 6 --fold 0 --no-show

python scripts/neural_analysis/neural_encoding_patch_embedding/psth_stack_patch_neurons.py \
  --session 72 --neuron-idx 208 153 211 112 --model dino full --k 6 --fold 0 --no-show
```
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODELS,
    NEURAL_DIR,
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
    load_ridge_predictions,
    load_selection_records,
    load_session_spikes,
    model_color,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.psth_stack_patch import (  # noqa: E402
    format_bps_text,
    plot_psth_stack,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.psth_stack_patch_selected_vs_unselected import (  # noqa: E402, E501
    HEIGHT_PER_NEURON,
    LINE_SCALE,
    SELECTION_KS,
    load_wins_vs_dino,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.top_bps_psth_compare_models_patch import (  # noqa: E402, E501
    parse_trials_idx_arg,
    trials_idx_to_bounds,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

# wider than the 1.0 in of `psth_stack_patch.py`: 1.96 x 1.6 in (4 neurons) is the 328 x 268 px
# aspect ratio of the reference stack
FIGURE_WIDTH = 1.96


def check_neuron_indices(neuron_indices: list[int], n_neurons: int) -> None:
    """Check that the neurons to stack exist in the session.

    Args:
        neuron_indices: neurons from top to bottom.
        n_neurons: number of neurons of the session.

    Raises:
        ValueError: if an index is outside `range(n_neurons)`.
    """
    invalid = [idx for idx in neuron_indices if not 0 <= idx < n_neurons]
    if invalid:
        raise ValueError(
            f'neuron indices {invalid} are not in this session, which has neurons 0..{n_neurons - 1}'
        )


def main() -> None:
    """Write one PSTH stack of the given neurons of one session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='78', choices=SESSION_EIDS)
    parser.add_argument(
        '--neuron-idx',
        type=int,
        nargs='+',
        required=True,
        help='neurons from top to bottom, e.g. --neuron-idx 3 15 42 77',
    )
    parser.add_argument(
        '--model',
        nargs='+',
        default=['full'],
        choices=MODELS,
        help='one or two models, e.g. --model dino full',
    )
    parser.add_argument('--k', type=int, default=6, choices=SELECTION_KS)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument(
        '--trial-idx',
        type=parse_trials_idx_arg,
        default=None,
        metavar='N | K,N',
        help='0-based trials: N alone = first N trials; two values = indices K..N inclusive',
    )
    parser.add_argument('--selection-dir', type=Path, default=SELECTION_DIR)
    parser.add_argument('--no-normalize', action='store_true', help='plot raw PSTH values')
    parser.add_argument(
        '--width',
        type=float,
        default=FIGURE_WIDTH,
        help='figure width in inches',
    )
    parser.add_argument(
        '--height-per-neuron',
        type=float,
        default=HEIGHT_PER_NEURON,
        help='figure height in inches of every stacked neuron, so more neurons extend the figure',
    )
    parser.add_argument(
        '--line-scale',
        type=float,
        default=LINE_SCALE,
        help='factor on the line widths of `psth_stack_patch.py`, 1 for the default widths',
    )
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output', type=Path, default=None)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if len(args.model) > 2:
        parser.error('--model takes one or two models')

    spikes = load_session_spikes(SESSION_EIDS[args.session], args.neural_dir)
    # the held-out trials, and so the ground truth, are the same for every model
    gt, preds = None, []
    for model in args.model:
        _, gt, pred = load_ridge_predictions(args.session, model, args.k, args.fold, spikes)
        preds.append(pred)
    t0, t1 = trials_idx_to_bounds(gt.shape[0], args.trial_idx)
    gt, preds = gt[t0:t1], [pred[t0:t1] for pred in preds]

    neuron_indices = args.neuron_idx
    try:
        check_neuron_indices(neuron_indices, gt.shape[2])
    except ValueError as err:
        parser.error(str(err))

    records, _ = load_selection_records(args.session, args.k, args.selection_dir)
    bps_by_model = [records[('ridge', args.session, model, args.k, 0)].bps for model in args.model]
    wins_vs_dino = load_wins_vs_dino(args.session, args.k, args.selection_dir)

    bps_text = format_bps_text(
        neuron_indices,
        bps_by_model,
        args.model,
        session=args.session,
        k=args.k,
        fold=args.fold,
    )
    # the BPS is the csv mean over the folds, only the PSTHs belong to the fold
    bps_text = bps_text.replace(
        f', fold {args.fold}\n',
        f', mean BPS over folds, PSTHs of fold {args.fold}\n',
        1,
    )
    # one line per neuron follows the two header lines, in the order of `neuron_indices`
    lines = bps_text.splitlines()
    lines[2:] = [
        f'{line} | wins_vs_dino={int(wins_vs_dino[idx])}'
        for line, idx in zip(lines[2:], neuron_indices, strict=True)
    ]
    bps_text = '\n'.join(lines) + '\n'

    ids = '_'.join(str(idx) for idx in neuron_indices)
    save_path = args.output or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_psth_stack_neurons',
        f'psth_stack_{args.session}_{"_".join(args.model)}_k{args.k}_fold{args.fold}'
        f'_neurons_{ids}.png',
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
        figsize=(args.width, args.height_per_neuron * len(neuron_indices)),
        line_scale=args.line_scale,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )
    print(save_path)
    print(save_path.with_suffix('.txt'))


if __name__ == '__main__':
    main()
