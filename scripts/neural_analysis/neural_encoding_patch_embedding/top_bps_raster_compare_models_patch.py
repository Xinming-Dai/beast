#!/usr/bin/env python3
"""Activity raster of two patch-embedding ridge encoders for one neuron of one session.

Patch-embedding counterpart of
``neural_encoding_cls_token/top_bps_raster_compare_models_figure3.py``: three raster panels
(ground truth, model 1, model 2) of one neuron. This is the raster row of
``top_bps_psth_compare_models_patch.py`` without the PSTH row, drawn larger. Held-out predictions
of one fold are scored on the 48 time bins ridge predicts.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/top_bps_raster_compare_models_patch.py \
  --session 78 --models dino full --k 6 --fold 0 --no-show
```
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    MODELS,
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    RIDGE_DURATION_S,
    SESSION_EIDS,
    load_ridge_predictions,
    load_session_spikes,
    model_color,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.top_bps_psth_compare_models_patch import (  # noqa: E402
    _normalized_activity_raster,
    parse_trials_idx_arg,
    trials_idx_to_bounds,
)
from scripts.neural_analysis.plot_helpers import (  # noqa: E402
    PANEL_LABEL_FONT_KWARGS,
    _default_output_path_helper,
    bps_per_neuron,
)

TEXT_FONT_SIZE = 14


def _apply_axis_tick_font(ax: plt.Axes) -> None:
    ax.tick_params(axis='both', labelsize=TEXT_FONT_SIZE)
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), fontsize=TEXT_FONT_SIZE)


def _add_single_bps_label(ax: plt.Axes, bps: float, value_color: str) -> None:
    text = HPacker(
        children=[
            TextArea('BPS: ', textprops={'color': 'black', 'size': TEXT_FONT_SIZE}),
            TextArea(f'{bps:.2f}', textprops={'color': value_color, 'size': TEXT_FONT_SIZE}),
        ],
        align='center',
        pad=0,
        sep=1,
    )
    box = AnchoredOffsetbox(
        loc='lower left',
        child=text,
        pad=0,
        frameon=False,
        bbox_to_anchor=(0.0, 1.04),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(box)


def _plot_activity_raster(
    ax: plt.Axes,
    activity: np.ndarray,
    *,
    method_name: str,
    time_s: np.ndarray,
    trial_start: int,
    bps: float | None = None,
    bps_color: str = 'black',
    show_ylabel: bool = False,
    show_xlabel: bool = True,
) -> AxesImage:
    raster = _normalized_activity_raster(activity)
    trial_stop = trial_start + activity.shape[0] - 1
    im = ax.imshow(
        raster,
        aspect='auto',
        cmap='bwr',
        vmin=-1.0,
        vmax=1.0,
        interpolation='nearest',
        extent=(
            float(time_s[0]),
            float(time_s[-1]),
            trial_stop + 0.5,
            trial_start - 0.5,
        ),
    )
    if bps is not None:
        _add_single_bps_label(ax, bps, bps_color)
    ax.text(
        1.0,
        1.04,
        method_name,
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=TEXT_FONT_SIZE,
        color='black',
        clip_on=False,
    )
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_xticks([float(time_s[0]), float(time_s[-1])])
    if show_xlabel:
        ax.set_xticklabels(['0', f'{time_s[-1]:.1f}'])
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_yticks([trial_start, trial_stop])
    else:
        ax.set_yticks([])
    ax.tick_params(axis='both', width=1.0, length=3, pad=1)
    _apply_axis_tick_font(ax)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(1.0)
    return im


def plot_top_bps_raster_compare(
    gt1: np.ndarray,
    pred1: np.ndarray,
    gt2: np.ndarray,
    pred2: np.ndarray,
    methods: tuple[str, str],
    neuron_index: int,
    *,
    duration_s: float = 1.0,
    method1_label: str | None = None,
    method2_label: str | None = None,
    method_colors: tuple[str, str],
    panel_label: str | None = 'D',
    trials_idx: tuple[int, ...] | None = None,
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
    title_prefix: str = '',
) -> plt.Figure:
    """Plot a 3-panel activity raster from in-memory gt/pred arrays.

    Args:
        gt1: ground truth of method 1, shape (trials, time, neurons).
        pred1: predicted rates of method 1, same shape as ``gt1``.
        gt2: ground truth of method 2, same shape as ``gt1``.
        pred2: predicted rates of method 2, same shape as ``gt1``.
        methods: method names, used as labels when no label is given.
        neuron_index: neuron to plot.
        duration_s: length of the time axis in seconds.
        method1_label: display label of method 1.
        method2_label: display label of method 2.
        method_colors: colors of the BPS values of method 1 and method 2.
        panel_label: panel letter, ``None`` to omit.
        trials_idx: trial subset, see :func:`trials_idx_to_bounds`.
        save_path: output image path.
        also_save_pdf: also write a PDF next to ``save_path``.
        show: open an interactive window.
        title_prefix: text in front of the printed BPS summary, e.g. the session.

    Returns:
        The matplotlib figure.
    """
    if gt1.shape != pred1.shape or gt2.shape != pred2.shape:
        raise ValueError('Internal gt/pred shape mismatch')
    if gt1.shape != gt2.shape:
        raise ValueError(f'Method shapes do not match: {gt1.shape} vs {gt2.shape}')
    if not np.allclose(gt1, gt2, equal_nan=True):
        warnings.warn(
            'Ground-truth arrays differ between methods; plotting GT from method1', stacklevel=2
        )

    t0, t1 = trials_idx_to_bounds(gt1.shape[0], trials_idx)
    gt1 = gt1[t0:t1]
    gt2 = gt2[t0:t1]
    pred1 = pred1[t0:t1]
    pred2 = pred2[t0:t1]

    n_neurons = gt1.shape[2]
    if not (0 <= neuron_index < n_neurons):
        raise IndexError(f'neuron index {neuron_index} out of range for n_neurons={n_neurons}')

    bps1 = bps_per_neuron(gt1, pred1)
    bps2 = bps_per_neuron(gt2, pred2)
    time_s = np.linspace(0.0, float(duration_s), gt1.shape[1])
    label1 = method1_label or methods[0]
    label2 = method2_label or methods[1]
    print(
        f'[{title_prefix}] neuron {neuron_index} BPS: {label1}={bps1[neuron_index]:.2f}, '
        f'{label2}={bps2[neuron_index]:.2f}'
    )

    fig, axes = plt.subplots(3, 1, figsize=(3.0, 4.24), dpi=200)
    _plot_activity_raster(
        axes[0],
        gt1[:, :, neuron_index],
        method_name='Ground Truth',
        time_s=time_s,
        trial_start=t0,
        show_ylabel=True,
        show_xlabel=False,
    )
    _plot_activity_raster(
        axes[1],
        pred1[:, :, neuron_index],
        method_name=label1,
        time_s=time_s,
        trial_start=t0,
        bps=float(bps1[neuron_index]),
        bps_color=method_colors[0],
        show_ylabel=True,
        show_xlabel=False,
    )
    _plot_activity_raster(
        axes[2],
        pred2[:, :, neuron_index],
        method_name=label2,
        time_s=time_s,
        trial_start=t0,
        bps=float(bps2[neuron_index]),
        bps_color=method_colors[1],
        show_ylabel=True,
        show_xlabel=True,
    )
    if panel_label:
        fig.text(0.005, 0.98, panel_label, **PANEL_LABEL_FONT_KWARGS, va='top', ha='left')

    fig.subplots_adjust(left=0.20, right=0.95, top=0.94, bottom=0.10, hspace=0.55)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        if also_save_pdf:
            pdf_path = save_path.with_suffix('.pdf')
            if pdf_path.resolve() != save_path.resolve():
                fig.savefig(pdf_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    """Write one raster figure comparing two models on one neuron."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='78', choices=SESSION_EIDS)
    parser.add_argument(
        '--models',
        nargs=2,
        metavar=('MODEL1', 'MODEL2'),
        default=('dino', 'full'),
        choices=MODELS,
    )
    parser.add_argument('--k', type=int, default=6, choices=PCA_KS)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument(
        '--neuron-index',
        type=int,
        default=None,
        help='neuron to plot (default: the top-BPS neuron of MODEL2)',
    )
    parser.add_argument(
        '--trial-idx',
        type=parse_trials_idx_arg,
        default=None,
        metavar='N | K,N',
        help='0-based trials: N alone = first N trials; two values = indices K..N inclusive',
    )
    parser.add_argument('--panel-label', default=None)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    spikes = load_session_spikes(SESSION_EIDS[args.session], args.neural_dir)
    model1, model2 = args.models
    _, gt1, pred1 = load_ridge_predictions(args.session, model1, args.k, args.fold, spikes)
    _, gt2, pred2 = load_ridge_predictions(args.session, model2, args.k, args.fold, spikes)

    neuron_index = args.neuron_index
    if neuron_index is None:
        bps2 = bps_per_neuron(gt2, pred2)
        neuron_index = int(np.argmax(np.nan_to_num(bps2, nan=-np.inf)))
        print(f'top-BPS neuron of {model2}: {neuron_index}')

    save_path = args.output or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_raster',
        f'raster_{args.session}_{model1}_vs_{model2}_k{args.k}_fold{args.fold}.png',
    )
    plot_top_bps_raster_compare(
        gt1,
        pred1,
        gt2,
        pred2,
        (MODEL_LABELS[model1], MODEL_LABELS[model2]),
        neuron_index,
        duration_s=RIDGE_DURATION_S,
        method_colors=(model_color(model1), model_color(model2)),
        panel_label=args.panel_label,
        trials_idx=args.trial_idx,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
        title_prefix=f'session {args.session}',
    )
    print(save_path)


if __name__ == '__main__':
    main()
