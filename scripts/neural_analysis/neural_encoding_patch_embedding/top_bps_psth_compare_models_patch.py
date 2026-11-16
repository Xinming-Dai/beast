#!/usr/bin/env python3
"""PSTH panels and activity raster of two patch-embedding ridge encoders for one session.

Patch-embedding counterpart of
``neural_encoding_cls_token/top_bps_psth_compare_models_figure3.py``: PSTH panels of the top-BPS
neurons (ground truth vs. both models) and a raster row (ground truth, model 1, model 2) for one
neuron. Held-out predictions of one fold are scored on the 48 time bins ridge predicts. Fonts,
colors and BPS come from ``plot_helpers.py`` and ``patch_helpers.py``.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/top_bps_psth_compare_models_patch.py \
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
from scripts.neural_analysis.neuron_ranking_and_filtering.rank_neurons_bps_corr_patch import (  # noqa: E402
    RANKING_DIR,
    ranking_path,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (  # noqa: E402
    add_ranking_args,
    neurons_from_ranking,
    thresholds_from_args,
)
from scripts.neural_analysis.plot_helpers import (  # noqa: E402
    AXIS_LABEL_FONT_KWARGS,
    AXIS_TICK_LABEL_FONT_KWARGS,
    PANEL_LABEL_FONT_KWARGS,
    _default_output_path_helper,
    bps_per_neuron,
)

GROUND_TRUTH_COLOR = 'black'


def _psth(trace: np.ndarray) -> np.ndarray:
    """Average over trials, ignoring NaNs."""
    return np.nanmean(np.asarray(trace, dtype=np.float64), axis=0)


def parse_trials_idx_arg(s: str) -> tuple[int, ...]:
    """Parse ``--trial-idx`` as one int (trial count) or two ints (first/last 0-based index)."""
    parts = [p for p in s.replace(',', ' ').split() if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('--trial-idx: expected N or K,N')
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError('--trial-idx: values must be integers') from e
    if len(nums) > 2:
        raise argparse.ArgumentTypeError('--trial-idx: pass one number (N) or two (K, N)')
    return nums


def trials_idx_to_bounds(
    n_trials_full: int, trials_idx: tuple[int, ...] | None
) -> tuple[int, int]:
    """Map 0-based trial selection to ``(start, stop)``; same semantics as ``viz_single_cell.py``."""
    if trials_idx is None:
        return 0, n_trials_full
    if len(trials_idx) == 1:
        n = int(trials_idx[0])
        if n < 1:
            raise IndexError('--trial-idx N: N must be >= 1 (count of trials from index 0)')
        if n > n_trials_full:
            raise IndexError(f'--trial-idx N={n} exceeds n_trials={n_trials_full}')
        return 0, n
    k, last = int(trials_idx[0]), int(trials_idx[1])
    if k < 0 or last < k:
        raise IndexError(f'--trial-idx K,N: need 0 <= K <= N, got K={k}, N={last}')
    if last >= n_trials_full:
        raise IndexError(f'--trial-idx upper index {last} >= n_trials={n_trials_full}')
    start, stop = k, last + 1
    if stop <= start or stop > n_trials_full:
        raise IndexError(f'Trial slice [{start}:{stop}) invalid for n_trials={n_trials_full}')
    return start, stop


def _apply_axis_tick_font(ax: plt.Axes) -> None:
    ax.tick_params(axis='both', labelsize=AXIS_TICK_LABEL_FONT_KWARGS['fontsize'])
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), **AXIS_TICK_LABEL_FONT_KWARGS)


def _normalize_traces(traces: list[np.ndarray]) -> list[np.ndarray]:
    finite_chunks = [t[np.isfinite(t)] for t in traces if np.any(np.isfinite(t))]
    if not finite_chunks:
        return traces
    finite_values = np.concatenate(finite_chunks)
    if finite_values.size == 0:
        return traces
    lo = float(np.nanmin(finite_values))
    hi = float(np.nanmax(finite_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [np.zeros_like(t, dtype=np.float64) for t in traces]
    return [(t - lo) / (hi - lo) for t in traces]


def _add_bps_label(
    ax: plt.Axes,
    bps1: float,
    bps2: float,
    colors: tuple[str, str],
) -> None:
    text = HPacker(
        children=[
            TextArea('BPS: ', textprops={'color': 'black', 'size': 10}),
            TextArea(f'{bps1:.2f}', textprops={'color': colors[0], 'size': 10}),
            TextArea(' vs. ', textprops={'color': 'black', 'size': 10}),
            TextArea(f'{bps2:.2f}', textprops={'color': colors[1], 'size': 10}),
        ],
        align='center',
        pad=0,
        sep=1,
    )
    box = AnchoredOffsetbox(
        loc='upper right',
        child=text,
        pad=0,
        frameon=False,
        bbox_to_anchor=(1.0, 1.12),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    ax.add_artist(box)


def _add_single_bps_label(ax: plt.Axes, bps: float, value_color: str) -> None:
    text = HPacker(
        children=[
            TextArea('BPS: ', textprops={'color': 'black', 'size': 10}),
            TextArea(f'{bps:.2f}', textprops={'color': value_color, 'size': 10}),
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


def _normalized_activity_raster(activity: np.ndarray) -> np.ndarray:
    activity = np.asarray(activity, dtype=np.float64)
    mean = np.nanmean(activity, axis=0, keepdims=True)
    centered = activity - mean
    scale = np.nanpercentile(np.abs(centered), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(centered / scale, -1.0, 1.0)


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
        fontsize=10,
        color='black',
        clip_on=False,
    )
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_xticks([float(time_s[0]), float(time_s[-1])])
    ax.set_xticklabels(['0', f'{time_s[-1]:.1f}'])
    ax.set_xlabel('Time (s)', labelpad=0, **AXIS_LABEL_FONT_KWARGS)
    if show_ylabel:
        ax.set_ylabel('Trial Index', **AXIS_LABEL_FONT_KWARGS)
        ax.set_yticks([trial_start, trial_stop])
    else:
        ax.set_yticks([])
    ax.tick_params(axis='both', width=1.0, length=3, pad=1)
    _apply_axis_tick_font(ax)
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(1.0)
    return im


def plot_top_bps_psth_compare(
    gt1: np.ndarray,
    pred1: np.ndarray,
    gt2: np.ndarray,
    pred2: np.ndarray,
    methods: tuple[str, str],
    neuron_indices: list[int],
    *,
    duration_s: float = 1.0,
    method1_label: str | None = None,
    method2_label: str | None = None,
    method_colors: tuple[str, str],
    panel_label: str | None = 'D',
    trials_idx: tuple[int, ...] | None = None,
    raster_neuron_index: int | None = None,
    normalize: bool = True,
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Plot PSTH panels and an activity-raster row from in-memory gt/pred arrays.

    Args:
        gt1: ground truth of method 1, shape (trials, time, neurons).
        pred1: predicted rates of method 1, same shape as ``gt1``.
        gt2: ground truth of method 2, same shape as ``gt1``.
        pred2: predicted rates of method 2, same shape as ``gt1``.
        methods: method names, used as labels when no label is given.
        neuron_indices: neurons shown as PSTH panels.
        duration_s: length of the time axis in seconds.
        method1_label: display label of method 1.
        method2_label: display label of method 2.
        method_colors: colors of method 1 and method 2.
        panel_label: panel letter, ``None`` to omit.
        trials_idx: trial subset, see :func:`trials_idx_to_bounds`.
        raster_neuron_index: neuron of the raster row, default the first PSTH neuron.
        normalize: min-max normalize the PSTH traces.
        save_path: output image path.
        also_save_pdf: also write a PDF next to ``save_path``.
        show: open an interactive window.

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
    for neuron_idx in neuron_indices:
        if not (0 <= neuron_idx < n_neurons):
            raise IndexError(f'neuron index {neuron_idx} out of range for n_neurons={n_neurons}')
    raster_idx = neuron_indices[0] if raster_neuron_index is None else int(raster_neuron_index)
    if not (0 <= raster_idx < n_neurons):
        raise IndexError(
            f'raster neuron index {raster_idx} out of range for n_neurons={n_neurons}'
        )

    bps1 = bps_per_neuron(gt1, pred1)
    bps2 = bps_per_neuron(gt2, pred2)
    n_panels = len(neuron_indices)
    fig_width = max(2.0 * n_panels, 3.0)
    fig = plt.figure(figsize=(fig_width, 4.2), dpi=200)
    grid = fig.add_gridspec(
        2,
        12,
        height_ratios=(1.0, 1.0),
        hspace=0.78,
        wspace=0.45,
    )
    col_edges = np.linspace(0, 12, n_panels + 1).round().astype(int)
    axes_flat = [
        fig.add_subplot(grid[0, col_edges[k] : col_edges[k + 1]]) for k in range(n_panels)
    ]
    time_s = np.linspace(0.0, float(duration_s), gt1.shape[1])

    for ax, neuron_idx in zip(axes_flat, neuron_indices, strict=True):
        gt_trace = _psth(gt1[:, :, neuron_idx])
        pred1_trace = _psth(pred1[:, :, neuron_idx])
        pred2_trace = _psth(pred2[:, :, neuron_idx])
        if normalize:
            gt_trace, pred1_trace, pred2_trace = _normalize_traces(
                [gt_trace, pred1_trace, pred2_trace]
            )

        ax.plot(time_s, gt_trace, color=GROUND_TRUTH_COLOR, linewidth=1.1)
        ax.plot(time_s, pred1_trace, color=method_colors[0], linewidth=3.0, alpha=0.9)
        ax.plot(time_s, pred2_trace, color=method_colors[1], linewidth=2.5)
        _add_bps_label(
            ax,
            float(bps1[neuron_idx]),
            float(bps2[neuron_idx]),
            method_colors,
        )

        ax.set_xlim(0.0, float(duration_s))
        ax.set_xticks([0.0, float(duration_s)])
        ax.set_xticklabels(['0', f'{duration_s:.1f}'])
        ax.tick_params(axis='both', width=1.2, length=3, pad=1)
        _apply_axis_tick_font(ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.set_yticks([0.0, 1.0] if normalize else [])

    axes_flat[0].set_ylabel(
        'Normalized\nFiring Rate' if normalize else 'Firing Rate',
        **AXIS_LABEL_FONT_KWARGS,
    )
    for ax in axes_flat[1:]:
        ax.tick_params(axis='y', left=False, labelleft=False)
        ax.spines['left'].set_visible(False)

    label1 = method1_label or methods[0]
    label2 = method2_label or methods[1]
    raster_axes = [
        fig.add_subplot(grid[1, 0:4]),
        fig.add_subplot(grid[1, 4:8]),
        fig.add_subplot(grid[1, 8:12]),
    ]
    im = _plot_activity_raster(
        raster_axes[0],
        gt1[:, :, raster_idx],
        method_name='Ground Truth',
        time_s=time_s,
        trial_start=t0,
        show_ylabel=True,
    )
    _plot_activity_raster(
        raster_axes[1],
        pred1[:, :, raster_idx],
        method_name=label1,
        time_s=time_s,
        trial_start=t0,
        bps=float(bps1[raster_idx]),
        bps_color=method_colors[0],
    )
    _plot_activity_raster(
        raster_axes[2],
        pred2[:, :, raster_idx],
        method_name=label2,
        time_s=time_s,
        trial_start=t0,
        bps=float(bps2[raster_idx]),
        bps_color=method_colors[1],
    )
    cax = fig.add_axes((0.935, 0.17, 0.012, 0.26))
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_ticks([-1.0, 0.0, 1.0])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=1)

    label_colors = (GROUND_TRUTH_COLOR, *method_colors)
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=0, label=text)
        for color, text in zip(label_colors, ('Ground Truth', label1, label2))
    ]
    legend = fig.legend(
        handles=handles,
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        frameon=False,
        ncol=3,
        fontsize=10,
        handlelength=0,
        handletextpad=0,
        columnspacing=0.9,
    )
    for text, color in zip(legend.get_texts(), label_colors):
        text.set_color(color)
    if panel_label:
        fig.text(0.005, 0.98, panel_label, **PANEL_LABEL_FONT_KWARGS, va='top', ha='left')

    fig.subplots_adjust(left=0.08, right=0.90, top=0.82, bottom=0.14, wspace=0.45, hspace=0.95)
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
    """Write one PSTH and raster figure comparing two models."""
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
        '--neuron-indices',
        type=int,
        nargs='+',
        default=None,
        help='PSTH neurons (default: top 4 by BPS of MODEL2)',
    )
    parser.add_argument('--raster-neuron-index', type=int, default=None)
    add_ranking_args(parser)
    parser.add_argument(
        '--trial-idx',
        type=parse_trials_idx_arg,
        default=None,
        metavar='N | K,N',
        help='0-based trials: N alone = first N trials; two values = indices K..N inclusive',
    )
    parser.add_argument('--no-normalize', action='store_true', help='plot raw PSTH values')
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

    neuron_indices = args.neuron_indices
    if neuron_indices is None and args.ranking:
        path = ranking_path(args.session, model2, args.k, args.fold, args.ranking_dir or RANKING_DIR)
        neuron_indices = neurons_from_ranking(
            args,
            4,
            path=path,
            expected={'session': args.session, 'model': model2, 'k': args.k, 'fold': args.fold},
            n_neurons=gt2.shape[2],
        )
        if not neuron_indices:
            min_psth_corr, min_modulation = thresholds_from_args(args)
            parser.error(
                f'no neuron of {path} has psth r >= {min_psth_corr} and modulation >= '
                f'{min_modulation}, lower the thresholds or pass --no-filter'
            )
        print(f'top neurons of {model2} from {path}: {neuron_indices}')
    elif neuron_indices is None:
        bps2 = bps_per_neuron(gt2, pred2)
        neuron_indices = [int(i) for i in np.argsort(np.nan_to_num(bps2, nan=-np.inf))[::-1][:4]]
        print(f'top-BPS neurons of {model2}: {neuron_indices}')

    save_path = args.output or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_psth_raster',
        f'psth_raster_{args.session}_{model1}_vs_{model2}_k{args.k}_fold{args.fold}.png',
    )
    plot_top_bps_psth_compare(
        gt1,
        pred1,
        gt2,
        pred2,
        (MODEL_LABELS[model1], MODEL_LABELS[model2]),
        neuron_indices,
        duration_s=RIDGE_DURATION_S,
        method_colors=(model_color(model1), model_color(model2)),
        panel_label=args.panel_label,
        trials_idx=args.trial_idx,
        raster_neuron_index=args.raster_neuron_index,
        normalize=not args.no_normalize,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )
    print(save_path)


if __name__ == '__main__':
    main()
