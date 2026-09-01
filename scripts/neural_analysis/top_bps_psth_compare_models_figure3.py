#!/usr/bin/env python3
"""Figure 3 PSTH panels plus an activity-raster row, comparing two methods for one EID.

This is ``top_bps_psth_compare_models_figure3.py`` (the PSTH row of
``psth_activity_raster_figure3.py``, with a "BPS: X vs. Y" label per panel) with the
activity-raster row from ``psth_activity_raster_figure3.py`` added back below it. The
expected folder layout matches the other Figure 3 scripts:

```
figure3/
├── ResNet/
│   └── <eid>/encoding_results*.npy
└── CLS/
    └── <eid>/encoding_results*.npy
```

Example:

```
python scripts/neural_analysis/top_bps_raster_compare_models_figure3.py \
  --results-dir /path/to/figure3 \
  --eid 4b00df29-3769-43be-bb40-128b1cba6d35 \
  --methods ResNet CLS \
  --neuron-indices 0 149 435 478 \
  --method1-label ResNet \
  --method2-label CLS
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from scripts.neural_analysis.plot_helpers import (
    AXIS_LABEL_FONT_KWARGS,
    AXIS_TICK_LABEL_FONT_KWARGS,
    EID_SET,
    METHOD_LIST,
    PANEL_LABEL_FONT_KWARGS,
    _default_output_path_helper,
    bps_per_neuron,
    iter_eid_encoding_npys,
    normalize_eid,
)


RESULTS_DIR = Path("/projects/bfsr/xdai3/project3d/iclr_plotting/SABLE_zero_shot_encoding")
GROUND_TRUTH_COLOR = "black"
METHOD1_COLOR = "0.75"
METHOD2_COLOR = "#F28E2B"


def _encoding_dict(path: Path) -> dict:
    """Load a dict-like encoding result from ``.npy`` or simple ``.npz`` wrappers."""
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.lib.npyio.NpzFile):
        try:
            if len(raw.files) == 1:
                arr = raw[raw.files[0]]
                inner = arr.item() if arr.dtype == object and arr.shape == () else arr
            else:
                inner = {key: raw[key] for key in raw.files}
        finally:
            raw.close()
    else:
        inner = raw.item() if raw.dtype == object and raw.shape == () else raw
    if not isinstance(inner, dict):
        raise TypeError(f"Expected dict in {path}, got {type(inner).__name__}")
    return inner


def _resolve_eid_key(root: dict, eid: str) -> str:
    want = normalize_eid(eid)
    for key in root:
        if normalize_eid(key) == want:
            return str(key)
    raise KeyError(f"EID {eid!r} not found in {list(root.keys())}")


def _gt_pred_for_encoder(path: Path, eid: str, encoder: str) -> tuple[np.ndarray, np.ndarray]:
    root = _encoding_dict(path)
    eid_key = _resolve_eid_key(root, eid)
    session = root[eid_key]
    if not isinstance(session, dict):
        raise TypeError(f"Expected dict for EID {eid_key!r}, got {type(session).__name__}")
    if encoder not in session:
        raise KeyError(
            f"Encoder {encoder!r} not found under EID {eid_key!r}; keys: {list(session.keys())}"
        )
    block = session[encoder]
    if not isinstance(block, dict):
        raise TypeError(f"Expected dict for encoder {encoder!r}, got {type(block).__name__}")
    gt = np.asarray(block["gt"], dtype=np.float64)
    pred = np.asarray(block["pred"], dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"gt/pred must match 3D (trials,time,neurons); got {gt.shape}, {pred.shape}")
    return gt, pred


def encoding_result_for_method(results_dir: Path, method: str, eid: str) -> Path:
    """Find an ``encoding_results*.npy`` file for one method/EID."""
    method_dir = results_dir.expanduser().resolve() / method
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {method_dir}")
    matches = sorted(iter_eid_encoding_npys(method_dir, frozenset({normalize_eid(eid)})))
    if not matches:
        raise FileNotFoundError(
            f"No encoding_results*.npy under {method_dir} in a directory named {eid!r}"
        )
    return matches[0]


def _psth(trace: np.ndarray) -> np.ndarray:
    """Average over trials, ignoring NaNs."""
    return np.nanmean(np.asarray(trace, dtype=np.float64), axis=0)


def parse_trials_idx_arg(s: str) -> tuple[int, ...]:
    """Parse ``--trial-idx`` as one int (trial count) or two ints (first/last 0-based index)."""
    parts = [p for p in s.replace(",", " ").split() if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--trial-idx: expected N or K,N")
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError("--trial-idx: values must be integers") from e
    if len(nums) > 2:
        raise argparse.ArgumentTypeError("--trial-idx: pass one number (N) or two (K, N)")
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
            raise IndexError("--trial-idx N: N must be >= 1 (count of trials from index 0)")
        if n > n_trials_full:
            raise IndexError(f"--trial-idx N={n} exceeds n_trials={n_trials_full}")
        return 0, n
    k, last = int(trials_idx[0]), int(trials_idx[1])
    if k < 0 or last < k:
        raise IndexError(f"--trial-idx K,N: need 0 <= K <= N, got K={k}, N={last}")
    if last >= n_trials_full:
        raise IndexError(f"--trial-idx upper index {last} >= n_trials={n_trials_full}")
    start, stop = k, last + 1
    if stop <= start or stop > n_trials_full:
        raise IndexError(f"Trial slice [{start}:{stop}) invalid for n_trials={n_trials_full}")
    return start, stop


def _apply_axis_tick_font(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=AXIS_TICK_LABEL_FONT_KWARGS["fontsize"])
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


def _add_bps_label(ax: plt.Axes, bps1: float, bps2: float) -> None:
    text = HPacker(
        children=[
            TextArea("BPS: ", textprops={"color": "black", "size": 10}),
            TextArea(f"{bps1:.2f}", textprops={"color": METHOD1_COLOR, "size": 10}),
            TextArea(" vs. ", textprops={"color": "black", "size": 10}),
            TextArea(f"{bps2:.2f}", textprops={"color": METHOD2_COLOR, "size": 10}),
        ],
        align="center",
        pad=0,
        sep=1,
    )
    box = AnchoredOffsetbox(
        loc="upper right",
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
            TextArea("BPS: ", textprops={"color": "black", "size": 10}),
            TextArea(f"{bps:.2f}", textprops={"color": value_color, "size": 10}),
        ],
        align="center",
        pad=0,
        sep=1,
    )
    box = AnchoredOffsetbox(
        loc="lower left",
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
    bps_color: str = "black",
    show_ylabel: bool = False,
) -> plt.AxesImage:
    raster = _normalized_activity_raster(activity)
    trial_stop = trial_start + activity.shape[0] - 1
    im = ax.imshow(
        raster,
        aspect="auto",
        cmap="bwr",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
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
        ha="right",
        va="bottom",
        fontsize=10,
        color="black",
        clip_on=False,
    )
    ax.set_xlim(float(time_s[0]), float(time_s[-1]))
    ax.set_xticks([float(time_s[0]), float(time_s[-1])])
    ax.set_xticklabels(["0", f"{time_s[-1]:.1f}"])
    ax.set_xlabel("Time (s)", labelpad=0, **AXIS_LABEL_FONT_KWARGS)
    if show_ylabel:
        ax.set_ylabel("Trial Index", **AXIS_LABEL_FONT_KWARGS)
        ax.set_yticks([trial_start, trial_stop])
    else:
        ax.set_yticks([])
    ax.tick_params(axis="both", width=1.0, length=3, pad=1)
    _apply_axis_tick_font(ax)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_linewidth(1.0)
    return im


def plot_top_bps_raster_compare_models(
    results_dir: Path,
    eid: str,
    methods: tuple[str, str],
    neuron_indices: list[int],
    *,
    encoder: str = "cnn",
    duration_s: float = 1.0,
    method1_label: str | None = None,
    method2_label: str | None = None,
    panel_label: str | None = "D",
    trials_idx: tuple[int, ...] | None = None,
    raster_neuron_index: int | None = None,
    normalize: bool = True,
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
) -> plt.Figure:
    """Plot PSTH panels comparing two methods, plus an activity-raster row for one neuron."""
    path1 = encoding_result_for_method(results_dir, methods[0], eid)
    path2 = encoding_result_for_method(results_dir, methods[1], eid)

    gt1, pred1 = _gt_pred_for_encoder(path1, eid, encoder)
    gt2, pred2 = _gt_pred_for_encoder(path2, eid, encoder)
    if gt1.shape != pred1.shape or gt2.shape != pred2.shape:
        raise ValueError("Internal gt/pred shape mismatch")
    if gt1.shape != gt2.shape:
        raise ValueError(f"Method shapes do not match: {gt1.shape} vs {gt2.shape}")
    if not np.allclose(gt1, gt2, equal_nan=True):
        warnings.warn("Ground-truth arrays differ between methods; plotting GT from method1", stacklevel=2)

    t0, t1 = trials_idx_to_bounds(gt1.shape[0], trials_idx)
    gt1 = gt1[t0:t1]
    gt2 = gt2[t0:t1]
    pred1 = pred1[t0:t1]
    pred2 = pred2[t0:t1]

    n_neurons = gt1.shape[2]
    for neuron_idx in neuron_indices:
        if not (0 <= neuron_idx < n_neurons):
            raise IndexError(f"neuron index {neuron_idx} out of range for n_neurons={n_neurons}")
    raster_idx = neuron_indices[0] if raster_neuron_index is None else int(raster_neuron_index)
    if not (0 <= raster_idx < n_neurons):
        raise IndexError(f"raster neuron index {raster_idx} out of range for n_neurons={n_neurons}")

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
        fig.add_subplot(grid[0, col_edges[k] : col_edges[k + 1]])
        for k in range(n_panels)
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
        ax.plot(time_s, pred1_trace, color=METHOD1_COLOR, linewidth=3.0, alpha=0.9)
        ax.plot(time_s, pred2_trace, color=METHOD2_COLOR, linewidth=2.5)
        _add_bps_label(ax, float(bps1[neuron_idx]), float(bps2[neuron_idx]))

        ax.set_xlim(0.0, float(duration_s))
        ax.set_xticks([0.0, float(duration_s)])
        ax.set_xticklabels(["0", f"{duration_s:.1f}"])
        ax.tick_params(axis="both", width=1.2, length=3, pad=1)
        _apply_axis_tick_font(ax)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.set_yticks([0.0, 1.0] if normalize else [])

    axes_flat[0].set_ylabel(
        "Normalized\nFiring Rate" if normalize else "Firing Rate",
        **AXIS_LABEL_FONT_KWARGS,
    )
    for ax in axes_flat[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)

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
        method_name="Ground Truth",
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
        bps_color=METHOD1_COLOR,
    )
    _plot_activity_raster(
        raster_axes[2],
        pred2[:, :, raster_idx],
        method_name=label2,
        time_s=time_s,
        trial_start=t0,
        bps=float(bps2[raster_idx]),
        bps_color=METHOD2_COLOR,
    )
    cax = fig.add_axes((0.935, 0.17, 0.012, 0.26))
    colorbar = fig.colorbar(im, cax=cax)
    colorbar.set_ticks([-1.0, 0.0, 1.0])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=1)

    label_colors = (GROUND_TRUTH_COLOR, METHOD1_COLOR, METHOD2_COLOR)
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=0, label=text)
        for color, text in zip(label_colors, ("Ground Truth", label1, label2))
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper right",
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
        fig.text(0.005, 0.98, panel_label, **PANEL_LABEL_FONT_KWARGS, va="top", ha="left")

    fig.subplots_adjust(left=0.08, right=0.90, top=0.82, bottom=0.14, wspace=0.45, hspace=0.95)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        if also_save_pdf:
            pdf_path = save_path.with_suffix(".pdf")
            if pdf_path.resolve() != save_path.resolve():
                fig.savefig(pdf_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Root directory containing one subfolder per method",
    )
    p.add_argument(
        "--eid",
        required=True,
        choices=sorted(EID_SET),
        help="EID/session id to plot",
    )
    p.add_argument(
        "--methods",
        nargs=2,
        required=True,
        choices=METHOD_LIST,
        metavar=("METHOD1", "METHOD2"),
        help="Two method folder names. METHOD1 is grey, METHOD2 is green.",
    )
    p.add_argument(
        "--neuron-indices",
        nargs="+",
        type=int,
        required=True,
        help="Neuron indices to plot, one panel per neuron",
    )
    p.add_argument(
        "--encoder",
        choices=("cnn", "rrr"),
        default="cnn",
        help="Encoder block inside encoding_results*.npy (default: cnn)",
    )
    p.add_argument("--duration-s", type=float, default=1.0, help="Right x-axis time in seconds")
    p.add_argument("--method1-label", type=str, default=None, help="Legend label for grey method")
    p.add_argument("--method2-label", type=str, default=None, help="Legend label for green method")
    p.add_argument("--panel-label", type=str, default=None, help="Panel letter (empty to omit)")
    p.add_argument("--no-normalize", action="store_true", help="Plot raw PSTH values instead of 0-1 normalized traces")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Save figure here (default: <results-dir>/top_bps_raster_compare_models_figure3_"
            "<timestamp>/top_bps_raster_compare_models_figure3.png)"
        ),
    )
    p.add_argument("--save-pdf", action="store_true", help="Also save a PDF next to the PNG")
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    p.add_argument(
        "--trial-idx",
        type=parse_trials_idx_arg,
        default=None,
        metavar="N | K,N",
        help=(
            "0-based trials: N alone = first N trials (indices 0..N-1); "
            "two values = indices K..N inclusive. Default: all trials."
        ),
    )
    p.add_argument(
        "--raster-neuron-index",
        type=int,
        default=None,
        help="Neuron index for the activity raster row (default: first --neuron-indices value)",
    )
    args = p.parse_args()

    save_path = args.output
    if save_path is None:
        save_path = _default_output_path_helper(
            args.results_dir,
            "top_bps_raster_compare_models_figure3",
            "top_bps_raster_compare_models_figure3.png",
        )

    plot_top_bps_raster_compare_models(
        args.results_dir,
        args.eid,
        tuple(args.methods),
        list(args.neuron_indices),
        encoder=args.encoder,
        duration_s=args.duration_s,
        method1_label=args.method1_label,
        method2_label=args.method2_label,
        panel_label=args.panel_label or None,
        trials_idx=args.trial_idx,
        raster_neuron_index=args.raster_neuron_index,
        normalize=not args.no_normalize,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
