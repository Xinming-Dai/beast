#!/usr/bin/env python3
"""SSIM and PSNR bar plots across decoding methods, aggregated over EIDs.

Folder structure:

```text
figure2/
├── resnet/
│   ├── <eid>/
│   │   └── psnr_ssim_metrics*.npz
├── beast/
...
```

Under each method folder, paths use :func:`iter_eid_metrics_npys` (recursive EID
directory discovery, matching any ``psnr_ssim_metrics*.npz`` per EID).

```bash
python scripts/neural_analysis/decoding_metrics_figure2.py \
  --results-dir /path/to/figure2 --methods resnet beast sable
```
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from scripts.neural_analysis.plot_helpers import (
    EID_SET,
    ENCODING_BASE_COLORS,
    _default_output_path_helper,
    normalize_eid,
)

# base figure size (width, height) in inches for a single bar chart; scale it with
# FIGURE_SIZE_SCALE below to resize the plots while keeping this width/height ratio.
BASE_FIGSIZE: tuple[float, float] = (2.8, 2.0)
FIGURE_SIZE_SCALE: float = 0.8

RESULTS_DIR = Path("/projects/bfsr/xdai3/project3d/iclr_plotting/ibl/SABLE_zero_shot_decoding_finetuned_masked")
DEFAULT_METHODS: list[str] = ["resnet", "beast", "sable"]
DEFAULT_METHOD_LABELS: list[str] = ["ResNet AE", "BEAST", "SABLE"]

BASELINE_METHOD: str = "mean" # drawn as a dashed horizontal line
BASELINE_LABEL: str = "Mean frame"
BASELINE_COLOR: str = ENCODING_BASE_COLORS["random_baseline"]
METRIC_BASE_COLORS: dict[str, str] = {
    "resnet": ENCODING_BASE_COLORS["resnet"],
    "beast": ENCODING_BASE_COLORS["beast"],
    "sable": ENCODING_BASE_COLORS["sable_dino"],
}


def _figsize() -> tuple[float, float]:
    width, height = BASE_FIGSIZE
    return width * FIGURE_SIZE_SCALE, height * FIGURE_SIZE_SCALE


def _allowed_eid_set(eids: list[str] | None) -> frozenset[str] | None:
    """None / empty = no name filter (any folder containing psnr_ssim_metrics*.npz)."""
    if not eids:
        return None
    return frozenset(normalize_eid(e) for e in eids)


def iter_eid_metrics_npys(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
):
    """Yield ``psnr_ssim_metrics*.npz`` paths for matching EID folders.

    If ``allowed_eids`` is set, only directories whose **name** is in that set (compared
    via :func:`normalize_eid`) are searched. Exactly one ``psnr_ssim_metrics*.npz`` file
    is expected under each matching EID directory. If ``allowed_eids`` is ``None``, every
    matching file under ``method_dir`` is included.
    """
    if not method_dir.is_dir():
        return
    if allowed_eids is None:
        yield from method_dir.rglob("psnr_ssim_metrics*.npz")
        return

    seen: set[Path] = set()
    for d in method_dir.rglob("*"):
        if not d.is_dir() or normalize_eid(d.name) not in allowed_eids:
            continue
        matches = sorted(met for met in d.rglob("psnr_ssim_metrics*.npz") if met.is_file())
        if len(matches) > 1:
            raise ValueError(
                f"Expected one psnr_ssim_metrics*.npz under EID directory {d}, "
                f"found {len(matches)}: {[str(p) for p in matches]}"
            )
        if matches and matches[0] not in seen:
            seen.add(matches[0])
            yield matches[0]


def _finite_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def aggregate_metrics_vectors_for_method(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect per-frame PSNR and SSIM vectors across all matching EIDs."""
    psnr_vectors: list[np.ndarray] = []
    ssim_vectors: list[np.ndarray] = []
    for npz in sorted(iter_eid_metrics_npys(method_dir, allowed_eids), key=lambda p: p.parent.name):
        metrics = np.load(npz)
        psnr_vectors.append(_finite_1d(metrics["psnr"]))
        ssim_vectors.append(_finite_1d(metrics["ssim"]))
    if not psnr_vectors:
        return None
    return np.concatenate(psnr_vectors), np.concatenate(ssim_vectors)


def _mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.nanmean(values))
    se = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, se


def collect_method_stats(
    results_dir: Path,
    methods: list[str],
    allowed_eids: frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (means_psnr, means_ssim, se_psnr, se_ssim), one entry per method (nan if missing)."""
    n = len(methods)
    means_psnr = np.full(n, np.nan)
    means_ssim = np.full(n, np.nan)
    se_psnr = np.full(n, np.nan)
    se_ssim = np.full(n, np.nan)

    for i, method in enumerate(methods):
        agg = aggregate_metrics_vectors_for_method(results_dir / method, allowed_eids)
        if agg is None:
            continue
        psnr, ssim = agg
        means_psnr[i], se_psnr[i] = _mean_and_standard_error(psnr)
        means_ssim[i], se_ssim[i] = _mean_and_standard_error(ssim)

    return means_psnr, means_ssim, se_psnr, se_ssim


@dataclass
class MetricPanel:
    """Data and axis settings of one metric panel (e.g. PSNR or SSIM).

    Attributes:
        y_label: y axis label.
        means: mean metric per method (nan entries are skipped).
        se: standard error per method.
        bottom_max: upper limit of the lower (compressed) axis of the broken y axis.
        decimals: decimals in the value labels above the bars.
        baseline_value: value of the dashed baseline line, or None for no line.
    """

    y_label: str
    means: np.ndarray
    se: np.ndarray
    bottom_max: float
    decimals: int = 2
    baseline_value: float | None = None


def _zoomed_y_range(
    values: np.ndarray,
    below: float = 0.6,
    above: float = 0.5,
) -> tuple[float, float]:
    """Y range of the zoomed (upper) axis, padded around the spread of ``values``.

    Padding is a fraction of the spread: ``below`` keeps the shortest bar visibly tall and
    ``above`` leaves room for the value labels.
    """
    span = max(float(values.max() - values.min()), 1e-6)
    return float(values.min()) - below * span, float(values.max()) + above * span


def _make_panel_axes(n_panels: int) -> tuple[plt.Figure, list[tuple[plt.Axes, plt.Axes]]]:
    """Create a figure with ``n_panels`` stacked broken-axis panels.

    Returns:
        The figure and one ``(upper_axis, lower_axis)`` pair per panel, top panel first.
    """
    width, height = _figsize()
    # extra panels add less than a full panel of height since the x labels are shared
    fig = plt.figure(figsize=(width, height * (1 + 0.8 * (n_panels - 1))), dpi=200)
    # margins are explicit because tight_layout ignores the figure-level ylabels
    outer = fig.add_gridspec(
        n_panels,
        1,
        left=0.32,
        right=0.98,
        top=0.9 if n_panels == 1 else 0.93,
        bottom=0.28 if n_panels == 1 else 0.17,
        hspace=0.2,
    )
    axes = []
    for i in range(n_panels):
        inner = outer[i].subgridspec(2, 1, height_ratios=[7, 1], hspace=0.08)
        axes.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))
    return fig, axes


def _draw_metric_panel(
    ax: plt.Axes,
    ax_bottom: plt.Axes,
    panel: MetricPanel,
    method_labels: list[str],
    colors: list[str],
    *,
    baseline_color: str,
    show_xticklabels: bool,
) -> None:
    """Draw one broken-axis bar panel.

    The y axis is broken: the short lower axis shows ``0`` to ``panel.bottom_max`` and the tall
    upper axis zooms into the range the methods occupy, so small differences stay visible.

    Args:
        ax: upper (zoomed) axis.
        ax_bottom: lower (compressed) axis.
        panel: data and settings of this panel.
        method_labels: tick label per bar.
        colors: bar color per method.
        baseline_color: color of the dashed baseline line.
        show_xticklabels: whether to show the method names under the lower axis.
    """
    n = len(method_labels)
    width = 0.55
    # spacing wider than width leaves a gap between bars so the value labels above
    # neighboring bars don't collide.
    bar_center_spacing = 1.3
    x = np.arange(n, dtype=np.float64) * bar_center_spacing
    means, se = panel.means, panel.se

    visible = means[np.isfinite(means)]
    if panel.baseline_value is not None and np.isfinite(panel.baseline_value):
        visible = np.append(visible, panel.baseline_value)
    y_lo, y_hi = _zoomed_y_range(visible) if visible.size else (0.0, 1.0)

    for i in range(n):
        if not np.isfinite(means[i]):
            continue
        for axis in (ax, ax_bottom):
            axis.bar(
                x[i],
                means[i],
                width,
                yerr=se[i] if axis is ax else None,
                color=colors[i],
                edgecolor="none",
                linewidth=2,
                capsize=2,
                error_kw={"elinewidth": 0.8},
            )
        ax.annotate(
            f"{means[i]:.{panel.decimals}f}",
            (x[i], means[i]),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            # white backing keeps the baseline line from striking through the value
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            zorder=4,
        )

    if panel.baseline_value is not None and np.isfinite(panel.baseline_value):
        ax.axhline(
            panel.baseline_value, color=baseline_color, linestyle="--", linewidth=1.0, zorder=3
        )

    fontweight = "medium"
    for axis in (ax, ax_bottom):
        axis.set_xlim(x.min() - bar_center_spacing, x.max() + bar_center_spacing)
        axis.set_xticks(x)
        axis.tick_params(axis="y", labelsize=8, width=1.2, length=4, direction="out")
        axis.tick_params(axis="x", length=0, width=2.25, labelbottom=False)
        axis.grid(False)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
        axis.spines["left"].set_linewidth(1.2)
    ax.set_ylim(y_lo, y_hi)
    ax_bottom.set_ylim(0.0, panel.bottom_max)
    ax_bottom.set_yticks([0.0])
    for axis in (ax, ax_bottom):
        plt.setp(axis.get_yticklabels(), fontweight=fontweight)
    ax.spines["bottom"].set_visible(False)
    ax_bottom.spines["bottom"].set_linewidth(1.5)

    if show_xticklabels:
        ax_bottom.set_xticklabels(
            method_labels, rotation=20, ha="right", fontsize=10, fontweight=fontweight
        )
        ax_bottom.tick_params(axis="x", labelbottom=True, labelcolor="black")

    # diagonal break marks where the left spine is interrupted
    break_kwargs = {
        "marker": [(-1, -0.6), (1, 0.6)],
        "markersize": 7,
        "linestyle": "none",
        "color": "black",
        "markeredgewidth": 1.0,
        "clip_on": False,
    }
    ax.plot([0], [0], transform=ax.transAxes, **break_kwargs)
    ax_bottom.plot([0], [1], transform=ax_bottom.transAxes, **break_kwargs)


def plot_metric_panels(
    method_labels: list[str],
    panels: list[MetricPanel],
    colors: list[str],
    *,
    baseline_label: str = BASELINE_LABEL,
    baseline_color: str = BASELINE_COLOR,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Stacked bar charts (one per metric, first on top) with SE error bars.

    With one panel this is the standalone single-metric figure. With several, the method names
    appear only under the bottom panel and a single baseline legend sits above the top panel.

    Args:
        method_labels: tick label per bar.
        panels: one :class:`MetricPanel` per metric, top to bottom.
        colors: bar color per method.
        baseline_label: legend text of the dashed baseline line.
        baseline_color: color of the baseline line.
        save_path: where to save the figure, if any.
        show: whether to show the figure (it is closed otherwise).

    Returns:
        The matplotlib figure.
    """
    fig, axes = _make_panel_axes(len(panels))
    for i, ((ax, ax_bottom), panel) in enumerate(zip(axes, panels)):
        _draw_metric_panel(
            ax,
            ax_bottom,
            panel,
            method_labels,
            colors,
            baseline_color=baseline_color,
            show_xticklabels=i == len(panels) - 1,
        )
        # y label centered across the upper and lower axes of this panel
        y_center = (ax.get_position().y1 + ax_bottom.get_position().y0) / 2
        fig.text(
            0.03,
            y_center,
            panel.y_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="medium",
        )

    if any(p.baseline_value is not None and np.isfinite(p.baseline_value) for p in panels):
        handle = Line2D([0], [0], color=baseline_color, linestyle="--", linewidth=1.0)
        # one legend above the top panel, outside the axes, covers every panel
        axes[0][0].legend(
            [handle],
            [baseline_label],
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=False,
            fontsize=7,
            handlelength=1.8,
            borderaxespad=0.2,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=400, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Root with one subfolder per method")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Method folder names")
    p.add_argument(
        "--method-labels",
        nargs="+",
        default=None,
        help="Display label per method (default: DEFAULT_METHOD_LABELS, aligned to --methods)",
    )
    p.add_argument(
        "--baseline",
        default=BASELINE_METHOD,
        help="Folder of the mean-frame baseline drawn as a dashed line ('none' to omit it)",
    )
    p.add_argument(
        "--layout",
        choices=["combined", "separate", "both"],
        default="both",
        help="combined: PSNR above SSIM in one figure; separate: one figure per metric",
    )
    p.add_argument("--eids", nargs="*", default=None, help="EIDs to include (default: EID_SET)")
    p.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory for the output images")
    p.add_argument("--no-show", action="store_true")

    args = p.parse_args()

    method_labels = args.method_labels
    if method_labels is None:
        method_labels = (
            DEFAULT_METHOD_LABELS if args.methods == DEFAULT_METHODS else list(args.methods)
        )
    if len(method_labels) != len(args.methods):
        raise ValueError(
            f"--method-labels must match --methods in length: {len(method_labels)} vs {len(args.methods)}"
        )

    results_dir = args.results_dir.expanduser().resolve()
    allowed = _allowed_eid_set(args.eids if args.eids is not None else sorted(EID_SET))
    colors = [METRIC_BASE_COLORS[method] for method in args.methods]

    means_psnr, means_ssim, se_psnr, se_ssim = collect_method_stats(results_dir, args.methods, allowed)
    baseline_psnr = baseline_ssim = None
    if args.baseline.lower() != "none":
        base_psnr, base_ssim, _, _ = collect_method_stats(results_dir, [args.baseline], allowed)
        # a missing baseline folder gives nan, which the panels treat as absent
        baseline_psnr, baseline_ssim = float(base_psnr[0]), float(base_ssim[0])

    panel_psnr = MetricPanel("PSNR", means_psnr, se_psnr, 10.0, 2, baseline_psnr)
    panel_ssim = MetricPanel("SSIM", means_ssim, se_ssim, 0.5, 3, baseline_ssim)

    output_root = args.output_dir if args.output_dir is not None else results_dir
    # all figures land in the same timestamped folder rather than one each.
    combined_save_path = _default_output_path_helper(
        output_root, "decoding_metrics", "psnr_ssim_barplot.png"
    )
    save_dir = combined_save_path.parent
    show = not args.no_show

    if args.layout in ("combined", "both"):
        plot_metric_panels(
            method_labels,
            [panel_psnr, panel_ssim],
            colors,
            save_path=combined_save_path,
            show=show,
        )
    if args.layout in ("separate", "both"):
        plot_metric_panels(
            method_labels, [panel_ssim], colors, save_path=save_dir / "ssim_barplot.png", show=show
        )
        plot_metric_panels(
            method_labels, [panel_psnr], colors, save_path=save_dir / "psnr_barplot.png", show=show
        )


if __name__ == "__main__":
    main()
