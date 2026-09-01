#!/usr/bin/env python3
"""SSIM and PSNR bar plots across decoding methods, aggregated over EIDs.

Folder structure:

```text
figure2/
├── resnet/
│   ├── <eid>/
│   │   └── psnr_ssim_metrics.npz
├── beast/
...
```

Under each method folder, paths use :func:`iter_eid_metrics_npys` (recursive EID
directory discovery, one ``psnr_ssim_metrics.npz`` per EID).

```bash
python scripts/neural_analysis/decoding_metrics_figure2.py \
  --results-dir /path/to/figure2 --methods resnet beast sable
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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

RESULTS_DIR = Path("/projects/bfsr/xdai3/project3d/iclr_plotting/SABLE_zero_shot_decoding")
DEFAULT_METHODS: list[str] = ["resnet", "beast", "sable"]
DEFAULT_METHOD_LABELS: list[str] = ["ResNet AE", "BEAST", "SABLE"]
METRIC_BASE_COLORS: dict[str, str] = {
    "resnet": ENCODING_BASE_COLORS["resnet"],
    "beast": ENCODING_BASE_COLORS["beast"],
    "sable": ENCODING_BASE_COLORS["sable_dino"],
}


def _figsize() -> tuple[float, float]:
    width, height = BASE_FIGSIZE
    return width * FIGURE_SIZE_SCALE, height * FIGURE_SIZE_SCALE


def _allowed_eid_set(eids: list[str] | None) -> frozenset[str] | None:
    """None / empty = no name filter (any folder containing psnr_ssim_metrics.npz)."""
    if not eids:
        return None
    return frozenset(normalize_eid(e) for e in eids)


def iter_eid_metrics_npys(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
):
    """Yield ``psnr_ssim_metrics.npz`` paths for matching EID folders.

    If ``allowed_eids`` is set, only directories whose **name** is in that set (compared
    via :func:`normalize_eid`) are searched. Exactly one ``psnr_ssim_metrics.npz`` file
    is expected under each matching EID directory. If ``allowed_eids`` is ``None``, every
    matching file under ``method_dir`` is included.
    """
    if not method_dir.is_dir():
        return
    if allowed_eids is None:
        yield from method_dir.rglob("psnr_ssim_metrics.npz")
        return

    seen: set[Path] = set()
    for d in method_dir.rglob("*"):
        if not d.is_dir() or normalize_eid(d.name) not in allowed_eids:
            continue
        matches = sorted(met for met in d.rglob("psnr_ssim_metrics.npz") if met.is_file())
        if len(matches) > 1:
            raise ValueError(
                f"Expected one psnr_ssim_metrics.npz under EID directory {d}, "
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


def plot_metric_bars(
    method_labels: list[str],
    means: np.ndarray,
    se: np.ndarray,
    colors: list[str],
    *,
    y_label: str,
    headroom: float,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Bar chart of one metric (e.g. SSIM or PSNR) across methods, with SE error bars."""
    n = len(method_labels)
    width = 0.55
    # spacing wider than width leaves a gap between bars so the value labels above
    # neighboring bars don't collide.
    bar_center_spacing = 1.3
    x = np.arange(n, dtype=np.float64) * bar_center_spacing
    fig, ax = plt.subplots(figsize=_figsize(), dpi=200)

    finite_means = means[np.isfinite(means)]
    y_lim = float(finite_means.max()) + headroom if finite_means.size else headroom

    for i in range(n):
        if not np.isfinite(means[i]):
            continue
        ax.bar(
            x[i],
            means[i],
            width,
            yerr=se[i],
            color=colors[i],
            edgecolor="none",
            linewidth=2,
            capsize=2,
            error_kw={"elinewidth": 0.8},
        )
        ax.text(x[i], means[i] + 0.015, f"{means[i]:.2f}", ha="center", va="bottom", fontsize=9)

    fontweight = "medium"
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=20, ha="right", fontsize=10, fontweight=fontweight)
    ax.set_xlim(x.min() - bar_center_spacing, x.max() + bar_center_spacing)
    ax.set_ylabel(y_label, fontsize=10, fontweight=fontweight)
    ax.tick_params(axis="y", labelsize=9, width=1.5, length=7, direction="out")
    ax.tick_params(axis="x", length=0, width=2.25)
    plt.setp(ax.get_yticklabels(), fontweight=fontweight)
    plt.setp(ax.get_xticklabels(), fontweight=fontweight)

    ax.set_ylim(0.0, y_lim)
    ax.grid(False)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.5)

    fig.tight_layout(pad=0.4)

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

    output_root = args.output_dir if args.output_dir is not None else results_dir
    # both plots land in the same timestamped folder rather than one each.
    ssim_save_path = _default_output_path_helper(output_root, "decoding_metrics", "ssim_barplot.png")
    psnr_save_path = ssim_save_path.parent / "psnr_barplot.png"

    plot_metric_bars(
        method_labels,
        means_ssim,
        se_ssim,
        colors,
        y_label="SSIM",
        headroom=0.15,
        save_path=ssim_save_path,
        show=not args.no_show,
    )
    plot_metric_bars(
        method_labels,
        means_psnr,
        se_psnr,
        colors,
        y_label="PSNR",
        headroom=4.0,
        save_path=psnr_save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
