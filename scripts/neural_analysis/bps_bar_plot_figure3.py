#!/usr/bin/env python3
"""Grouped bar plot: mean BPS across EIDs (error bars = SE) per method × encoder.

Folder structure:

```text
figure3/
├── BEAST/
│   ├── <eid>/
│   │   └── encoding_results*.npy
├── ResNet/
...
```

Each subfolder contains a directory for each EID.
Each EID directory contains one or more `encoding_results*.npy` files.
The `encoding_results*.npy` file is a numpy array of shape `(num_neural_trials, num_neural_bins, num_neurons)`.
The `num_neural_trials` is the number of neural trials.

```
python "src/analyses/neural_analysis/bps_bar_plot_figure3.py" \
  --results-dir /path/to/figure3
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from analyses.neural_analysis.plot_helpers import (
    EID_SET,
    METHOD_LIST,
    _default_output_path_helper,
    iter_eid_encoding_npys,
    normalize_eid,
)
from analyses.neural_analysis.scatter_plot_figure3 import bps_vector_from_npy

# Default layout: one subfolder per method name; under each, directories whose names appear in EID.
RESULTS_DIR = Path("/work/nvme/bfsr/xdai3/project3d/plotting/figure3")
BpsSource = Literal["stored", "per-neuron"]
PlotEncoders = Literal["cnn", "rrr", "both"]

# Qualitative colormap name (matplotlib). Bar group i uses color i (wrapped). RRR = base; TCN = lightened.
BAR_CMAP = "Set2"
# Optional: set to a non-empty list to pin (RRR hex, TCN hex) per group in order instead of BAR_CMAP. e.g. [("#66bb6a", "#c8e6c9"), ("#5c8bc9", "#90caf9")]
BAR_COLOR_PAIRS: list[tuple[str, str]] | None = None


def _as_scalar_bps(x: object) -> float:
    a = np.asarray(x)
    return float(np.nanmean(a))


def _bps_rrr_cnn_from_dict(data: dict) -> tuple[float, float]:
    """Return (rrr_bps, cnn_bps) from encoding_results payload (nested or flat)."""
    if "rrr" in data and "cnn" in data:
        rrr, cnn = data["rrr"], data["cnn"]
        if isinstance(rrr, dict) and isinstance(cnn, dict):
            return _as_scalar_bps(rrr["bps"]), _as_scalar_bps(cnn["bps"])
    for v in data.values():
        if isinstance(v, dict) and "rrr" in v and "cnn" in v:
            rrr, cnn = v["rrr"], v["cnn"]
            if isinstance(rrr, dict) and isinstance(cnn, dict):
                return _as_scalar_bps(rrr["bps"]), _as_scalar_bps(cnn["bps"])
    raise KeyError("Could not find ['rrr']['bps'] and ['cnn']['bps'] in encoding_results dict")


def _allowed_eid_set(eids: list[str] | None) -> frozenset[str] | None:
    """None / empty = no name filter (any folder containing encoding_results*.npy)."""
    if not eids:
        return None
    return frozenset(normalize_eid(e) for e in eids)


def bar_color_pair(index: int, *, cmap_name: str = BAR_CMAP) -> tuple[str, str]:
    """(RRR color, TCN color) for bar group ``index`` — no method names required."""
    if BAR_COLOR_PAIRS:
        c_rrr, c_cnn = BAR_COLOR_PAIRS[index % len(BAR_COLOR_PAIRS)]
        return c_rrr, c_cnn
    cmap = plt.colormaps[cmap_name]
    if isinstance(cmap, mcolors.ListedColormap) and len(cmap.colors) > 0:
        rgb = np.asarray(cmap.colors[index % len(cmap.colors)], dtype=float)[:3]
    else:
        n = max(int(getattr(cmap, "N", 256)), 1)
        t = (index % n) / max(n - 1, 1)
        rgb = np.asarray(cmap(t), dtype=float)[:3]
    rrr = mcolors.to_hex(rgb)
    lit = tuple(min(1.0, 0.52 + 0.48 * c) for c in rgb)
    tcn = mcolors.to_hex(lit)
    return rrr, tcn


def load_bps_pair(npy_path: Path) -> tuple[float, float]:
    raw = np.load(npy_path, allow_pickle=True)
    inner = raw.item() if raw.dtype == object and raw.shape == () else raw
    if not isinstance(inner, dict):
        raise TypeError(f"Expected dict in {npy_path}, got {type(inner).__name__}")
    return _bps_rrr_cnn_from_dict(inner)


def aggregate_bps_for_method(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect per-EID RRR and CNN BPS; return (rrr_values, cnn_values) or None if empty."""
    rrr_list: list[float] = []
    cnn_list: list[float] = []
    for npy in sorted(iter_eid_encoding_npys(method_dir, allowed_eids), key=lambda p: p.parent.name):
        try:
            rrr, cnn = load_bps_pair(npy)
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to read BPS from {npy}") from e
        rrr_list.append(rrr)
        cnn_list.append(cnn)
    if not rrr_list:
        return None
    return np.asarray(rrr_list, dtype=np.float64), np.asarray(cnn_list, dtype=np.float64)


def _finite_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _eid_for_encoding_npy(npy_path: Path, allowed_eids: frozenset[str] | None) -> str | None:
    if allowed_eids is None:
        return None
    for ancestor in npy_path.parents:
        if normalize_eid(ancestor.name) in allowed_eids:
            return ancestor.name
    return None


def aggregate_bps_vectors_for_method(
    method_dir: Path,
    allowed_eids: frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect per-neuron RRR and CNN BPS vectors across all matching EIDs."""
    rrr_vectors: list[np.ndarray] = []
    cnn_vectors: list[np.ndarray] = []
    for npy in sorted(iter_eid_encoding_npys(method_dir, allowed_eids), key=lambda p: p.parent.name):
        eid = _eid_for_encoding_npy(npy, allowed_eids)
        try:
            rrr = bps_vector_from_npy(npy, eid, "rrr")
            cnn = bps_vector_from_npy(npy, eid, "cnn")
        except (KeyError, TypeError, ValueError) as e:
            raise RuntimeError(f"Failed to compute per-neuron BPS from {npy}") from e
        rrr_vectors.append(_finite_1d(rrr))
        cnn_vectors.append(_finite_1d(cnn))
    if not rrr_vectors:
        return None
    return np.concatenate(rrr_vectors), np.concatenate(cnn_vectors)


def _mean_and_standard_error(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.nanmean(values))
    se = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return mean, se


def plot_bps_grouped_bar(
    results_dir: Path,
    methods: list[str],
    *,
    eids: list[str] | None = None,
    panel_label: str | None = "B",
    y_label: str = "Bits Per Spike",
    rrr_label: str = "RRR",
    tcn_label: str = "TCN",
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
    bps_source: BpsSource = "stored",
    plot_encoders: PlotEncoders = "cnn",
) -> plt.Figure:
    if plot_encoders not in {"cnn", "rrr", "both"}:
        raise ValueError(f"Unknown plot_encoders: {plot_encoders!r}")

    results_dir = results_dir.expanduser().resolve()
    allowed = _allowed_eid_set(eids if eids is not None else EID_SET)
    n = len(methods)
    means_rrr = np.full(n, np.nan)
    means_cnn = np.full(n, np.nan)
    se_rrr = np.full(n, np.nan)
    se_cnn = np.full(n, np.nan)

    for i, method in enumerate(methods):
        sub = results_dir / method
        if bps_source == "stored":
            agg = aggregate_bps_for_method(sub, allowed)
        elif bps_source == "per-neuron":
            agg = aggregate_bps_vectors_for_method(sub, allowed)
        else:
            raise ValueError(f"Unknown bps_source: {bps_source!r}")
        if agg is None:
            continue
        rrr, cnn = agg
        if bps_source == "stored":
            means_rrr[i] = float(np.mean(rrr))
            means_cnn[i] = float(np.mean(cnn))
            se_rrr[i] = float(np.std(rrr, ddof=1) / np.sqrt(rrr.size)) if rrr.size > 1 else 0.0
            se_cnn[i] = float(np.std(cnn, ddof=1) / np.sqrt(cnn.size)) if cnn.size > 1 else 0.0
        else:
            means_rrr[i], se_rrr[i] = _mean_and_standard_error(rrr)
            means_cnn[i], se_cnn[i] = _mean_and_standard_error(cnn)

    x = np.arange(n, dtype=np.float64)
    width = 0.36 if plot_encoders == "both" else 0.5
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)

    first_rrr_legend = True
    first_cnn_legend = True
    for i, _ in enumerate(methods):
        c_rrr, c_cnn = bar_color_pair(i)
        if plot_encoders in {"rrr", "both"} and np.isfinite(means_rrr[i]):
            xpos = x[i] - width / 2 if plot_encoders == "both" else x[i]
            ax.bar(
                xpos,
                means_rrr[i],
                width,
                yerr=se_rrr[i],
                color=c_rrr,
                hatch="//",
                edgecolor="black",
                linewidth=0.6,
                capsize=3,
                error_kw={"elinewidth": 1.0, "color": "black"},
                label=rrr_label if first_rrr_legend else None,
            )
            first_rrr_legend = False
            ax.text(
                xpos,
                means_rrr[i] * 0.05,
                f"{means_rrr[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )
        if plot_encoders in {"cnn", "both"} and np.isfinite(means_cnn[i]):
            xpos = x[i] + width / 2 if plot_encoders == "both" else x[i]
            ax.bar(
                xpos,
                means_cnn[i],
                width,
                yerr=se_cnn[i],
                color=c_cnn,
                edgecolor="black",
                linewidth=0.6,
                capsize=3,
                error_kw={"elinewidth": 1.0, "color": "black"},
                label=tcn_label if first_cnn_legend else None,
            )
            first_cnn_legend = False
            ax.text(
                xpos,
                means_cnn[i] * 0.05,
                f"{means_cnn[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="upper left",
        frameon=True,
        ncol=2,
        facecolor="none",
        edgecolor="none",
    )
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)
    fig.tight_layout()
    if panel_label:
        ax.text(
            -0.06,
            1.02,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
            zorder=10,
        )
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
        help="Root directory containing one subfolder per method name",
    )
    p.add_argument(
        "--methods",
        nargs="*",
        default=METHOD_LIST,
        help="Method folder names (x-axis order)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Save figure to this file (default: "
            "<results-dir>/bps_bar_plot_<YYYYMMDD_HHMMSS>/bps_bar_plot.png)"
        ),
    )
    p.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save a PDF next to the primary file (same basename, .pdf)",
    )
    p.add_argument("--no-show", action="store_true", help="Do not open interactive window")
    p.add_argument("--panel-label", type=str, default="B", help="Panel letter (empty to omit)")
    p.add_argument(
        "--all-eids",
        action="store_true",
        help="Do not filter by folder name (any directory containing encoding_results*.npy)",
    )
    p.add_argument(
        "--bps-source",
        choices=("stored", "per-neuron"),
        default="stored",
        help=(
            "How to aggregate BPS: 'stored' uses saved scalar bps values per EID; "
            "'per-neuron' recomputes per-neuron BPS vectors and concatenates them"
        ),
    )
    p.add_argument(
        "--plot-encoders",
        choices=("cnn", "rrr", "both"),
        default="cnn",
        help="Which encoder bars to plot (default: cnn only)",
    )
    args = p.parse_args()

    save_path = args.output if args.output is not None else _default_output_path_helper(args.results_dir, "bps_bar_plot", "bps_bar_plot.png")

    plot_bps_grouped_bar(
        args.results_dir,
        list(args.methods),
        eids=[] if args.all_eids else None,
        panel_label=args.panel_label or None,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
        bps_source=args.bps_source,
        plot_encoders=args.plot_encoders,
    )


if __name__ == "__main__":
    main()

