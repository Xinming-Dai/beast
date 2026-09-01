#!/usr/bin/env python3
"""Encoding BPS bar plot across methods, aggregated over EIDs.

Folder structure:

```text
figure3/
├── random_baseline/
│   ├── <eid>/
│   │   └── encoding_results*.npy
├── keypoints/
...
```

Under each method folder, paths use :func:`iter_eid_encoding_npys` in
``plot_helpers.py`` (recursive EID directory discovery).

```bash
python scripts/neural_analysis/bar_plot_encoding_figure3.py \
  --results-dir /path/to/figure3 --methods random_baseline keypoints pca resnet beast sable_dino
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
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHODS,
    EID_SET,
    ENCODING_BASE_COLORS,
    _default_output_path_helper,
    _lightened_hex,
    iter_eid_encoding_npys,
    normalize_eid,
)
from scripts.neural_analysis.scatter_plot_figure3 import bps_vector_from_npy

RESULTS_DIR = Path("/projects/bfsr/xdai3/project3d/iclr_plotting/SABLE_zero_shot_encoding")


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


def method_color_pairs(methods: list[str], *, plot_encoders: str) -> list[tuple[str, str]]:
    """(RRR color, CNN color) per method, from :data:`ENCODING_BASE_COLORS`."""
    pairs = []
    for method in methods:
        if method not in ENCODING_BASE_COLORS:
            raise KeyError(f"No color defined for method {method!r}; add it to ENCODING_BASE_COLORS")
        base = ENCODING_BASE_COLORS[method]
        cnn_color = _lightened_hex(base) if plot_encoders == "both" else base
        pairs.append((base, cnn_color))
    return pairs


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


def collect_method_stats(
    results_dir: Path,
    methods: list[str],
    allowed_eids: frozenset[str] | None,
    *,
    bps_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (means_rrr, means_cnn, se_rrr, se_cnn), one entry per method (nan if missing)."""
    if bps_source not in {"stored", "per-neuron"}:
        raise ValueError(f"Unknown bps_source: {bps_source!r}")

    n = len(methods)
    means_rrr = np.full(n, np.nan)
    means_cnn = np.full(n, np.nan)
    se_rrr = np.full(n, np.nan)
    se_cnn = np.full(n, np.nan)

    for i, method in enumerate(methods):
        sub = results_dir / method
        if bps_source == "stored":
            agg = aggregate_bps_for_method(sub, allowed_eids)
        else:
            agg = aggregate_bps_vectors_for_method(sub, allowed_eids)
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

    return means_rrr, means_cnn, se_rrr, se_cnn


def _label_position(value: float, label_offset: float) -> tuple[float, str]:
    """Label sits above positive bars and below negative/near-zero bars."""
    if value >= 0:
        return value + label_offset, "bottom"
    return value - label_offset, "top"


def plot_encoding_bars(
    method_labels: list[str],
    means_rrr: np.ndarray,
    means_cnn: np.ndarray,
    se_rrr: np.ndarray,
    se_cnn: np.ndarray,
    color_pairs: list[tuple[str, str]],
    *,
    plot_encoders: str = "cnn",
    rrr_label: str | None = None,
    tcn_label: str | None = None,
    y_label: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> plt.Figure:
    if plot_encoders not in {"cnn", "rrr", "both"}:
        raise ValueError(f"Unknown plot_encoders: {plot_encoders!r}")

    n = len(method_labels)
    width = 0.28 if plot_encoders == "both" else 0.4
    # bar centers spaced by less than 1.0 pulls groups closer on the x axis (vs np.arange alone).
    bar_center_spacing = 0.75
    # padding between the y axis / right edge and the outermost bars, smaller than
    # bar_center_spacing so the first and last bars sit closer to the axes than to each other.
    edge_padding = 0.4
    x = np.arange(n, dtype=np.float64) * bar_center_spacing
    fig, ax = plt.subplots(figsize=(0.68 * n, 2.0), dpi=200)

    relevant_means = np.concatenate([
        means_rrr[np.isfinite(means_rrr)] if plot_encoders in {"rrr", "both"} else np.array([]),
        means_cnn[np.isfinite(means_cnn)] if plot_encoders in {"cnn", "both"} else np.array([]),
    ])
    y_max = float(relevant_means.max()) if relevant_means.size else 1.0
    y_min_data = float(relevant_means.min()) if relevant_means.size else 0.0
    y_lim = y_max * 1.30
    # leave headroom below zero (plus room for a below-bar label) whenever a bar dips negative,
    # so a near-zero value's bar and text label don't collide with the zero line.
    label_offset = y_lim * 0.06
    y_min = min(0.0, y_min_data - 3 * label_offset)

    first_rrr_legend = True
    first_cnn_legend = True
    for i in range(n):
        c_rrr, c_cnn = color_pairs[i]
        if plot_encoders in {"rrr", "both"} and np.isfinite(means_rrr[i]):
            xpos = x[i] - width / 2 if plot_encoders == "both" else x[i]
            ax.bar(
                xpos,
                means_rrr[i],
                width,
                yerr=se_rrr[i],
                color=c_rrr,
                edgecolor="none",
                linewidth=2,
                capsize=2,
                error_kw={"elinewidth": 0.8},
                label=rrr_label if first_rrr_legend else None,
            )
            first_rrr_legend = False
            y_text, va_text = _label_position(means_rrr[i], label_offset)
            ax.text(xpos, y_text, f"{means_rrr[i]:.3f}", ha="center", va=va_text, fontsize=11)
        if plot_encoders in {"cnn", "both"} and np.isfinite(means_cnn[i]):
            xpos = x[i] + width / 2 if plot_encoders == "both" else x[i]
            ax.bar(
                xpos,
                means_cnn[i],
                width,
                yerr=se_cnn[i],
                color=c_cnn,
                edgecolor="none",
                linewidth=2,
                capsize=2,
                error_kw={"elinewidth": 0.8},
                label=tcn_label if first_cnn_legend else None,
            )
            first_cnn_legend = False
            y_text, va_text = _label_position(means_cnn[i], label_offset)
            ax.text(xpos, y_text, f"{means_cnn[i]:.3f}", ha="center", va=va_text, fontsize=11)

    fontweight = "medium"
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=20, ha="right", fontsize=12, fontweight=fontweight)
    ax.set_ylabel(y_label or "Avg BPS", fontsize=12, fontweight=fontweight)
    ax.tick_params(axis="y", labelsize=11, width=1.5, length=7, direction="out")
    ax.tick_params(axis="x", length=0, width=2.25, pad=1)
    plt.setp(ax.get_yticklabels(), fontweight=fontweight)
    plt.setp(ax.get_xticklabels(), fontweight=fontweight)

    ax.set_ylim(y_min, y_lim)
    ax.grid(False)

    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    # zero-reference line in place of the bottom spine, since the axes now extend below zero
    # to make room for negative bars (e.g. random_baseline).
    ax.axhline(0, color="black", linewidth=1.5, zorder=0.5)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.12),
        frameon=False,
        ncol=2,
        fontsize=8,
        handlelength=1.2,
        columnspacing=1.2,
    )

    ax.set_xlim(x.min() - edge_padding, x.max() + edge_padding)
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
    p.add_argument(
        "--plot-encoders",
        choices=("cnn", "rrr", "both"),
        default="cnn",
        help="Which encoder(s) to plot (default: cnn)",
    )
    p.add_argument(
        "--bps-source",
        choices=("stored", "per-neuron"),
        default="per-neuron",
        help="Aggregate stored per-EID BPS, or recompute per-neuron BPS from gt/pred",
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Output image path")
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

    means_rrr, means_cnn, se_rrr, se_cnn = collect_method_stats(
        results_dir, args.methods, allowed, bps_source=args.bps_source,
    )
    color_pairs = method_color_pairs(args.methods, plot_encoders=args.plot_encoders)

    save_path = (
        args.output
        if args.output is not None
        else _default_output_path_helper(results_dir, "encoding_barplot", "encoding_barplot.png")
    )

    plot_encoding_bars(
        method_labels,
        means_rrr,
        means_cnn,
        se_rrr,
        se_cnn,
        color_pairs,
        plot_encoders=args.plot_encoders,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
