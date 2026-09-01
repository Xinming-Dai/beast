#!/usr/bin/env python3
"""Per-neuron BPS scatter for one EID: method A (y) vs method B (x).

Folder structure:

```text
figure3/
├── BEAST/
│   ├── <eid>/
│   │   └── encoding_results*.npy
├── ResNet/
...
```

Under each method folder, paths use :func:`iter_eid_encoding_npys` in
``plot_helpers.py`` (recursive EID directory discovery).

```bash
python src/analyses/neural_analysis/scatter_plot_figure3.py \
  --results-dir /path/to/figure3 --eid <uuid> --methods BEAST ResNet
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
    AXIS_LABEL_FONT_KWARGS,
    AXIS_TICK_LABEL_FONT_KWARGS,
    EID_SET,
    METHOD_LIST,
    PANEL_LABEL_FONT_KWARGS,
    _default_output_path_helper,
    iter_eid_encoding_npys,
    normalize_eid,
)
from scripts.neural_analysis.viz_single_cell import bps_per_neuron

RESULTS_DIR = Path("/work/nvme/bfsr/xdai3/project3d/plotting/figure3")
TEXT_FONT_SIZE = 12
FIG_SIZE = 2.2


def _encoding_dict(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    inner = raw.item() if raw.dtype == object and raw.shape == () else raw
    if not isinstance(inner, dict):
        raise TypeError(f"Expected dict in {path}, got {type(inner).__name__}")
    return inner


def _resolve_eid_key(root: dict, eid: str | None) -> str:
    if eid is None:
        if len(root) == 1:
            return str(next(iter(root)))
        raise ValueError(
            "encoding_results*.npy contains multiple EIDs; pass `eid=` explicitly. "
            f"Available keys: {list(root.keys())}"
        )
    want = str(eid).strip().lower()
    for k in root:
        if str(k).strip().lower() == want:
            return str(k)
    raise KeyError(f"EID {eid!r} not found in encoding file; keys: {list(root.keys())}")


def _gt_pred_for_encoder(
    root: dict, eid: str | None, encoder: str
) -> tuple[np.ndarray, np.ndarray]:
    eid_key = _resolve_eid_key(root, eid)
    sess = root[eid_key]
    if encoder not in sess:
        raise KeyError(
            f"Encoder {encoder!r} not under EID {eid_key!r}; keys: {list(sess.keys())}"
        )
    block = sess[encoder]
    if not isinstance(block, dict):
        raise TypeError(f"Expected dict for encoder block, got {type(block).__name__}")
    gt = np.asarray(block["gt"], dtype=np.float64)
    pred = np.asarray(block["pred"], dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"gt/pred must be 3D and match; got {gt.shape}, {pred.shape}")
    return gt, pred


def encoding_npy_for_method(
    results_dir: Path, method_folder: str, eid: str
) -> Path:
    """Resolve an ``encoding_results*.npy`` for one method via :func:`iter_eid_encoding_npys`."""
    method_dir = results_dir.expanduser().resolve() / method_folder
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {method_dir}")
    allowed = frozenset({normalize_eid(eid)})
    matches = sorted(iter_eid_encoding_npys(method_dir, allowed))
    if not matches:
        raise FileNotFoundError(
            f"No encoding_results*.npy under {method_dir} in a directory named for EID {eid!r}"
        )
    return matches[0]


def bps_vector_from_npy(npy_path: Path, eid: str | None, encoder: str) -> np.ndarray:
    root = _encoding_dict(npy_path)
    gt, pred = _gt_pred_for_encoder(root, eid, encoder)
    return bps_per_neuron(gt, pred)


def _apply_axis_tick_font(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=AXIS_TICK_LABEL_FONT_KWARGS["fontsize"])
    plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), **AXIS_TICK_LABEL_FONT_KWARGS)


def plot_bps_scatter(
    x_bps: np.ndarray,
    y_bps: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    panel_label: str | None = "C",
    show_bps: bool = False,
    show_axis_labels: bool = True,
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
) -> plt.Figure:
    mask = np.isfinite(x_bps) & np.isfinite(y_bps)
    x = np.asarray(x_bps[mask], dtype=np.float64)
    y = np.asarray(y_bps[mask], dtype=np.float64)

    mx = float(np.nanmean(x)) if x.size else float("nan")
    my = float(np.nanmean(y)) if y.size else float("nan")
    print(f"Mean BPS: {x_label} vs {y_label} = {mx:.2f} vs {my:.2f}")

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE), dpi=150)

    lo = -0.25
    hi = 1.85
    ticks = (-0.25, 0.45, 1.15, 1.85)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    for g in (0.45, 1.15):
        ax.axhline(g, color="0.85", linewidth=0.8, zorder=0)
        ax.axvline(g, color="0.85", linewidth=0.8, zorder=0)

    ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.55", linewidth=1.0, zorder=1)

    ax.scatter(
        x,
        y,
        s=18,
        facecolors=(0.25, 0.25, 0.25),
        edgecolors="none",
        alpha=0.35,
        zorder=2,
    )

    if show_axis_labels:
        ax.set_xlabel(x_label, **AXIS_LABEL_FONT_KWARGS)
        ax.set_ylabel(y_label, **AXIS_LABEL_FONT_KWARGS)
    _apply_axis_tick_font(ax)

    if panel_label:
        ax.text(
            -0.06,
            1.02,
            panel_label,
            transform=ax.transAxes,
            **PANEL_LABEL_FONT_KWARGS,
            va="bottom",
            ha="left",
            clip_on=False,
            zorder=10,
        )

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.0)

    fig.tight_layout()

    if show_bps and np.isfinite(mx) and np.isfinite(my):
        ha = "right"
        va = "bottom"
        x_right = 0.98
        y_b = 0.02
        s3 = f"{my:.2f}"
        s2 = "  vs  "
        s1 = f"{mx:.2f}"
        t3 = ax.text(
            x_right, y_b, s3, transform=ax.transAxes, fontsize=TEXT_FONT_SIZE, va=va, ha=ha, color="#1b5e20"
        )
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        w3 = t3.get_window_extent(renderer=r).width / fig.bbox.width
        t2 = ax.text(
            x_right - w3, y_b, s2, transform=ax.transAxes, fontsize=TEXT_FONT_SIZE, va=va, ha=ha, color="black"
        )
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        w2 = t2.get_window_extent(renderer=r).width / fig.bbox.width
        ax.text(
            x_right - w3 - w2,
            y_b,
            s1,
            transform=ax.transAxes,
            fontsize=TEXT_FONT_SIZE,
            va=va,
            ha=ha,
            color="0.45",
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
    default_eid = sorted(EID_SET)[0]
    default_methods = list(METHOD_LIST)
    if len(default_methods) < 2:
        raise RuntimeError("METHOD_LIST must contain at least two method folder names")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Root with one subfolder per method (e.g. BEAST, ResNet)",
    )
    p.add_argument(
        "--eid",
        type=str,
        default=default_eid,
        help=f"Session UUID (default: first sorted EID in EID_SET: {default_eid})",
    )
    p.add_argument(
        "--methods",
        nargs=2,
        metavar=("METHOD_Y", "METHOD_X"),
        default=(default_methods[0], default_methods[1]),
        help="Two method folder names: first → y-axis, second → x-axis",
    )
    p.add_argument(
        "--encoder",
        choices=("rrr", "cnn"),
        default="cnn",
        help="Which encoder block in encoding_results*.npy to use (default: cnn)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: timestamped folder under results-dir)",
    )
    p.add_argument("--save-pdf", action="store_true", help="Also save PDF")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--panel-label", type=str, default=None, help="Panel letter (empty to omit)")
    p.add_argument("--show-bps", action="store_true", help="Show mean BPS text on the plot")
    p.add_argument(
        "--no-axis-labels", action="store_true", help="Hide the x and y axis labels"
    )

    args = p.parse_args()
    method_y, method_x = args.methods

    path_y = encoding_npy_for_method(args.results_dir, method_y, args.eid)
    path_x = encoding_npy_for_method(args.results_dir, method_x, args.eid)

    bps_y = bps_vector_from_npy(path_y, args.eid, args.encoder)
    bps_x = bps_vector_from_npy(path_x, args.eid, args.encoder)

    if bps_x.shape != bps_y.shape:
        raise ValueError(
            f"Neuron count mismatch for scatter: {method_x} has {bps_x.shape}, "
            f"{method_y} has {bps_y.shape}"
        )

    save_path = (
        args.output
        if args.output is not None
        else _default_output_path_helper(
            args.results_dir, "scatter_plot", "scatter_plot.png"
        )
    )

    plot_bps_scatter(
        bps_x,
        bps_y,
        x_label=method_x,
        y_label=method_y,
        panel_label=args.panel_label or None,
        show_bps=args.show_bps,
        show_axis_labels=not args.no_axis_labels,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
