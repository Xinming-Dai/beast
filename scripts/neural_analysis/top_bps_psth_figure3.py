#!/usr/bin/env python3
"""Plot PSTHs for the top-BPS neurons in one encoding-results file.

This is a lightweight alternative to ``viz_single_cell.py`` for result files where
``mean_X`` has a singleton time axis, e.g. ``mean_X.shape[0] == 1`` while
``gt.shape[1] == 60``. In that case the latent-based single-cell visualization
cannot be built, but PSTHs can still be plotted directly from ``gt`` and ``pred``.

Example:

```
python src/analyses/neural_analysis/top_bps_psth_figure3.py \
  --encoding-npy /path/to/encoding_results.npy
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np

from scripts.neural_analysis.plot_helpers import bps_per_neuron, normalize_eid


DEFAULT_RESULTS_DIR = Path("/work/nvme/bfsr/xdai3/project3d/plotting/figure3")
GROUND_TRUTH_COLOR = "black"
PRED_COLOR = "#2f8f2f"


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


def _resolve_eid_key(root: dict, eid: str | None) -> str:
    if not root:
        raise ValueError("encoding results did not contain any EID entries")
    if eid is None:
        if len(root) == 1:
            return str(next(iter(root)))
        raise ValueError(
            "encoding results contain multiple EIDs; pass --eid. "
            f"Available keys: {sorted(str(key) for key in root)}"
        )

    want = normalize_eid(eid)
    for key in root:
        if normalize_eid(str(key)) == want:
            return str(key)
    raise KeyError(f"EID {eid!r} not found; keys: {sorted(str(key) for key in root)}")


def find_encoding_results_npy(results_dir: Path, method_folder: str, eid: str) -> Path:
    """Find an ``encoding_results*.npy`` for ``eid`` under one Figure 3 method folder."""
    method_dir = results_dir.expanduser().resolve() / method_folder
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {method_dir}")

    want = normalize_eid(eid)
    matches: list[Path] = []
    for candidate_dir in method_dir.rglob("*"):
        if candidate_dir.is_dir() and normalize_eid(candidate_dir.name) == want:
            matches.extend(sorted(candidate_dir.rglob("encoding_results*.npy")))
    matches = [path for path in matches if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No encoding_results*.npy under {method_dir} in a directory named {eid!r}"
        )
    if len(matches) > 1:
        raise ValueError(f"Expected one match, found {len(matches)}: {[str(p) for p in matches]}")
    return matches[0]


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
    """Map 0-based trial selection to ``(start, stop)`` for slicing trials."""
    if trials_idx is None:
        return 0, n_trials_full
    if len(trials_idx) == 1:
        n = int(trials_idx[0])
        if n < 1:
            raise IndexError("--trial-idx N: N must be >= 1")
        if n > n_trials_full:
            raise IndexError(f"--trial-idx N={n} exceeds n_trials={n_trials_full}")
        return 0, n

    start, last = int(trials_idx[0]), int(trials_idx[1])
    if start < 0 or last < start:
        raise IndexError(f"--trial-idx K,N: need 0 <= K <= N, got K={start}, N={last}")
    if last >= n_trials_full:
        raise IndexError(f"--trial-idx upper index {last} >= n_trials={n_trials_full}")
    return start, last + 1


def top_neuron_indices_by_bps(bps: np.ndarray, n_top: int) -> np.ndarray:
    """Return finite-BPS neuron indices sorted descending by BPS."""
    if n_top < 1:
        raise ValueError("--top-bps must be >= 1")
    finite = np.isfinite(bps)
    if not np.any(finite):
        raise ValueError("No finite BPS values were available")
    idx = np.nonzero(finite)[0]
    order = np.argsort(-bps[finite])
    return idx[order[: min(n_top, order.size)]]


def _psth(trace: np.ndarray) -> np.ndarray:
    """Average a single-neuron ``(trials, time)`` trace over trials."""
    return np.nanmean(np.asarray(trace, dtype=np.float64), axis=0)


def _normalize_traces(traces: list[np.ndarray]) -> list[np.ndarray]:
    finite_chunks = [trace[np.isfinite(trace)] for trace in traces if np.any(np.isfinite(trace))]
    if not finite_chunks:
        return traces
    finite_values = np.concatenate(finite_chunks)
    lo = float(np.nanmin(finite_values))
    hi = float(np.nanmax(finite_values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [np.zeros_like(trace, dtype=np.float64) for trace in traces]
    return [(trace - lo) / (hi - lo) for trace in traces]


def plot_top_bps_psth(
    encoding_npy: Path,
    *,
    method: str = "cnn",
    eid: str | None = None,
    top_bps: int = 5,
    duration_s: float = 1.0,
    trials_idx: tuple[int, ...] | None = None,
    normalize: bool = True,
    save_path: Path | None = None,
    also_save_pdf: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Plot PSTH panels for the top-BPS neurons in one encoder block."""
    root = _encoding_dict(encoding_npy)
    eid_key = _resolve_eid_key(root, eid)
    session = root[eid_key]
    if not isinstance(session, dict):
        raise TypeError(f"Expected dict for EID {eid_key!r}, got {type(session).__name__}")
    if method not in session:
        raise KeyError(f"Method {method!r} not found under EID {eid_key!r}; keys: {list(session)}")

    block = session[method]
    gt = np.asarray(block["gt"], dtype=np.float64)
    pred = np.asarray(block["pred"], dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"gt/pred must match 3D (trials,time,neurons); got {gt.shape}, {pred.shape}")

    mean_x = block.get("mean_X")
    if mean_x is not None:
        mean_x = np.asarray(mean_x)
        if mean_x.ndim >= 1 and mean_x.shape[0] != gt.shape[1]:
            print(
                f"[{method}] mean_X time dim {mean_x.shape[0]} does not match gt time dim "
                f"{gt.shape[1]}; plotting PSTH-only top-BPS neurons."
            )

    t0, t1 = trials_idx_to_bounds(gt.shape[0], trials_idx)
    if (t0, t1) != (0, gt.shape[0]):
        print(f"[{method}] Using trial indices {t0}..{t1 - 1} ({t1 - t0} trials)")
    gt = gt[t0:t1]
    pred = pred[t0:t1]

    bps = bps_per_neuron(gt, pred)
    neuron_indices = top_neuron_indices_by_bps(bps, top_bps)
    print(
        f"[{method}] Top {len(neuron_indices)} by BPS (index, bps): "
        + ", ".join(f"({int(idx)}, {bps[idx]:.4f})" for idx in neuron_indices)
    )

    n_panels = int(neuron_indices.size)
    n_cols = min(5, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_width = max(2.0 * n_cols, 3.0)
    fig_height = max(1.8 * n_rows, 1.8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=200, squeeze=False)
    axes_flat = axes.ravel()
    time_s = np.linspace(0.0, float(duration_s), gt.shape[1])

    for ax, neuron_idx in zip(axes_flat, neuron_indices):
        neuron_idx = int(neuron_idx)
        gt_trace = _psth(gt[:, :, neuron_idx])
        pred_trace = _psth(pred[:, :, neuron_idx])
        if normalize:
            gt_trace, pred_trace = _normalize_traces([gt_trace, pred_trace])

        ax.plot(time_s, gt_trace, color=GROUND_TRUTH_COLOR, linewidth=1.1)
        ax.plot(time_s, pred_trace, color=PRED_COLOR, linewidth=2.5)
        ax.set_title(f"Neuron {neuron_idx}\nBPS {bps[neuron_idx]:.2f}", fontsize=9)
        ax.set_xlim(0.0, float(duration_s))
        ax.set_xticks([0.0, float(duration_s)])
        ax.set_xticklabels(["0", f"{duration_s:.1f}"])
        ax.set_yticks([0.0, 1.0] if normalize else [])
        ax.tick_params(axis="both", labelsize=8, width=1.0, length=3, pad=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    for row_idx in range(n_rows):
        axes[row_idx, 0].set_ylabel("Normalized\nFiring Rate" if normalize else "Firing Rate", fontsize=9)
    for ax_idx, ax in enumerate(axes_flat[:n_panels]):
        if ax_idx % n_cols == 0:
            continue
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines["left"].set_visible(False)
    fig.supxlabel("Time (s)", fontsize=9, y=0.02)
    handles = [
        plt.Line2D([0], [0], color=GROUND_TRUTH_COLOR, linewidth=1.1, label="Ground Truth"),
        plt.Line2D([0], [0], color=PRED_COLOR, linewidth=2.5, label=method),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=False,
        ncol=2,
        fontsize=9,
        handlelength=1.2,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.16, wspace=0.35, hspace=0.75)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved {save_path.resolve()}")
        if also_save_pdf:
            pdf_path = save_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved {pdf_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, neuron_indices, bps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding-npy",
        type=Path,
        default=None,
        help="Path to encoding_results*.npy. If omitted, use --results-dir/--method-folder/--eid.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Figure 3 root directory, used only when --encoding-npy is omitted.",
    )
    parser.add_argument(
        "--method-folder",
        type=str,
        default="CLS",
        help="Folder under --results-dir, used only when --encoding-npy is omitted.",
    )
    parser.add_argument("--eid", type=str, default=None, help="Session EID; required if results contain multiple EIDs.")
    parser.add_argument(
        "--method",
        choices=("cnn", "rrr"),
        default="cnn",
        help="Encoder block inside encoding results. Default only calculates cnn.",
    )
    parser.add_argument("--top-bps", type=int, default=10, help="Number of top-BPS neurons to plot.")
    parser.add_argument("--duration-s", type=float, default=1.0, help="Right x-axis time in seconds.")
    parser.add_argument("--no-normalize", action="store_true", help="Plot raw PSTH values instead of normalized traces.")
    parser.add_argument(
        "--trial-idx",
        type=parse_trials_idx_arg,
        default=None,
        metavar="N | K,N",
        help="0-based trials: N alone = first N trials; two values = indices K..N inclusive.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: <encoding_results parent>/top_bps_psth_<method>.png.",
    )
    parser.add_argument("--save-pdf", action="store_true", help="Also save a PDF next to the PNG.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    args = parser.parse_args()

    encoding_npy = args.encoding_npy
    if encoding_npy is None:
        if args.eid is None:
            raise ValueError("--eid is required when --encoding-npy is omitted")
        encoding_npy = find_encoding_results_npy(args.results_dir, args.method_folder, args.eid)
    encoding_npy = encoding_npy.expanduser().resolve()
    if not encoding_npy.is_file():
        raise FileNotFoundError(encoding_npy)

    save_path = args.output
    if save_path is None:
        save_path = encoding_npy.parent / f"top_bps_psth_{args.method}.png"

    plot_top_bps_psth(
        encoding_npy,
        method=args.method,
        eid=args.eid,
        top_bps=args.top_bps,
        duration_s=args.duration_s,
        trials_idx=args.trial_idx,
        normalize=not args.no_normalize,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
