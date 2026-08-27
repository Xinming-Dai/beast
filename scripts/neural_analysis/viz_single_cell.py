#!/usr/bin/env python3
"""Visualize encoding (spike prediction) using ``viz_single_cell`` from ``eval_utils``.

``encoding_results.npy`` stores ``gt`` / ``pred`` with shape ``(n_trials, n_timesteps, n_neurons)``.
Per-trial latent inputs are not stored; we tile ``mean_X`` across trials for ``X``.

**Default:** pick the ``--top-bps`` neurons with highest bits-per-spike (``rates=pred``,
``spikes=gt`` via ``bits_per_spike``) and plot each; the figure shows neuron **index** and **BPS**.

**Single neuron:** set ``--top-bps 0`` and ``--neuron-index <k>``.

**Trial subset:** ``--trial-idx N`` uses the first N trials (0-based indices ``0 .. N-1``).
``--trial-idx K,N`` (or ``K N``) uses trials at indices ``K .. N`` inclusive (0-based).
Omit the flag to use all trials. BPS and plots use only the selected trials.

By default, ``--data-dir`` is the ``combined_latents`` run directory (used only when
``--encoding-npy`` is omitted). Figures go to ``<parent of encoding_results.npy>/figs_single_cell``
unless ``--save-dir`` is set.

By default, figures are produced for **both** ``rrr`` and ``cnn``; pass ``--method rrr`` or
``--method cnn`` for a single model.

python /src/analyses/neural_analysis/viz_single_cell.py \
  --encoding-npy path/to/encoding_results.npy

Note: if you use the new std calculation method, you can't use this script. ValueError: mean_X time dim 1 does not match gt time dim 60. 
Old method works:
    # mean = np.mean(arr, axis=0) # (T, N)
    # std = np.std(arr, axis=0) # (T, N)
    # std = np.clip(std, 1e-8, None) # (T, N)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analyses.neural_analysis.eval_utils import bits_per_spike, viz_single_cell


def find_encoding_results_npy(data_dir: Path) -> Path:
    """Return path to ``encoding_results.npy`` under ``data_dir`` (recursive)."""
    data_dir = data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {data_dir}")
    matches = sorted(data_dir.rglob("encoding_results.npy"))
    if not matches:
        raise FileNotFoundError(
            f"No encoding_results.npy found under {data_dir}"
        )
    if len(matches) > 1:
        import warnings

        warnings.warn(
            f"Multiple encoding_results.npy under {data_dir}; using {matches[0]!s}",
            stacklevel=2,
        )
    return matches[0]


def _encoding_dict(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    inner = raw.item() if raw.dtype == object and raw.shape == () else raw
    if not isinstance(inner, dict):
        raise TypeError(f"Expected dict in .npy, got {type(inner).__name__}")
    return inner


def build_X_from_mean(mean_X: np.ndarray, n_trials: int) -> np.ndarray:
    """(T, F) -> (K, T, F) by repeating the same latent slice for every trial."""
    mean_X = np.asarray(mean_X, dtype=np.float64)
    if mean_X.ndim != 2:
        raise ValueError(f"mean_X must be 2D (T, F), got shape {mean_X.shape}")
    return np.tile(mean_X[np.newaxis, :, :], (n_trials, 1, 1))


def latent_task_metadata() -> tuple[dict, list, dict, list]:
    """Minimal task/behavior labels for PSTH and raster when X is latent-only."""
    var_name2idx = {"z0": [0]}
    var_tasklist = ["z0"]
    var_value2label: dict = {"z0": {}}
    var_behlist = ["z0"]
    return var_name2idx, var_tasklist, var_value2label, var_behlist


def bps_per_neuron(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Bits per spike for each neuron: ``spikes=gt[..., j]``, ``rates=pred[..., j]``."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape or gt.ndim != 3:
        raise ValueError(f"gt/pred must match 3D (K,T,N); got {gt.shape}, {pred.shape}")
    n = gt.shape[2]
    out = np.full(n, np.nan, dtype=np.float64)
    for j in range(n):
        g = gt[..., j : j + 1]
        p = pred[..., j : j + 1]
        if np.nansum(g) <= 0:
            continue
        b = bits_per_spike(p, g)
        if np.isinf(b):
            b = np.nan
        out[j] = b
    return out


def parse_trials_idx_arg(s: str) -> tuple[int, ...]:
    """Parse ``--trial-idx`` as one int (trial count) or two ints (first/last 0-based index)."""
    parts = [p for p in s.replace(",", " ").split() if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--trial-idx: expected N or K,N")
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--trial-idx: values must be integers"
        ) from e
    if len(nums) > 2:
        raise argparse.ArgumentTypeError(
            "--trial-idx: pass one number (N) or two (K, N)"
        )
    return nums


def trials_idx_to_bounds(
    n_trials_full: int, trials_idx: tuple[int, ...] | None
) -> tuple[int, int]:
    """Map 0-based trial selection to ``(start, stop)`` for ``gt[start:stop]`` (stop exclusive)."""
    if trials_idx is None:
        return 0, n_trials_full
    if len(trials_idx) == 1:
        n = int(trials_idx[0])
        if n < 1:
            raise IndexError(
                "--trial-idx N: N must be >= 1 (count of trials from index 0)"
            )
        if n > n_trials_full:
            raise IndexError(
                f"--trial-idx N={n} exceeds n_trials={n_trials_full}"
            )
        return 0, n
    k, last = int(trials_idx[0]), int(trials_idx[1])
    if k < 0 or last < k:
        raise IndexError(
            f"--trial-idx K,N: need 0 <= K <= N, got K={k}, N={last}"
        )
    if last >= n_trials_full:
        raise IndexError(
            f"--trial-idx upper index {last} >= n_trials={n_trials_full}"
        )
    start, stop = k, last + 1
    if stop <= start or stop > n_trials_full:
        raise IndexError(
            f"Trial slice [{start}:{stop}) invalid for n_trials={n_trials_full}"
        )
    return start, stop


def top_neuron_indices_by_bps(bps: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest finite BPS values (descending order)."""
    k = int(k)
    if k <= 0:
        return np.array([], dtype=int)
    finite = np.isfinite(bps)
    if not np.any(finite):
        return np.array([], dtype=int)
    idx = np.nonzero(finite)[0]
    scores = bps[finite]
    order_local = np.argsort(-scores)
    take = min(k, len(order_local))
    return idx[order_local[:take]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory to search for encoding_results.npy (recursive; default eval run combined_latents root)",
    )
    p.add_argument(
        "--encoding-npy",
        type=Path,
        default=None,
        help="Explicit path to encoding_results.npy (skips search under --data-dir)",
    )
    p.add_argument(
        "--method",
        dest="methods",
        nargs="+",
        choices=("rrr", "cnn"),
        default=None,
        metavar="METHOD",
        help="Model block(s) to plot: rrr and/or cnn (default: both)",
    )
    p.add_argument(
        "--top-bps",
        type=int,
        default=5,
        metavar="N",
        help="Plot N neurons with highest BPS (pred vs gt). Use 0 with --neuron-index for a single neuron.",
    )
    p.add_argument(
        "--neuron-index",
        type=int,
        default=0,
        help="Only used when --top-bps is 0: plot exactly this neuron index.",
    )
    p.add_argument(
        "--trial-idx",
        type=parse_trials_idx_arg,
        default=None,
        metavar="N | K,N",
        help="0-based trials: N alone = first N trials (indices 0..N-1); two values = indices K..N "
        "inclusive (e.g. 5,19). Default: all trials.",
    )
    p.add_argument(
        "--subtract-psth",
        choices=("task", "global", "none"),
        default="task",
        help="Passed to viz_single_cell (use 'global' if latent tiling is degenerate)",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for saved figures (default: <encoding_results.npy parent>/figs_single_cell)",
    )
    p.add_argument(
        "--save-name",
        type=str,
        default="",
        help="Optional filename stem (without .png); default encodes eid, method, neuron index",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Also call plt.show() after saving (needs a display)",
    )
    args = p.parse_args()

    methods: list[str] = list(args.methods) if args.methods is not None else ["rrr", "cnn"]

    encoding_npy = (
        args.encoding_npy.expanduser().resolve()
        if args.encoding_npy is not None
        else find_encoding_results_npy(args.data_dir)
    )
    if not encoding_npy.is_file():
        raise FileNotFoundError(encoding_npy)

    save_dir = (
        args.save_dir.expanduser().resolve()
        if args.save_dir is not None
        else encoding_npy.parent / "figs_single_cell"
    )

    root = _encoding_dict(encoding_npy)
    eid = next(iter(root))
    sess = root[eid]

    for m in methods:
        if m not in sess:
            raise KeyError(f"Method {m!r} not in results; keys: {list(sess.keys())}")

    var_name2idx, var_tasklist, var_value2label, var_behlist = latent_task_metadata()

    subtract = None if args.subtract_psth == "none" else args.subtract_psth

    os.makedirs(save_dir, exist_ok=True)

    import matplotlib.pyplot as plt

    print(f"Encoding: {encoding_npy.resolve()}")

    for method in methods:
        block = sess[method]

        gt = np.asarray(block["gt"])
        pred = np.asarray(block["pred"])
        mean_X = np.asarray(block["mean_X"])

        if gt.ndim != 3 or pred.ndim != 3:
            raise ValueError(
                f"Expected gt/pred shaped (K,T,N), got {gt.shape}, {pred.shape}"
            )
        n_trials_full = gt.shape[0]
        t0, t1 = trials_idx_to_bounds(n_trials_full, args.trial_idx)
        if (t0, t1) != (0, n_trials_full):
            print(
                f"  [{method}] Using trial indices {t0}..{t1 - 1} "
                f"(0-based, inclusive), {t1 - t0} of {n_trials_full} trials"
            )
        gt = gt[t0:t1]
        pred = pred[t0:t1]
        n_trials, n_time, n_neurons = gt.shape

        bps_vec = bps_per_neuron(gt, pred)
        if args.top_bps > 0:
            neuron_indices = top_neuron_indices_by_bps(bps_vec, args.top_bps)
            print(
                f"  [{method}] Top {len(neuron_indices)} by BPS (index, bps): "
                + ", ".join(f"({int(i)}, {bps_vec[i]:.4f})" for i in neuron_indices)
            )
        else:
            ni = int(args.neuron_index)
            if not (0 <= ni < n_neurons):
                raise IndexError(f"neuron_index {ni} out of range for n_neurons={n_neurons}")
            neuron_indices = np.array([ni], dtype=int)
            print(
                f"  [{method}] Single neuron {ni}, BPS={bps_vec[ni] if np.isfinite(bps_vec[ni]) else float('nan'):.4f}"
            )

        X = build_X_from_mean(mean_X, n_trials)
        if X.shape[1] != n_time:
            raise ValueError(
                f"mean_X time dim {X.shape[1]} does not match gt time dim {n_time}"
            )

        for ni in neuron_indices:
            ni = int(ni)
            y = gt[:, :, ni]
            y_pred = pred[:, :, ni]
            bps_val = float(bps_vec[ni]) if np.isfinite(bps_vec[ni]) else float("nan")

            r2_psth, r2_trial = viz_single_cell(
                X,
                y,
                y_pred,
                var_name2idx,
                var_tasklist,
                var_value2label,
                var_behlist,
                subtract_psth=subtract,
                aligned_tbins=[],
                clusby="y_pred",
                neuron_idx=str(ni),
                neuron_region="encoding",
                method=method,
                save_path=str(save_dir),
                save_plot=True,
                bps=bps_val,
            )

            default_name = f"encoding_{str(ni)}_{r2_trial:.2f}_{method}.png"
            out_path = save_dir / default_name
            if args.save_name.strip():
                renamed = save_dir / f"{args.save_name.strip()}_{ni}_{method}.png"
                if out_path.is_file():
                    os.replace(out_path, renamed)
                    out_path = renamed

            print(
                f"    plot idx={ni}  BPS={bps_val:.6g}  PSTH R²={r2_psth:.6g}  trial R²={r2_trial:.6g}  -> {out_path.resolve()}"
            )

            if args.show:
                plt.show()
            plt.close("all")


if __name__ == "__main__":
    main()
