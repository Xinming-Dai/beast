"""Build a beast-compatible ``<eid>_aligned.npz`` from Cheese3D ephys trigger data.

Cheese3D analog of IBL's (external) ``extract_neural_data.py``
(see ``/u/xdai3/project3d/erayzer_cls/E-RayZer-private/docs/ibl_neural_behavior_extraction.md``).
Reads the trigger-synchronized alignment produced by the Cheese3D dataset's own
``align_ephys.py`` (``spike/<eid>_100fps.npz`` — hardware-clock-accurate per-camera frame
indices and per-trigger spike counts), bins spikes into fixed-length trial windows, filters
units by mean firing rate, and splits trials into train/val/test.

Writes two artifacts under ``--neural-output-dir/<eid>/``:

* ``<eid>_aligned.npz`` — ``train/val/test_spikes`` and ``train/val/test_intervals``, matching
  the key contract ``beast.sable_encoding_decoding.neural.run_encoding_decoding`` expects.
* ``frame_manifest.json`` — per split, per trial, the raw video frame index for all six Cheese3D
  cameras (``BC``, ``L``, ``R``, ``TC``, ``TL``, ``TR``), consumed by
  ``extract_cheese3d_eval_frames.py`` to pull the literal corresponding frames from
  ``videos_ephys/*.mp4``. Keeping all six cameras (rather than only the pair a given SABLE config
  trains on) means a future config with a different camera pairing can reuse this manifest
  without re-deriving trial windows from the trigger CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beast.logging import log_step

_DEFAULT_ALIGNMENT_NPZ = (
    '/work/hdd/bfsr/xdai3/cheese3d/spike/20250523_B1_ephys-record_awake_000_100fps.npz'
)
_DEFAULT_EID = '20250523_B1_ephys-record_awake_000'
_ALL_CAMERAS = ('BC', 'L', 'R', 'TC', 'TL', 'TR')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; ``None`` uses ``sys.argv``.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build <eid>_aligned.npz and a frame manifest from Cheese3D ephys data.',
    )
    parser.add_argument('--alignment-npz', type=str, default=_DEFAULT_ALIGNMENT_NPZ)
    parser.add_argument('--eid', type=str, default=_DEFAULT_EID)
    parser.add_argument('--neural-output-dir', type=str, required=True)
    parser.add_argument('--trial-len-sec', type=float, default=1.0)
    parser.add_argument(
        '--num-trials',
        type=int,
        default=None,
        help='subsample this many trial windows (seed 42); default keeps all non-overlapping '
             'windows spanning the session',
    )
    parser.add_argument(
        '--fr-thresh',
        type=float,
        default=0.2,
        help='keep units whose mean firing rate over the full session exceeds this (Hz)',
    )
    parser.add_argument('--train-frac', type=float, default=0.7)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(argv)


def _build_trial_windows(
    timestamps_s: np.ndarray,
    trial_len_sec: float,
    num_trials: int | None,
    seed: int,
) -> np.ndarray:
    """Choose non-overlapping trial window start times spanning the session.

    Args:
        timestamps_s: per-trigger session-relative timestamps, ascending.
        trial_len_sec: width of each trial window in seconds.
        num_trials: if set, randomly subsample this many windows (seed-controlled);
            otherwise every window is kept.
        seed: RNG seed for subsampling.

    Returns:
        sorted array of window start times (seconds).
    """
    session_end = float(timestamps_s[-1])
    n_windows = int(session_end // trial_len_sec)
    starts = np.arange(n_windows, dtype=np.float64) * trial_len_sec
    if num_trials is not None and num_trials < len(starts):
        rng = np.random.RandomState(seed)
        chosen = rng.choice(len(starts), size=num_trials, replace=False)
        starts = np.sort(starts[chosen])
    return starts


def _bin_trials(
    timestamps_s: np.ndarray,
    spike_counts: np.ndarray,
    frame_indices: np.ndarray,
    starts: np.ndarray,
    trial_len_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum trigger-level spike counts into trial windows and pick a representative frame.

    The representative frame per camera is taken from the first trigger row inside the
    window (mirrors ``align_ephys.py``'s ``downsample(..., reduce='first')`` when
    down-sampling the trigger stream to a lower target fps).

    Args:
        timestamps_s: per-trigger session-relative timestamps, ascending.
        spike_counts: ``[N, n_units]`` per-trigger spike counts.
        frame_indices: ``[N, n_cameras]`` per-trigger, per-camera 0-based MP4 frame index.
        starts: trial window start times (seconds), ascending.
        trial_len_sec: trial window width in seconds.

    Returns:
        Tuple of ``(trial_spikes [K, n_units], trial_intervals [K, 2], trial_frame_idx
        [K, n_cameras])``; windows with zero trigger rows are dropped.
    """
    ends = starts + trial_len_sec
    lo = np.searchsorted(timestamps_s, starts, side='left')
    hi = np.searchsorted(timestamps_s, ends, side='left')

    trial_spikes = []
    trial_intervals = []
    trial_frame_idx = []
    for start, end, i_lo, i_hi in zip(starts, ends, lo, hi):
        if i_hi <= i_lo:
            continue
        trial_spikes.append(spike_counts[i_lo:i_hi].sum(axis=0))
        trial_intervals.append((start, end))
        trial_frame_idx.append(frame_indices[i_lo])

    return (
        np.asarray(trial_spikes, dtype=np.int32),
        np.asarray(trial_intervals, dtype=np.float64),
        np.asarray(trial_frame_idx, dtype=np.int32),
    )


def _filter_units_by_firing_rate(
    spike_counts: np.ndarray,
    timestamps_s: np.ndarray,
    cluster_names: np.ndarray,
    fr_thresh: float,
) -> np.ndarray:
    """Return the boolean keep-mask for units whose mean session firing rate exceeds threshold.

    Args:
        spike_counts: ``[N, n_units]`` per-trigger spike counts over the full session.
        timestamps_s: per-trigger session-relative timestamps, ascending.
        cluster_names: ``[n_units]`` unit names, for logging.
        fr_thresh: minimum mean firing rate (Hz) to keep a unit.

    Returns:
        boolean array ``[n_units]``, ``True`` for units to keep.
    """
    duration = float(timestamps_s[-1] - timestamps_s[0])
    rates = spike_counts.sum(axis=0).astype(np.float64) / duration
    keep = rates > fr_thresh
    for name, rate, kept in zip(cluster_names, rates, keep):
        log_step(f'unit {name}: mean rate {rate:.3f} Hz -> {"keep" if kept else "drop"}')
    return keep


def _split_trials(
    n_trials: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Shuffle and split trial indices into train/val/test.

    Args:
        n_trials: total number of trials.
        train_frac: fraction assigned to train.
        val_frac: fraction assigned to val.
        test_frac: fraction assigned to test (unused directly; remainder after train/val).
        seed: RNG seed for the shuffle.

    Returns:
        dict mapping split name to an array of trial indices (into the original, unshuffled
        trial arrays).
    """
    del test_frac  # remainder after train/val; kept for CLI symmetry with IBL's convention
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_trials)
    n_train = round(train_frac * n_trials)
    n_val = round(val_frac * n_trials)
    return {
        'train': order[:n_train],
        'val': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }


def main(argv: list[str] | None = None) -> None:
    """Build ``<eid>_aligned.npz`` and ``frame_manifest.json`` for one Cheese3D ephys session."""
    args = parse_args(argv)

    alignment = np.load(args.alignment_npz, allow_pickle=True)
    timestamps_s = np.asarray(alignment['timestamps_s'], dtype=np.float64)
    spike_counts = np.asarray(alignment['spike_counts'], dtype=np.int32)
    frame_indices = np.asarray(alignment['frame_indices'], dtype=np.int32)
    cluster_names = np.asarray(alignment['cluster_names'])
    view_names = [str(v) for v in alignment['view_names']]
    if list(view_names) != list(_ALL_CAMERAS):
        raise ValueError(f'unexpected view_names order {view_names}; expected {_ALL_CAMERAS}')

    keep_units = _filter_units_by_firing_rate(
        spike_counts, timestamps_s, cluster_names, args.fr_thresh,
    )
    n_units_kept = int(keep_units.sum())
    if n_units_kept == 0:
        raise ValueError(f'--fr-thresh={args.fr_thresh} drops all units; lower the threshold.')
    log_step(f'keeping {n_units_kept}/{len(cluster_names)} units')

    starts = _build_trial_windows(timestamps_s, args.trial_len_sec, args.num_trials, args.seed)
    trial_spikes, trial_intervals, trial_frame_idx = _bin_trials(
        timestamps_s, spike_counts, frame_indices, starts, args.trial_len_sec,
    )
    trial_spikes = trial_spikes[:, keep_units]
    n_trials = len(trial_spikes)
    log_step(f'{n_trials} trial windows of {args.trial_len_sec}s built')

    splits = _split_trials(n_trials, args.train_frac, args.val_frac, args.test_frac, args.seed)

    neural_output_dir = Path(args.neural_output_dir) / args.eid
    neural_output_dir.mkdir(parents=True, exist_ok=True)

    npz_kwargs: dict[str, np.ndarray] = {}
    manifest_splits: dict[str, list[dict]] = {}
    for split_name, idx in splits.items():
        # T=1 timestep per trial (one bin per 1s window).
        npz_kwargs[f'{split_name}_spikes'] = trial_spikes[idx][:, None, :]
        npz_kwargs[f'{split_name}_intervals'] = trial_intervals[idx]

        manifest_splits[split_name] = [
            {
                'neural_trial_idx': local_i,
                'neural_bin_idx': 0,
                'neural_interval_sec': trial_intervals[global_i].tolist(),
                'frame_index': {
                    cam: int(trial_frame_idx[global_i, cam_i])
                    for cam_i, cam in enumerate(_ALL_CAMERAS)
                },
            }
            for local_i, global_i in enumerate(idx)
        ]

    aligned_npz_path = neural_output_dir / f'{args.eid}_aligned.npz'
    np.savez(aligned_npz_path, **npz_kwargs)
    log_step(f'wrote {aligned_npz_path}')

    params = {
        'eid': args.eid,
        'alignment_npz': str(args.alignment_npz),
        'trial_len_sec': args.trial_len_sec,
        'num_trials_requested': args.num_trials,
        'n_trials': n_trials,
        'fr_thresh': args.fr_thresh,
        'cluster_names_kept': cluster_names[keep_units].tolist(),
        'train_frac': args.train_frac,
        'val_frac': args.val_frac,
        'test_frac': args.test_frac,
        'seed': args.seed,
        'split_sizes': {k: len(v) for k, v in splits.items()},
    }
    params_path = neural_output_dir / 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    log_step(f'wrote {params_path}')

    manifest = {
        'eid': args.eid,
        'view_names': list(_ALL_CAMERAS),
        'splits': manifest_splits,
    }
    manifest_path = neural_output_dir / 'frame_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    log_step(f'wrote {manifest_path}')


if __name__ == '__main__':
    main()
