"""Extract exact eval-layout frames for an IBL session from raw two-view videos.

SABLE/IBL analog of ``extract_cheese3d_eval_frames.py``. Reads the ``{eid}_aligned.npz``
written by ``extract_sable_neural_data.py`` (per-split ``{split}_intervals``, shape
``(n_trials, 2)``, seconds), rebuilds the same 60-bin-per-trial time grid used to
interpolate behaviors during neural extraction (bin edges at
``t_beg + (bin_idx + 1) * binsize`` for ``binsize = 1 / 60``), finds the nearest raw video
frame to each bin edge via the camera's own timestamps, and pulls those frames with a single
sequential OpenCV decode pass per camera video — cheap relative to seeking per frame, and,
unlike an ffmpeg ``select`` filter listing every wanted frame in one argument, has no OS
argument-length limit to run into even when tens of thousands of frames are wanted per
camera (e.g. 400 trials x 60 bins/trial).

Frames are written to::

    {output_dir}/{camera}Camera.video/_iblrig_{camera}Camera.downsampled.{eid}/
        {split}/interval{trial_idx}timebin{bin_idx}.png

matching the "eval layout"
``beast.data.sable_dataset.SABLEDataset._discover_eval_split_records``
expects, alongside a ``frame_index_mapping.json`` per split keyed by filename with
``{camera}_source_frame_index``, ``neural_trial_idx``, ``neural_bin_idx``, and
``neural_interval_sec`` — matching the layout already on disk for previously-extracted
sessions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from beast.logging import log_step

_DEFAULT_RAW_VIDEO_DIR = '/work/hdd/bfsr/xdai3/IBL-2view'
_ALL_CAMERAS = ('left', 'right')
_SPLITS = ('train', 'val', 'test')
_BINSIZE = 1 / 60
_N_BINS = 60


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; ``None`` uses ``sys.argv``.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Extract eval-layout frames for an IBL session from raw two-view videos.',
    )
    parser.add_argument('--eid', type=str, required=True)
    parser.add_argument(
        '--aligned-npz', type=str, default=None,
        help='path to {eid}_aligned.npz; defaults to {neural-data-dir}/{eid}/{eid}_aligned.npz.',
    )
    parser.add_argument(
        '--neural-data-dir', type=str, default=None,
        help='root written by extract_sable_neural_data.py; used when --aligned-npz is unset.',
    )
    parser.add_argument('--raw-video-dir', type=str, default=_DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=list(_ALL_CAMERAS),
        help="cameras to extract frames for (default: 'left right').",
    )
    return parser.parse_args(argv)


def _resolve_aligned_npz(args: argparse.Namespace) -> Path:
    """Resolve the path to ``{eid}_aligned.npz`` from CLI args.

    Args:
        args: parsed CLI arguments.

    Raises:
        ValueError: if neither ``--aligned-npz`` nor ``--neural-data-dir`` was given.

    Returns:
        path to the aligned npz.
    """
    if args.aligned_npz is not None:
        return Path(args.aligned_npz)
    if args.neural_data_dir is not None:
        return Path(args.neural_data_dir) / args.eid / f'{args.eid}_aligned.npz'
    raise ValueError('one of --aligned-npz or --neural-data-dir is required.')


def _bin_edge_times(intervals: np.ndarray) -> np.ndarray:
    """Compute the 60 bin-edge times per trial, matching the neural-extraction time grid.

    Args:
        intervals: array of shape ``(n_trials, 2)`` of ``[t_beg, t_end]`` per trial (seconds).

    Returns:
        array of shape ``(n_trials, 60)`` of bin-edge times.
    """
    t_beg = intervals[:, 0]
    bin_offsets = (np.arange(_N_BINS) + 1) * _BINSIZE
    return t_beg[:, None] + bin_offsets[None, :]


def _nearest_frame_idxs(timestamps: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Find the nearest video frame index for each query time.

    Args:
        timestamps: ascending per-frame timestamps (seconds) for one camera.
        query_times: times to look up, any shape.

    Returns:
        array (same shape as ``query_times``) of 0-based frame indices.
    """
    flat = query_times.ravel()
    idx = np.searchsorted(timestamps, flat, side='left')
    idx = np.clip(idx, 1, len(timestamps) - 1)
    left = idx - 1
    use_left = np.abs(timestamps[left] - flat) <= np.abs(timestamps[idx] - flat)
    nearest = np.where(use_left, left, idx)
    return nearest.reshape(query_times.shape)


def _extract_frames_for_camera(
    video_path: Path,
    frame_numbers: list[int],
    dest_paths_by_frame: dict[int, list[Path]],
) -> None:
    """Decode ``video_path`` once and write each selected frame to its destination path(s).

    A single sequential ``cv2.VideoCapture`` read pass, stopping once the highest wanted
    frame index has been reached — this avoids an ffmpeg ``select`` filter argument (one
    term per wanted frame) blowing past the OS's per-argument length limit when tens of
    thousands of frames are wanted.

    Args:
        video_path: source MP4.
        frame_numbers: sorted, deduplicated list of 0-based frame indices to extract.
        dest_paths_by_frame: maps each frame number to the destination path(s) it should be
            written to (a raw frame can back more than one trial/bin if timestamps are dense
            enough to share a nearest frame).

    Raises:
        RuntimeError: if the video ends before all wanted frames were found.
    """
    if not frame_numbers:
        return

    wanted = set(frame_numbers)
    last_wanted = frame_numbers[-1]
    n_found = 0

    log_step(f'decoding {video_path.name} for {len(wanted)} frames')
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_idx = 0
        while frame_idx <= last_wanted:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in wanted:
                for dest_path in dest_paths_by_frame[frame_idx]:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest_path), frame)
                n_found += 1
            frame_idx += 1
    finally:
        cap.release()

    if n_found != len(wanted):
        raise RuntimeError(
            f'{video_path}: expected {len(wanted)} frames, found {n_found}',
        )


def main(argv: list[str] | None = None) -> None:
    """Extract eval-layout frames and frame-index mappings for one IBL session."""
    args = parse_args(argv)
    eid = args.eid
    raw_video_dir = Path(args.raw_video_dir)
    output_dir = Path(args.output_dir)

    aligned = np.load(_resolve_aligned_npz(args))

    for camera in args.cameras:
        video_path = raw_video_dir / f'{camera}Camera.video' / (
            f'_iblrig_{camera}Camera.downsampled.{eid}.mp4'
        )
        timestamps_path = raw_video_dir / 'timestamps' / f'_ibl_{camera}Camera.times.{eid}.npy'
        if not timestamps_path.is_file():
            # right camera is synchronized to and shares the left camera's timestamps.
            timestamps_path = raw_video_dir / 'timestamps' / f'_ibl_leftCamera.times.{eid}.npy'
        timestamps = np.load(timestamps_path)

        session_dir = output_dir / f'{camera}Camera.video' / (
            f'_iblrig_{camera}Camera.downsampled.{eid}'
        )

        split_trial_frame_idxs: dict[str, np.ndarray] = {}
        frame_to_dests: dict[int, list[Path]] = {}
        for split in _SPLITS:
            key = f'{split}_intervals'
            if key not in aligned:
                continue
            intervals = aligned[key]
            if len(intervals) == 0:
                split_trial_frame_idxs[split] = np.zeros((0, _N_BINS), dtype=np.int64)
                continue
            bin_times = _bin_edge_times(intervals)
            frame_idxs = _nearest_frame_idxs(timestamps, bin_times)
            split_trial_frame_idxs[split] = frame_idxs

            for trial_idx, bin_idx in np.ndindex(frame_idxs.shape):
                frame_number = int(frame_idxs[trial_idx, bin_idx])
                dest = session_dir / split / f'interval{trial_idx}timebin{bin_idx}.png'
                frame_to_dests.setdefault(frame_number, []).append(dest)

        frame_numbers = sorted(frame_to_dests)
        _extract_frames_for_camera(video_path, frame_numbers, frame_to_dests)
        log_step(f'extracted {len(frame_numbers)} frames for camera {camera!r}')

        for split, intervals_key in ((s, f'{s}_intervals') for s in _SPLITS):
            if intervals_key not in aligned or split not in split_trial_frame_idxs:
                continue
            intervals = aligned[intervals_key]
            frame_idxs = split_trial_frame_idxs[split]
            mapping = {
                f'interval{trial_idx}timebin{bin_idx}.png': {
                    f'{camera}_source_frame_index': int(frame_idxs[trial_idx, bin_idx]),
                    'neural_trial_idx': int(trial_idx),
                    'neural_bin_idx': int(bin_idx),
                    'neural_interval_sec': intervals[trial_idx].tolist(),
                }
                for trial_idx, bin_idx in np.ndindex(frame_idxs.shape)
            }
            split_dir = session_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            mapping_path = split_dir / 'frame_index_mapping.json'
            with open(mapping_path, 'w') as f:
                json.dump(mapping, f, indent=2)
            log_step(f'wrote {mapping_path}')


if __name__ == '__main__':
    main()
