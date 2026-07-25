"""Extract exact eval-layout frames for a Cheese3D ephys session from raw videos.

Reads the ``frame_manifest.json`` written by ``extract_cheese3d_neural_data.py`` (per-trial,
per-camera raw MP4 frame indices, split into train/val/test) and pulls the literal
corresponding frames from the session's raw camera videos with a single ffmpeg ``select``-filter
decode pass per camera — cheap relative to seeking per frame, since it only requires one
sequential decode of each ~20-minute session video. Defaults to
``/work/hdd/bfsr/xdai3/cheese3d/videos/`` rather than the dataset's own ``videos_ephys/``
because, for this session, ``videos_ephys/``'s ``TR`` file is truncated (ffprobe: "moov atom not
found") while ``videos/`` has an intact copy of all six cameras — see
``docs/sable/neural_extraction.md``.

Extracts all six Cheese3D cameras (``BC``, ``L``, ``R``, ``TC``, ``TL``, ``TR``) by default so a
future SABLE config with a different camera pairing doesn't require re-decoding the videos;
restrict with ``--cameras``. Frames are written to::

    {output_dir}/{eid}/{camera}/{split}/interval{trial_idx}timebin0.png

matching the "eval layout" ``beast.data.sable_dataset.SABLEDataset._discover_eval_split_records``
expects. That method's on-disk contract keys frame indices by stereo *role*
(``left_source_frame_index`` / ``right_source_frame_index``, not by camera name), so this script
additionally writes ``frame_index_mapping.json`` — with those role-specific keys — into the
``--left-camera``/``--right-camera`` (and optional ``--center-camera``) directories only; the
other extracted cameras' frames sit on disk without a mapping file until a future run assigns
them a role.

Also copies a static per-camera calibration ``.npy`` sidecar (intrinsics/extrinsics; identical
for every frame of a given camera, matching ``Cheese3DDataset``'s expected
``{camera}/{split}/interval{N}timebin{M}.npy`` layout) next to each extracted PNG. This is
required even when ``training.use_camera_params`` is ``false``, because
``training.load_gt_camera_params_for_vis`` (set in
``configs/sable/sable_cheese3d_ephys_session.yaml``) unconditionally loads it for
visualization, and ``beast predict`` has no config-override mechanism to disable that per run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from beast.logging import log_step

_DEFAULT_RAW_VIDEO_DIR = '/work/hdd/bfsr/xdai3/cheese3d/videos'
_DEFAULT_CALIBRATION_DIR = '/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam'
_ALL_CAMERAS = ('BC', 'L', 'R', 'TC', 'TL', 'TR')
_RESIZE_WIDTH, _RESIZE_HEIGHT = 320, 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; ``None`` uses ``sys.argv``.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Extract eval-layout frames for a Cheese3D ephys session.',
    )
    parser.add_argument('--frame-manifest', type=str, required=True)
    parser.add_argument('--raw-video-dir', type=str, default=_DEFAULT_RAW_VIDEO_DIR)
    parser.add_argument(
        '--calibration-dir', type=str, default=_DEFAULT_CALIBRATION_DIR,
        help="root containing {eid}/{camera}/img00000000.npy static calibration sidecars "
             '(reused verbatim — the same intrinsics/extrinsics apply to every frame)',
    )
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument(
        '--cameras', type=str, nargs='+', default=list(_ALL_CAMERAS),
        help='cameras to extract frames for (default: all six)',
    )
    parser.add_argument('--left-camera', type=str, default='TL')
    parser.add_argument('--right-camera', type=str, default='TR')
    parser.add_argument(
        '--center-camera', type=str, default=None,
        help='optional third camera; writes a center_source_frame_index mapping too',
    )
    return parser.parse_args(argv)


def _find_camera_video(raw_video_dir: Path, eid: str, camera: str) -> Path:
    """Find the single MP4 for one camera in the raw ephys video directory.

    Args:
        raw_video_dir: directory containing ``{eid}_{camera}_{HH-MM-SS}.mp4`` files.
        eid: session id.
        camera: camera name, e.g. ``'TL'``.

    Returns:
        path to the matching MP4.

    Raises:
        FileNotFoundError: if zero or more than one match is found.
    """
    matches = sorted(raw_video_dir.glob(f'{eid}_{camera}_*.mp4'))
    if len(matches) != 1:
        raise FileNotFoundError(
            f'expected exactly one {camera!r} video under {raw_video_dir}, found {matches}',
        )
    return matches[0]


def _extract_frames_for_camera(
    video_path: Path,
    frame_numbers: list[int],
    dest_paths_by_frame: dict[int, list[Path]],
    calibration_npy: Path | None,
) -> None:
    """Decode ``video_path`` once and copy each selected frame to its destination path(s).

    Args:
        video_path: source MP4.
        frame_numbers: sorted, deduplicated list of 0-based frame indices to extract.
        dest_paths_by_frame: maps each frame number to the destination path(s) it should be
            written to (usually one, but a raw frame could in principle back more than one
            trial).
        calibration_npy: static per-camera calibration file to copy alongside each extracted
            PNG (same basename, ``.npy`` suffix); ``None`` skips this.
    """
    select_expr = '+'.join(f'eq(n\\,{n})' for n in frame_numbers)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_pattern = str(Path(tmp_dir) / '%06d.png')
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', f"select='{select_expr}',scale={_RESIZE_WIDTH}:{_RESIZE_HEIGHT}",
            '-vsync', '0', '-start_number', '0',
            tmp_pattern,
        ]
        log_step(f'running ffmpeg for {video_path.name} ({len(frame_numbers)} frames)')
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'ffmpeg failed for {video_path}:\n{result.stderr}')

        extracted = sorted(Path(tmp_dir).glob('*.png'))
        if len(extracted) != len(frame_numbers):
            raise RuntimeError(
                f'{video_path}: expected {len(frame_numbers)} extracted frames, '
                f'got {len(extracted)}',
            )
        for frame_number, src_path in zip(frame_numbers, extracted):
            for dest_path in dest_paths_by_frame[frame_number]:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                if calibration_npy is not None:
                    shutil.copy2(calibration_npy, dest_path.with_suffix('.npy'))


def _write_role_mapping(
    camera_dir: Path,
    split: str,
    role_key: str,
    trials: list[dict],
) -> None:
    """Write ``frame_index_mapping.json`` for one camera/split using the given role key.

    Args:
        camera_dir: e.g. ``{output_dir}/{eid}/TL``.
        split: split name (``'train'``, ``'val'``, or ``'test'``).
        role_key: ``'left'``, ``'right'``, or ``'center'`` — determines the JSON field name
            (``f'{role_key}_source_frame_index'``) per
            ``SABLEDataset._discover_eval_split_records``'s contract.
        trials: this split's trial entries from the frame manifest.
    """
    split_dir = camera_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    camera = camera_dir.name
    mapping = {
        f'interval{trial["neural_trial_idx"]}timebin{trial["neural_bin_idx"]}.png': {
            f'{role_key}_source_frame_index': trial['frame_index'][camera],
            'neural_trial_idx': trial['neural_trial_idx'],
            'neural_bin_idx': trial['neural_bin_idx'],
            'neural_interval_sec': trial['neural_interval_sec'],
        }
        for trial in trials
    }
    mapping_path = split_dir / 'frame_index_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    log_step(f'wrote {mapping_path}')


def main(argv: list[str] | None = None) -> None:
    """Extract eval-layout frames and role-based mapping files for a Cheese3D ephys session."""
    args = parse_args(argv)

    with open(args.frame_manifest) as f:
        manifest = json.load(f)
    eid = manifest['eid']
    raw_video_dir = Path(args.raw_video_dir)
    calibration_dir = Path(args.calibration_dir)
    output_dir = Path(args.output_dir) / eid

    for camera in args.cameras:
        video_path = _find_camera_video(raw_video_dir, eid, camera)
        calibration_npy = calibration_dir / eid / camera / 'img00000000.npy'
        if not calibration_npy.is_file():
            log_step(
                f'no calibration sidecar at {calibration_npy}; extracted frames for '
                f'{camera!r} will have no .npy sidecar',
                level='warning',
            )
            calibration_npy = None

        frame_to_dests: dict[int, list[Path]] = {}
        for split, trials in manifest['splits'].items():
            for trial in trials:
                frame_number = trial['frame_index'][camera]
                dest = (
                    output_dir / camera / split
                    / f'interval{trial["neural_trial_idx"]}timebin{trial["neural_bin_idx"]}.png'
                )
                frame_to_dests.setdefault(frame_number, []).append(dest)

        frame_numbers = sorted(frame_to_dests)
        _extract_frames_for_camera(video_path, frame_numbers, frame_to_dests, calibration_npy)
        log_step(f'extracted {len(frame_numbers)} frames for camera {camera}')

        camera_dir = output_dir / camera
        if camera == args.left_camera:
            for split, trials in manifest['splits'].items():
                _write_role_mapping(camera_dir, split, 'left', trials)
        if camera == args.right_camera:
            for split, trials in manifest['splits'].items():
                _write_role_mapping(camera_dir, split, 'right', trials)
        if args.center_camera and camera == args.center_camera:
            for split, trials in manifest['splits'].items():
                _write_role_mapping(camera_dir, split, 'center', trials)


if __name__ == '__main__':
    main()
