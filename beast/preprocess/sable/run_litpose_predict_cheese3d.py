"""Run Lightning Pose predict on Cheese3D sessions.

Cheese3D videos live in a single flat directory, named
``{session_id}_{cam}_{HH-MM-SS}.mp4`` (e.g.
``20250523_B1_ephys-record_awake_000_TL_18-24-03.mp4``), with cameras
``BC, L, R, TC, TL, TR``. This differs from the SABLE IBL-2view layout (one
subdirectory per camera, exact filename template), so this script discovers
sessions/videos directly instead of reusing ``extraction_sable.py``'s
``VideoNamingConfig``.

Lightning Pose's own view-matching (``lightning_pose.utils.io.collect_video_files_by_view``)
matches a camera view to a video by searching for the view name as a substring of the
filename, but its left-boundary check is broken (``(?<!0-9a-zA-Z)`` is a literal-string
lookbehind, not a character class, so it never actually blocks anything). Since ``'L'`` is
a suffix of ``'TL'`` (and ``'R'`` of ``'TR'``), this raises ``"File matches multiple
views"`` once all 6 Cheese3D cameras are passed together. This script works around it by
calling the Lightning Pose Python API directly (rather than shelling out to the ``litpose``
CLI) and monkeypatching in a corrected matcher with a proper character-class boundary.

Cameras also occasionally disagree on frame count by a frame or two (e.g. one camera drops
the last frame). Lightning Pose's multiview DALI reader shares a shuffle seed across per-view
readers to keep them frame-aligned, so it refuses to run when a session's cameras don't all
report the same frame count (``ValueError: Mismatched frame counts across views``). This
script checks frame counts up front and trims any camera video that's longer than the
session's minimum down to that length (keeping frames ``[0, min_count)``), writing the
trimmed copy under ``--trimmed-videos-dir`` and leaving the original video untouched.

Run this script in an environment where Lightning Pose is available, either by passing
``--litpose-repo`` (Lightning Pose source repo; no install required) or by activating the
``lp`` conda environment.

Usage::

    python beast/preprocess/sable/run_litpose_predict_cheese3d.py \\
      --root /work/hdd/bfsr/xdai3/cheese3d/videos \\
      --model-dir /path/to/cheese3d_lightning_pose_model \\
      --litpose-repo /u/xdai3/project3d/lightning-pose \\
      --session-ids 20250523_B1_ephys-record_awake_000 \\
      [--skip-existing] [--skip-viz] [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from beast.video import get_video_stats, trim_video

_CHEESE3D_CAMERAS = ['BC', 'L', 'R', 'TC', 'TL', 'TR']

# the trailing HH-MM-SS timestamp anchors the split between session_id and cam, so this
# has exactly one valid match per filename regardless of alternation order (unlike
# lightning-pose's own substring matcher, which is ambiguous between 'L'/'TL' and 'R'/'TR')
_CHEESE3D_FILENAME_RE = re.compile(
    r'^(?P<session_id>.+)_(?P<cam>BC|TC|TL|TR|L|R)_(?P<timestamp>\d{2}-\d{2}-\d{2})\.mp4$'
)


def discover_sessions(
    root: Path,
    cameras: list[str],
    session_ids: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Discover Cheese3D sessions and their per-camera video paths.

    Args:
        root: flat directory containing all Cheese3D session videos.
        cameras: camera names to require for a session to be included.
        session_ids: optional explicit list of session IDs to filter to.

    Returns:
        dict mapping session_id to a dict of {cam: video_path}, for sessions that have
        every camera in ``cameras`` present.

    Raises:
        FileNotFoundError: if root is not a directory.
        RuntimeError: if no matching sessions are found.
    """
    if not root.is_dir():
        raise FileNotFoundError(f'video directory not found: {root}')

    cam_set = set(cameras)
    by_session: dict[str, dict[str, Path]] = {}
    for f in root.iterdir():
        m = _CHEESE3D_FILENAME_RE.match(f.name)
        if m is None or m.group('cam') not in cam_set:
            continue
        by_session.setdefault(m.group('session_id'), {})[m.group('cam')] = f

    sessions = {
        sid: cam_videos
        for sid, cam_videos in by_session.items()
        if cam_set.issubset(cam_videos)
    }

    if session_ids:
        requested = set(session_ids)
        missing = sorted(requested - set(sessions))
        if missing:
            print(f'warning: session IDs not found (or missing cameras): {missing}', file=sys.stderr)
        sessions = {sid: v for sid, v in sessions.items() if sid in requested}

    if not sessions:
        raise RuntimeError(f'no matching session IDs found under {root} (cameras={cameras})')
    return sessions


def _fixed_collect_video_files_by_view(
    video_files: list[Path],
    view_names: list[str],
) -> dict[str, Path]:
    """Corrected replacement for lightning_pose.utils.io.collect_video_files_by_view.

    Same contract as the original (filenames must contain their view's name, delimited by
    non-alphanumeric characters), but with the boundary regex actually fixed: the original
    uses ``(?<!0-9a-zA-Z)``, a literal-string lookbehind rather than a character class, so it
    never blocks a match. That lets ``'L'`` match inside ``'TL'`` (and ``'R'`` inside
    ``'TR'``), which raises a false "matches multiple views" error for Cheese3D's camera
    names. Using ``[0-9A-Za-z]`` here fixes that.

    Args:
        video_files: candidate video paths.
        view_names: view names to find a matching video for.

    Returns:
        dict mapping view_name to its matched video path.

    Raises:
        ValueError: if a view matches zero or more than one video file.
    """
    video_files_by_view: dict[str, Path] = {}
    for view_name in view_names:
        for video_file in video_files:
            if re.search(
                rf'(?<![0-9A-Za-z]){re.escape(view_name)}(?![0-9A-Za-z])', video_file.stem
            ):
                if view_name not in video_files_by_view:
                    video_files_by_view[view_name] = video_file
                else:
                    raise ValueError(f'File matches multiple views: {video_file}')
        if view_name not in video_files_by_view:
            raise ValueError(f'File not found for view: {view_name}')
    return video_files_by_view


def _is_valid_trim(path: Path, expected_frames: int) -> bool:
    """Check whether a cached trimmed video is complete and has the expected frame count.

    A trim can be left truncated (e.g. missing the trailing moov atom) if a previous job
    was killed mid-``ffmpeg`` (walltime, preemption); ``get_video_stats`` raises ``OSError``
    on such files, so that's treated as invalid rather than propagating.

    Args:
        path: path to the cached trimmed video.
        expected_frames: frame count the video should have.

    Returns:
        True if the video is readable and has exactly ``expected_frames`` frames.
    """
    if not path.is_file():
        return False
    try:
        return get_video_stats(path)['total_frames'] == expected_frames
    except OSError:
        return False


def sync_session_frame_counts(
    cam_videos: dict[str, Path],
    trimmed_dir: Path,
) -> dict[str, Path]:
    """Trim any camera video longer than the session's minimum frame count.

    Lightning Pose's multiview DALI reader requires every view of a session to have the
    exact same frame count. Cameras that already match the session minimum are passed
    through unchanged; longer ones are trimmed to ``[0, min_count)`` and written under
    ``trimmed_dir`` (skipped if already trimmed).

    Args:
        cam_videos: mapping of camera name to source mp4 path.
        trimmed_dir: directory to write trimmed copies to.

    Returns:
        mapping of camera name to the path to use for prediction (original or trimmed).
    """
    frame_counts = {cam: get_video_stats(vpath)['total_frames'] for cam, vpath in cam_videos.items()}
    min_count = min(frame_counts.values())
    if len(set(frame_counts.values())) == 1:
        return cam_videos

    synced: dict[str, Path] = {}
    for cam, vpath in cam_videos.items():
        if frame_counts[cam] == min_count:
            synced[cam] = vpath
            continue
        out_path = trimmed_dir / vpath.name
        if not _is_valid_trim(out_path, min_count):
            # trim into a temp file and rename atomically, so a job killed mid-ffmpeg
            # (walltime, preemption) never leaves a truncated file at out_path for a later
            # run to mistake for a completed trim
            tmp_path = out_path.with_suffix('.mp4.tmp')
            trim_video(vpath, tmp_path, start_frame=0, end_frame=min_count - 1)
            tmp_path.replace(out_path)
        synced[cam] = out_path
    return synced


def _should_skip_session(model_dir: Path, cam_videos: dict[str, Path]) -> bool:
    """Return True if all prediction CSVs already exist for a session.

    Args:
        model_dir: Lightning Pose model directory.
        cam_videos: mapping of camera name to mp4 path.

    Returns:
        True if all per-camera prediction CSVs are present under ``video_preds/``.
    """
    vp = model_dir / 'video_preds'
    return all((vp / f'{vpath.stem}.csv').is_file() for vpath in cam_videos.values())


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Run Lightning Pose predict per Cheese3D session, calling the Lightning Pose '
            'Python API directly to work around a view-matching bug in lightning-pose when '
            'predicting on all 6 cameras together.'
        )
    )
    parser.add_argument('--root', type=Path, required=True, help='Cheese3D flat video directory')
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='Lightning Pose model directory (Cheese3D-trained checkpoint)',
    )
    parser.add_argument(
        '--session-ids',
        nargs='+',
        default=None,
        metavar='SESSION_ID',
        help='optional subset of session IDs to process',
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=_CHEESE3D_CAMERAS,
        metavar='CAM',
        help=f'camera names to require (default: {_CHEESE3D_CAMERAS})',
    )
    parser.add_argument(
        '--trimmed-videos-dir',
        type=Path,
        default=None,
        help=(
            'directory for frame-count-synced video copies (default: <model-dir>/trimmed_videos); '
            'used only for cameras whose frame count exceeds the session minimum'
        ),
    )
    parser.add_argument(
        '--litpose-repo',
        type=Path,
        default=None,
        help='path to the Lightning Pose source repo; prepended to sys.path (no install required)',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip a session if all prediction CSVs already exist',
    )
    parser.add_argument(
        '--skip-viz',
        action='store_true',
        help='skip generating labeled overlay videos',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print discovered sessions and resolved video paths without predicting',
    )
    return parser


def main() -> None:
    """Entry point for run_litpose_predict_cheese3d."""
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    trimmed_dir = (
        args.trimmed_videos_dir.expanduser().resolve()
        if args.trimmed_videos_dir is not None
        else model_dir / 'trimmed_videos'
    )

    sessions = discover_sessions(root, args.cameras, args.session_ids)

    if args.dry_run:
        for session_id, cam_videos in sessions.items():
            print(f'session={session_id}')
            frame_counts = {cam: get_video_stats(vpath)['total_frames'] for cam, vpath in cam_videos.items()}
            min_count = min(frame_counts.values())
            for cam, vpath in cam_videos.items():
                note = '' if frame_counts[cam] == min_count else f' (would trim to {min_count})'
                print(f'  {cam}: {vpath} [{frame_counts[cam]} frames]{note}')
        return

    sessions = {
        session_id: sync_session_frame_counts(cam_videos, trimmed_dir)
        for session_id, cam_videos in sessions.items()
    }

    if args.litpose_repo is not None:
        litpose_repo = args.litpose_repo.expanduser().resolve()
        if not litpose_repo.is_dir():
            raise FileNotFoundError(f'--litpose-repo directory not found: {litpose_repo}')
        sys.path.insert(0, str(litpose_repo))

    from lightning_pose.api.model import Model
    from lightning_pose.api import model as model_module

    model_module.io_utils.collect_video_files_by_view = _fixed_collect_video_files_by_view

    model = Model.from_dir2(model_dir)

    for session_id, cam_videos in sessions.items():
        if args.skip_existing and _should_skip_session(model_dir, cam_videos):
            print(f'skip (predictions exist): {session_id}')
            continue

        print(f'litpose predict: session_id={session_id}')
        model.predict_on_video_file_multiview(
            list(cam_videos.values()),
            generate_labeled_video=not args.skip_viz,
        )


if __name__ == '__main__':
    main()
