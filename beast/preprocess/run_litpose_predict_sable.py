"""Run Lightning Pose predict on SABLE IBL-2view sessions.

Resolves camera video paths from the IBL-2view directory layout and calls
``litpose predict`` for each session. Camera names are read from the config file.
Run this script in the ``lp`` conda environment before running ``beast extract_sable``
with ``litpose.enabled: true``.

Usage::

    conda activate lp
    python beast/preprocess/run_litpose_predict_sable.py \\
      --root /work/hdd/bfsr/xdai3/IBL-2view \\
      --model-dir /path/to/lightning_pose_model \\
      --config configs/multiview/extraction_pipeline_sable.yaml \\
      [--sessionids <eid1> <eid2>] \\
      [--skip-existing] [--dry-run] [-- --skip_viz]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from beast.preprocess.extraction_sable import _video_session_re
from beast.preprocess.config_sable import load_sable_config


def _discover_sessions(
    root: Path,
    cameras: list[str],
    video_naming,
    sessionids: list[str] | None = None,
) -> list[str]:
    """Find session IDs present in all camera video directories.

    Args:
        root: IBL-2view root directory.
        cameras: camera names to require.
        video_naming: VideoNamingConfig object.
        sessionids: optional explicit list of session IDs to filter to.

    Returns:
        sorted list of session ID strings.

    Raises:
        FileNotFoundError: if any camera video directory is missing.
    """
    per_cam_ids: list[set[str]] = []
    for cam in cameras:
        cam_dir = root / video_naming.camera_video_subdir.format(cam=cam)
        if not cam_dir.is_dir():
            raise FileNotFoundError(f'camera video directory not found: {cam_dir}')
        session_re = _video_session_re(cam, video_naming=video_naming)
        per_cam_ids.append({
            m.group(1)
            for f in cam_dir.iterdir()
            if (m := session_re.match(f.name))
        })

    ids = sorted(set.intersection(*per_cam_ids)) if per_cam_ids else []

    if sessionids:
        requested = set(sessionids)
        ids = [sid for sid in ids if sid in requested]

    return ids


def _video_paths(
    root: Path,
    cameras: list[str],
    session_id: str,
    video_naming,
) -> dict[str, Path]:
    """Return video paths for all cameras for one session.

    Args:
        root: IBL-2view root directory.
        cameras: camera names.
        session_id: session identifier.
        video_naming: VideoNamingConfig object.

    Returns:
        dict mapping camera name to mp4 path.
    """
    return {
        cam: root / video_naming.camera_video_subdir.format(cam=cam)
        / video_naming.video_filename.format(cam=cam, session_id=session_id)
        for cam in cameras
    }


def _should_skip(
    model_dir: Path,
    cam_videos: dict[str, Path],
) -> bool:
    """Return True if all prediction CSVs already exist.

    Args:
        model_dir: Lightning Pose model directory.
        cam_videos: dict of camera → mp4 path.

    Returns:
        True if all CSVs are present.
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
            'Run litpose predict per SABLE IBL-2view session. '
            'Resolves mp4 paths from the IBL-2view directory layout. '
            'Run in the lp conda environment.'
        )
    )
    parser.add_argument(
        '--root',
        type=Path,
        required=True,
        help='IBL-2view root directory',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='Lightning Pose model directory (passed to litpose predict)',
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='path to extraction_pipeline_sable.yaml; cameras and sessionids are read from it',
    )
    parser.add_argument(
        '--sessionids',
        nargs='+',
        default=None,
        metavar='SESSION_ID',
        help='optional list of session IDs; overrides config sessionids if provided',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='print litpose commands without running',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip a session if all prediction CSVs already exist',
    )
    parser.add_argument(
        '--litpose-bin',
        default='litpose',
        help='litpose executable name or path (default: litpose)',
    )
    parser.add_argument(
        'litpose_argv',
        nargs=argparse.REMAINDER,
        help='extra args forwarded to litpose predict (use: -- --skip_viz ...)',
    )
    return parser


def main() -> None:
    """Entry point for run_litpose_predict_sable."""
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    litpose_bin: str = args.litpose_bin

    # load config to get cameras / sessionids
    cfg = load_sable_config(args.config)
    cameras = list(cfg.cameras)
    config_sessionids = list(cfg.sessionids) if cfg.sessionids else None

    # CLI sessionids override config
    sessionids: list[str] | None = args.sessionids if args.sessionids is not None else config_sessionids

    if not args.dry_run:
        on_path = shutil.which(litpose_bin) is not None
        lp_path = Path(litpose_bin)
        executable_file = lp_path.is_file() and os.access(lp_path, os.X_OK)
        if not on_path and not executable_file:
            raise FileNotFoundError(
                f'litpose not found ({litpose_bin!r}). '
                'Install Lightning Pose and ensure it is on PATH, '
                'or pass --litpose-bin /path/to/litpose.'
            )

    session_ids = _discover_sessions(root, cameras, cfg.video_naming, sessionids)
    if not session_ids:
        print('no session IDs found; nothing to do.')
        return

    extra = list(args.litpose_argv)
    if extra and extra[0] == '--':
        extra = extra[1:]

    for session_id in session_ids:
        cam_videos = _video_paths(root, cameras, session_id, cfg.video_naming)
        for cam, vpath in cam_videos.items():
            if not vpath.is_file():
                raise FileNotFoundError(
                    f'video missing for sessionid={session_id} camera={cam}: {vpath}'
                )

        if args.skip_existing and _should_skip(model_dir, cam_videos):
            print(f'skip (predictions exist): {session_id}')
            continue

        cmd = [
            litpose_bin,
            'predict',
            str(model_dir),
            *[str(p) for p in cam_videos.values()],
            *extra,
        ]
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue

        print(f'litpose predict: sessionid={session_id}')
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
