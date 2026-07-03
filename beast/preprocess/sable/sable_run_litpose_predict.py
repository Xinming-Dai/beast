"""Run Lightning Pose predict on SABLE IBL-2view sessions.

Resolves camera video paths from the IBL-2view directory layout and calls
``litpose predict`` for each session. ``--config`` is optional; when omitted,
the standard IBL-2view camera names and naming conventions are used as defaults.

Run this script in an environment where Lightning Pose is available, either
by passing ``--litpose-repo`` (Lightning Pose source repo; no install required)
or by activating the ``lp`` conda environment (``--litpose-bin litpose``).

Usage::

    # pass session IDs directly (no config needed)
    python beast/preprocess/sable/run_litpose_predict_sable.py \\
      --root /work/hdd/bfsr/xdai3/IBL-2view \\
      --model-dir /path/to/lightning_pose_model \\
      --litpose-repo /u/xdai3/project3d/lightning-pose \\
      --session-ids <eid1> <eid2> \\
      [--skip-existing] [--dry-run] [-- --skip_viz]

    # use config for cameras / naming conventions
    python beast/preprocess/sable/run_litpose_predict_sable.py \\
      --root /work/hdd/bfsr/xdai3/IBL-2view \\
      --model-dir /path/to/lightning_pose_model \\
      --config configs/multiview/extraction_pipeline_sable.yaml \\
      --litpose-repo /u/xdai3/project3d/lightning-pose \\
      [-- --skip_viz]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

_IBL_CAMERAS = ['left', 'right']
_IBL_VIDEO_SUBDIR = '{cam}Camera.video'
_IBL_VIDEO_FILENAME = '_iblrig_{cam}Camera.downsampled.{session_id}.mp4'


def _ibl_video_path(root: Path, cam: str, session_id: str) -> Path:
    """Return the default IBL-2view mp4 path for one camera and session.

    Args:
        root: IBL-2view root directory.
        cam: camera name (e.g. 'left').
        session_id: session identifier.

    Returns:
        path to the mp4 file.
    """
    return (
        root
        / _IBL_VIDEO_SUBDIR.format(cam=cam)
        / _IBL_VIDEO_FILENAME.format(cam=cam, session_id=session_id)
    )


def _discover_sessions_ibl(
    root: Path,
    cameras: list[str],
    session_ids: list[str] | None,
) -> list[str]:
    """Discover sessions from IBL-2view camera video directories.

    Args:
        root: IBL-2view root directory.
        cameras: camera names to require.
        session_ids: optional explicit list to filter to.

    Returns:
        sorted list of session ID strings present in all camera dirs.

    Raises:
        FileNotFoundError: if any camera video directory is missing.
        RuntimeError: if no matching sessions are found.
    """
    import re

    per_cam: list[set[str]] = []
    for cam in cameras:
        cam_dir = root / _IBL_VIDEO_SUBDIR.format(cam=cam)
        if not cam_dir.is_dir():
            raise FileNotFoundError(f'camera video directory not found: {cam_dir}')
        sentinel = '__SID__'
        escaped = re.escape(_IBL_VIDEO_FILENAME.format(cam=cam, session_id=sentinel))
        pattern = re.compile('^' + escaped.replace(re.escape(sentinel), r'(.+)') + '$')
        per_cam.append({m.group(1) for f in cam_dir.iterdir() if (m := pattern.match(f.name))})

    ids = sorted(set.intersection(*per_cam)) if per_cam else []

    if session_ids:
        requested = set(session_ids)
        missing = sorted(requested - set(ids))
        if missing:
            print(f'warning: session IDs not found in camera dirs: {missing}', file=sys.stderr)
        ids = [s for s in ids if s in requested]

    if not ids:
        raise RuntimeError(f'no matching session IDs found under {root} (cameras={cameras})')
    return ids


def _should_skip_session(
    model_dir: Path,
    cam_videos: dict[str, Path],
    output_dir: Path | None,
) -> bool:
    """Return True if all prediction CSVs already exist.

    Args:
        model_dir: Lightning Pose model directory.
        cam_videos: mapping of camera name to mp4 path.
        output_dir: optional output directory; if given, checks there instead of
            ``<model_dir>/video_preds/``.

    Returns:
        True if all per-camera prediction CSVs are present.
    """
    check_dir = output_dir if output_dir is not None else model_dir / 'video_preds'
    return all((check_dir / f'{vpath.stem}.csv').is_file() for vpath in cam_videos.values())


def _copy_session_video_preds(
    model_dir: Path,
    output_dir: Path,
    cam_videos: dict[str, Path],
) -> None:
    """Copy prediction files for one session from ``video_preds/`` into ``output_dir``.

    Copies CSVs and metric sidecars matching each camera stem. Labeled overlay
    videos under ``labeled_videos/`` are copied if present.

    Args:
        model_dir: Lightning Pose model directory.
        output_dir: destination directory.
        cam_videos: mapping of camera name to mp4 path.
    """
    vp = model_dir / 'video_preds'
    if not vp.is_dir():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_src = vp / 'labeled_videos'
    for vpath in cam_videos.values():
        for path in sorted(vp.glob(f'{vpath.stem}*')):
            if path.is_file():
                shutil.copy2(path, output_dir / path.name)
        if labeled_src.is_dir():
            dest_labeled = output_dir / 'labeled_videos'
            for path in sorted(labeled_src.glob(f'{vpath.stem}*')):
                if path.is_file():
                    dest_labeled.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest_labeled / path.name)


def _build_cmd(
    model_dir: Path,
    cam_videos: dict[str, Path],
    extra: list[str],
    *,
    litpose_bin: str,
    litpose_repo: Path | None,
) -> tuple[list[str], dict[str, str] | None]:
    """Build the subprocess command and optional environment override.

    When ``litpose_repo`` is given, returns a command that calls
    ``python -m lightning_pose.cli.main predict`` with the repo prepended to
    ``PYTHONPATH``. Otherwise, uses the ``litpose_bin`` binary.

    Args:
        model_dir: Lightning Pose model directory.
        cam_videos: mapping of camera name to mp4 path.
        extra: extra args forwarded to ``litpose predict``.
        litpose_bin: litpose binary name or path (ignored when litpose_repo is set).
        litpose_repo: path to the Lightning Pose source repo, or None.

    Returns:
        (cmd, env) tuple. ``env`` is None when no environment override is needed.
    """
    video_args = [str(v) for v in cam_videos.values()]
    if litpose_repo is not None:
        env = os.environ.copy()
        existing_pp = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f'{litpose_repo}:{existing_pp}' if existing_pp else str(litpose_repo)
        cmd = [
            sys.executable,
            '-m', 'lightning_pose.cli.main',
            'predict',
            str(model_dir),
            *video_args,
            *extra,
        ]
        return cmd, env
    cmd = [litpose_bin, 'predict', str(model_dir), *video_args, *extra]
    return cmd, None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Run litpose predict per SABLE IBL-2view session. '
            'Resolves mp4 paths from the IBL-2view directory layout and cameras '
            'from the extraction config. Pass --litpose-repo to call Lightning Pose '
            'directly from source without activating the lp environment.'
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
        default=None,
        help=(
            'optional path to extraction_pipeline_sable.yaml; cameras and video naming are read '
            'from it. When omitted, defaults to cameras=[left, right] and standard IBL-2view '
            'naming conventions. --session-ids is required when --config is not given.'
        ),
    )
    parser.add_argument(
        '--session-ids',
        '--only-eids',
        nargs='+',
        default=None,
        dest='session_ids',
        metavar='SESSION_ID',
        help='optional subset of session IDs to process (overrides config sessionids)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=(
            'optional: after each successful litpose run, copy this session\'s prediction '
            'CSVs from <model-dir>/video_preds/ here; --skip-existing checks this directory '
            'instead of video_preds/'
        ),
    )
    litpose_group = parser.add_mutually_exclusive_group()
    litpose_group.add_argument(
        '--litpose-repo',
        type=Path,
        default=None,
        help=(
            'path to the Lightning Pose source repo; calls '
            '``python -m lightning_pose.cli.main predict`` with PYTHONPATH set accordingly '
            '(mutually exclusive with --litpose-bin)'
        ),
    )
    litpose_group.add_argument(
        '--litpose-bin',
        default='litpose',
        help='litpose executable name or path (default: litpose; mutually exclusive with --litpose-repo)',
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
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else None
    litpose_repo = args.litpose_repo.expanduser().resolve() if args.litpose_repo is not None else None

    if args.config is not None:
        from beast.preprocess.config_sable import load_sable_config
        from beast.preprocess.extraction_sable import _video_path, discover_sessions
        cfg = load_sable_config(args.config)
        cameras = list(cfg.cameras)
        video_naming = cfg.video_naming
        config_sessionids = list(cfg.sessionids) if cfg.sessionids else None
        sessionids: list[str] | None = (
            args.session_ids if args.session_ids is not None else config_sessionids
        )
        session_ids = discover_sessions(root, cameras, video_naming, sessionids)
        resolve_path = lambda cam, sid: _video_path(root, cam, sid, video_naming=video_naming)
    else:
        if not args.session_ids:
            build_parser().error('--session-ids is required when --config is not given')
        cameras = _IBL_CAMERAS
        session_ids = _discover_sessions_ibl(root, cameras, args.session_ids)
        resolve_path = lambda cam, sid: _ibl_video_path(root, cam, sid)

    if not args.dry_run:
        if litpose_repo is not None:
            if not litpose_repo.is_dir():
                raise FileNotFoundError(f'--litpose-repo directory not found: {litpose_repo}')
        else:
            litpose_bin = args.litpose_bin
            lp_path = Path(litpose_bin)
            on_path = shutil.which(litpose_bin) is not None
            executable_file = lp_path.is_file() and os.access(lp_path, os.X_OK)
            if not on_path and not executable_file:
                raise FileNotFoundError(
                    f'litpose not found ({litpose_bin!r}). '
                    'Install Lightning Pose and ensure it is on PATH, '
                    'pass --litpose-bin /path/to/litpose, '
                    'or pass --litpose-repo /path/to/lightning-pose-source.'
                )

    extra = list(args.litpose_argv)
    if extra and extra[0] == '--':
        extra = extra[1:]

    for session_id in session_ids:
        cam_videos = {cam: resolve_path(cam, session_id) for cam in cameras}
        for cam, vpath in cam_videos.items():
            if not vpath.is_file():
                raise FileNotFoundError(
                    f'video missing for session_id={session_id} camera={cam}: {vpath}'
                )

        if args.skip_existing and _should_skip_session(model_dir, cam_videos, output_dir):
            print(f'skip (predictions exist): {session_id}')
            continue

        cmd, env = _build_cmd(
            model_dir,
            cam_videos,
            extra,
            litpose_bin=args.litpose_bin,
            litpose_repo=litpose_repo,
        )
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            if output_dir is not None:
                print(f'  (then copy session outputs to {output_dir})')
            continue

        print(f'litpose predict: session_id={session_id}')
        subprocess.run(cmd, check=True, env=env)
        if output_dir is not None:
            _copy_session_video_preds(model_dir, output_dir, cam_videos)
            print(f'copied predictions for session_id={session_id} -> {output_dir}')


if __name__ == '__main__':
    main()
