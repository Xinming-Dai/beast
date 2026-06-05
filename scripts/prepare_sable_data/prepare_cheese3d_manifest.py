"""Create a frame-level Cheese3D manifest for Sable preparation work."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


IMAGE_RE = re.compile(r'^img(?P<idx>\d+)\.png$')
CAMERA_RE = re.compile(r'^img(?P<idx>\d+)\.npy$')
MASK_RE = re.compile(r'^mask(?P<idx>\d+)\.png$')


def parse_index(path: Path, pattern: re.Pattern[str]) -> int | None:
    """Parse a numeric frame index from a file name.

    Args:
        path: file path whose name should match ``pattern``.
        pattern: compiled regex with an ``idx`` capture group.

    Returns:
        integer frame index, or ``None`` when the file name does not match.
    """
    match = pattern.match(path.name)
    if match is None:
        return None
    return int(match.group('idx'))


def collect_indexed_files(directory: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    """Collect files in ``directory`` keyed by parsed frame index."""
    if not directory.exists():
        raise FileNotFoundError(f'Missing directory: {directory}')

    indexed: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        frame_idx = parse_index(path, pattern)
        if frame_idx is not None:
            indexed[frame_idx] = path.resolve()
    return indexed


def load_info(dataset_dir: Path) -> dict[str, Any]:
    """Load the Cheese3D ``info.json`` metadata file."""
    info_path = dataset_dir / 'info.json'
    if not info_path.exists():
        raise FileNotFoundError(f'Missing Cheese3D info file: {info_path}')
    with open(info_path) as f:
        return json.load(f)


def resolve_sessions(dataset_dir: Path, info: dict[str, Any], sessions: list[str] | None) -> list[str]:
    """Resolve session ids to include in the manifest."""
    if sessions:
        return sessions

    info_sessions = info.get('video_ids')
    if isinstance(info_sessions, list) and info_sessions:
        return [str(session_id) for session_id in info_sessions]

    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def resolve_views(info: dict[str, Any], views: list[str] | None) -> list[str]:
    """Resolve camera view names to include in each manifest row."""
    if views:
        return views

    info_views = info.get('available_views')
    if not isinstance(info_views, list) or not info_views:
        raise ValueError('Cheese3D info.json does not define available_views; pass --views.')
    return [str(view) for view in info_views]


def resolve_mask_dir(root: Path, session_id: str, view: str) -> Path | None:
    """Resolve the segmentation mask directory for one session/view pair."""
    mask_root = root / 'segmentation_masks'
    if not mask_root.exists():
        return None

    matches = sorted(path for path in mask_root.glob(f'{session_id}_{view}_*') if path.is_dir())
    if not matches:
        return None
    if len(matches) > 1:
        match_list = ', '.join(str(path) for path in matches)
        raise ValueError(
            f'Found multiple mask directories for session={session_id}, view={view}: {match_list}'
        )
    return matches[0]


def frame_indices_for_view(
    *,
    session_dir: Path,
    root: Path,
    session_id: str,
    view: str,
    require_masks: bool,
) -> tuple[set[int], dict[int, Path], dict[int, Path], dict[int, Path]]:
    """Collect valid frame indices and per-frame files for a session/view."""
    view_dir = session_dir / view
    images = collect_indexed_files(view_dir, IMAGE_RE)
    cameras = collect_indexed_files(view_dir, CAMERA_RE)

    masks: dict[int, Path] = {}
    mask_dir = resolve_mask_dir(root, session_id, view)
    if mask_dir is not None:
        masks = collect_indexed_files(mask_dir / 'masks', MASK_RE)
    elif require_masks:
        raise FileNotFoundError(
            f'Missing mask directory for session={session_id}, view={view}'
        )

    valid = set(images) & set(cameras)
    if require_masks:
        valid &= set(masks)
    return valid, images, cameras, masks


def select_frame_indices(
    frame_indices: set[int],
    *,
    start_frame: int,
    frame_step: int,
    max_frames: int | None,
) -> list[int]:
    """Apply deterministic frame filtering to a set of frame indices."""
    if frame_step <= 0:
        raise ValueError(f'frame_step must be positive, got {frame_step}')
    if start_frame < 0:
        raise ValueError(f'start_frame must be non-negative, got {start_frame}')

    selected = [
        frame_idx
        for frame_idx in sorted(frame_indices)
        if frame_idx >= start_frame and (frame_idx - start_frame) % frame_step == 0
    ]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def build_manifest_records(
    *,
    root: Path,
    views: list[str] | None = None,
    sessions: list[str] | None = None,
    start_frame: int = 0,
    frame_step: int = 1,
    max_frames_per_session: int | None = None,
    require_masks: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Cheese3D manifest records and a compact summary.

    Args:
        root: Cheese3D root containing ``cheese3d_cam/`` and ``segmentation_masks/``.
        views: view names to include. Defaults to ``info.json`` ``available_views``.
        sessions: session ids to include. Defaults to ``info.json`` ``video_ids``.
        start_frame: first allowed frame index.
        frame_step: deterministic sampling stride after ``start_frame``.
        max_frames_per_session: optional cap per session.
        require_masks: if true, each view/frame must have a segmentation mask.

    Returns:
        tuple of ``(records, summary)``.
    """
    root = root.resolve()
    dataset_dir = root / 'cheese3d_cam'
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Missing Cheese3D frame directory: {dataset_dir}')

    info = load_info(dataset_dir)
    session_ids = resolve_sessions(dataset_dir, info, sessions)
    view_names = resolve_views(info, views)

    records: list[dict[str, Any]] = []
    per_session: dict[str, int] = {}
    skipped_sessions: dict[str, str] = {}

    for session_id in session_ids:
        session_dir = dataset_dir / session_id
        if not session_dir.exists():
            skipped_sessions[session_id] = f'missing session directory: {session_dir}'
            continue

        files_by_view: dict[str, dict[str, dict[int, Path]]] = {}
        common_indices: set[int] | None = None
        try:
            for view in view_names:
                valid, images, cameras, masks = frame_indices_for_view(
                    session_dir=session_dir,
                    root=root,
                    session_id=session_id,
                    view=view,
                    require_masks=require_masks,
                )
                files_by_view[view] = {
                    'images': images,
                    'cameras': cameras,
                    'masks': masks,
                }
                common_indices = valid if common_indices is None else common_indices & valid
        except (FileNotFoundError, ValueError) as exc:
            skipped_sessions[session_id] = str(exc)
            continue

        selected_indices = select_frame_indices(
            common_indices or set(),
            start_frame=start_frame,
            frame_step=frame_step,
            max_frames=max_frames_per_session,
        )
        per_session[session_id] = len(selected_indices)

        for frame_idx in selected_indices:
            view_entries = {}
            for view in view_names:
                files = files_by_view[view]
                entry = {
                    'image_path': str(files['images'][frame_idx]),
                    'camera_path': str(files['cameras'][frame_idx]),
                }
                if frame_idx in files['masks']:
                    entry['mask_path'] = str(files['masks'][frame_idx])
                view_entries[view] = entry

            records.append({
                'dataset': 'cheese3d',
                'session_id': session_id,
                'frame_idx': frame_idx,
                'frame_name': f'img{frame_idx:08d}',
                'view_order': view_names,
                'views': view_entries,
            })

    summary = {
        'root': str(root),
        'views': view_names,
        'sessions_requested': session_ids,
        'num_sessions_requested': len(session_ids),
        'num_sessions_with_records': sum(1 for count in per_session.values() if count > 0),
        'num_records': len(records),
        'records_per_session': per_session,
        'skipped_sessions': skipped_sessions,
        'start_frame': start_frame,
        'frame_step': frame_step,
        'max_frames_per_session': max_frames_per_session,
        'require_masks': require_masks,
    }
    return records, summary


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write manifest records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + '\n')


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON object with stable indentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write('\n')


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Create a Cheese3D frame manifest for Sable preparation.',
    )
    parser.add_argument(
        '--cheese3d-root',
        type=Path,
        required=True,
        help='Root containing cheese3d_cam/, segmentation_masks/, config.yaml.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL manifest path.',
    )
    parser.add_argument(
        '--summary-output',
        type=Path,
        default=None,
        help='Output summary JSON path. Defaults to <output>.summary.json.',
    )
    parser.add_argument(
        '--views',
        nargs='+',
        default=None,
        help='Camera views to include. Defaults to info.json available_views.',
    )
    parser.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help='Session ids to include. Defaults to info.json video_ids.',
    )
    parser.add_argument(
        '--start-frame',
        type=int,
        default=0,
        help='First frame index to include.',
    )
    parser.add_argument(
        '--frame-step',
        type=int,
        default=1,
        help='Frame stride after start-frame.',
    )
    parser.add_argument(
        '--max-frames-per-session',
        type=int,
        default=None,
        help='Optional cap per session after applying start-frame and frame-step.',
    )
    parser.add_argument(
        '--allow-missing-masks',
        action='store_true',
        help='Do not require mask files in manifest rows.',
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    records, summary = build_manifest_records(
        root=args.cheese3d_root,
        views=args.views,
        sessions=args.sessions,
        start_frame=args.start_frame,
        frame_step=args.frame_step,
        max_frames_per_session=args.max_frames_per_session,
        require_masks=not args.allow_missing_masks,
    )
    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_suffix(args.output.suffix + '.summary.json')

    write_jsonl(records, args.output)
    write_json(summary, summary_output)
    print(f'Wrote {len(records)} Cheese3D manifest rows to {args.output}')
    print(f'Wrote summary to {summary_output}')


if __name__ == '__main__':
    main()
