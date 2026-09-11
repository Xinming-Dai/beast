#!/usr/bin/env python3
"""Precompute SAM3 foreground masks for the IBL neural-encoding eval frame tree.

Segments frames independently (no video tracking — frames are non-contiguous,
selected per neural trial/timebin) under
``{frames_dir}/{cam}Camera.video/_iblrig_{cam}Camera.downsampled.{session_id}/
{train,val,test}/*.png``, using ``frame_index_mapping.json`` in each split
directory to recover the original source frame index per PNG.

Masks are written to
``{output_root}/segmentation_masks/{session_id}/{cam}/mask{idx:08d}.png``,
mirroring the sibling ``depth_map/{session_id}/{cam}/depth{idx:08d}.npy`` tree.

Pass ``--config configs/multiview/extraction_pipeline_sable.yaml`` to supply
SAM3 parameters (text_prompt/num_objects/threshold) from the config's
``segmentation`` section without repeating them on the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from beast.preprocess.segment.sam3 import load_sam3_image_model, segment_image_with_text_prompt

logger = logging.getLogger(__name__)

_UUID_RE = re.compile(
    r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)

_SPLIT_NAMES = ('train', 'val', 'test')


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: path to the YAML file.

    Returns:
        parsed config as a dict.

    Raises:
        ValueError: if the file does not contain a YAML mapping.
    """
    with path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'expected a YAML mapping in {path}')
    return cfg


def _discover_eval_frames(
    frames_dir: Path,
    cam: str,
    cam_subdir_tmpl: str,
    eids: set[str] | None,
    splits: tuple[str, ...] = _SPLIT_NAMES,
) -> dict[str, dict[int, Path]]:
    """Discover session frames by scanning the eval frame tree's split directories.

    Expects ``{frames_dir}/{cam_subdir}/`` to contain one subdirectory per
    session (name contains a UUID), each holding ``train/``, ``val/``,
    ``test/`` subdirectories with a ``frame_index_mapping.json`` mapping PNG
    filenames to ``{cam}_source_frame_index``.

    Args:
        frames_dir: root of the eval frames directory.
        cam: camera name (e.g. 'left').
        cam_subdir_tmpl: template for the camera subdirectory.
        eids: if not None, only return sessions whose UUID is in this set.
        splits: which split subdirectories to scan; defaults to all of
            train/val/test.

    Returns:
        mapping from session_id to {source_frame_index: image_path}, deduped
        across splits (first occurrence wins).

    Raises:
        FileNotFoundError: if the camera subdirectory does not exist.
    """
    cam_root = frames_dir / cam_subdir_tmpl.format(cam=cam)
    if not cam_root.is_dir():
        raise FileNotFoundError(f'camera directory not found: {cam_root}')

    idx_key = f'{cam}_source_frame_index'
    grouped: dict[str, dict[int, Path]] = {}

    for session_dir in sorted(cam_root.iterdir()):
        if not session_dir.is_dir():
            continue
        m = _UUID_RE.search(session_dir.name)
        if not m:
            continue
        session_id = m.group(1)
        if eids is not None and session_id not in eids:
            continue

        frames: dict[int, Path] = {}
        for split in splits:
            split_dir = session_dir / split
            mapping_path = split_dir / 'frame_index_mapping.json'
            if not mapping_path.is_file():
                continue
            mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
            for filename, meta in mapping.items():
                frame_idx = meta.get(idx_key)
                if frame_idx is None:
                    continue
                frame_idx = int(frame_idx)
                if frame_idx in frames:
                    continue
                img_path = split_dir / filename
                if img_path.is_file():
                    frames[frame_idx] = img_path

        if frames:
            grouped[session_id] = frames
            logger.info(f'discovered session={session_id} camera={cam} frames={len(frames)}')

    return grouped


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        metavar='YAML',
        help='path to extraction_pipeline_sable.yaml; supplies segmentation defaults',
    )
    p.add_argument(
        '--frames-dir',
        type=Path,
        required=True,
        help='root of the extracted eval frame tree (contains {cam}Camera.video/)',
    )
    p.add_argument(
        '--output-root',
        type=Path,
        required=True,
        help='root to write segmentation_masks/{session}/{cam}/mask*.png under',
    )
    p.add_argument('--cameras', nargs='+', default=['left', 'right'])
    p.add_argument(
        '--eids',
        nargs='+',
        metavar='EID',
        default=None,
        help='only process these session UUIDs (overrides sessionids from --config)',
    )
    p.add_argument(
        '--overwrite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='overwrite existing masks (default: --no-overwrite)',
    )
    p.add_argument(
        '--split',
        nargs='+',
        choices=_SPLIT_NAMES,
        default=None,
        metavar='SPLIT',
        help='only process these split subdirectories (e.g. --split test); '
             'default processes train, val, and test',
    )
    return p


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s  %(name)s : %(message)s',
    )

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml_config(args.config.expanduser().resolve())

    seg_cfg: dict[str, Any] = cfg.get('segmentation') or {}
    video_naming_cfg: dict[str, Any] = cfg.get('video_naming') or {}

    text_prompt = str(seg_cfg.get('text_prompt', 'mouse'))
    num_objects_raw = seg_cfg.get('num_objects', 1)
    num_objects = int(num_objects_raw) if num_objects_raw is not None else None
    threshold = float(seg_cfg.get('threshold', 0.5))
    cam_subdir_tmpl = str(video_naming_cfg.get('camera_video_subdir', '{cam}Camera.video'))

    if args.eids is not None:
        eids: set[str] | None = set(args.eids)
    else:
        cfg_eids = cfg.get('sessionids')
        eids = {str(e) for e in cfg_eids} if cfg_eids else None

    frames_dir = args.frames_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    splits = tuple(args.split) if args.split else _SPLIT_NAMES

    logger.info(f'frames_dir={frames_dir}')
    logger.info(f'output_root={output_root}')
    logger.info(
        f'text_prompt={text_prompt!r} num_objects={num_objects} threshold={threshold} '
        f'eids={sorted(eids) if eids else "all"} splits={splits}',
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'loading SAM3 image model on {device}')
    model, processor = load_sam3_image_model(device)

    summary: dict[str, Any] = {}
    total_saved = 0

    for cam in args.cameras:
        grouped = _discover_eval_frames(frames_dir, cam, cam_subdir_tmpl, eids, splits=splits)
        for session_id, frames in grouped.items():
            out_dir = output_root / 'segmentation_masks' / session_id / cam
            out_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            skipped = 0
            total = len(frames)
            milestones = {max(1, total * pct // 10) for pct in range(1, 11)}
            for processed, (frame_idx, img_path) in enumerate(sorted(frames.items()), start=1):
                out_path = out_dir / f'mask{frame_idx:08d}.png'
                if out_path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    with Image.open(img_path) as img:
                        frame = np.asarray(img.convert('RGB'), dtype=np.uint8)
                    mask = segment_image_with_text_prompt(
                        frame,
                        model=model,
                        processor=processor,
                        device=device,
                        text_prompt=text_prompt,
                        num_object=num_objects,
                        threshold=threshold,
                    )
                    cv2.imwrite(str(out_path), mask)
                    saved += 1

                if processed in milestones:
                    pct = processed * 100 // total
                    logger.info(
                        f'progress session={session_id} camera={cam} '
                        f'{processed}/{total} ({pct}%) saved={saved} skipped={skipped}',
                    )

            total_saved += saved
            summary.setdefault(session_id, {})[cam] = {
                'saved': saved,
                'skipped_existing': skipped,
                'total_frames': len(frames),
            }
            logger.info(
                f'done session={session_id} camera={cam} '
                f'saved={saved} skipped={skipped}',
            )

    summary_dir = output_root / 'segmentation_masks'
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / 'sam3_precompute_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(
        json.dumps({
            'total_saved': total_saved,
            'summary': str(summary_path),
            'output_root': str(output_root),
        }),
    )


if __name__ == '__main__':
    main()
