"""SAM3 foreground-mask precomputation for SABLE IBL stereo dataset.

Reads extracted frames from pair_metadata.json and saves per-frame binary
SAM3 masks alongside the images as mask{n:08d}.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from beast.logging import log_step
from beast.preprocess.config_sable import SegmentationConfig
from beast.preprocess.segment.sam3 import load_sam3_image_model, segment_image_with_text_prompt


def _load_session_pairs(
    session_dir: Path,
    cam: str,
    *,
    overwrite: bool,
) -> list[tuple[int, Path]]:
    """Read pair_metadata.json and return (source_frame_index, image_path) pairs.

    Skips frames whose output mask file already exists when ``overwrite`` is False.

    Args:
        session_dir: session directory containing pair_metadata.json.
        cam: camera name (e.g. 'left').
        overwrite: if False, skip frames with existing mask*.png.

    Returns:
        list of (source_frame_index, image_path) sorted by source_frame_index.
    """
    metadata_path = session_dir / 'pair_metadata.json'
    if not metadata_path.is_file():
        return []
    payload = json.loads(metadata_path.read_text(encoding='utf-8'))
    pairs = payload.get('pairs', [])

    path_key = f'{cam}_path'
    idx_key = f'{cam}_source_frame_index'

    seen: dict[int, Path] = {}
    for item in pairs:
        rel = item.get(path_key)
        idx_raw = item.get(idx_key)
        if rel is None or idx_raw is None:
            continue
        frame_idx = int(idx_raw)
        img_path = (session_dir / str(rel)).resolve()
        if not img_path.is_file():
            continue
        if frame_idx in seen:
            continue
        if not overwrite:
            mask_path = session_dir / cam / f'mask{frame_idx:08d}.png'
            if mask_path.exists():
                continue
        seen[frame_idx] = img_path

    return sorted(seen.items())


def run_sam3_precompute(
    dataset_root: Path,
    seg_cfg: SegmentationConfig,
    cameras: list[str],
    *,
    sessionids: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Precompute SAM3 foreground masks for all sessions in a SABLE dataset root.

    Reads pair_metadata.json per session, runs SAM3 text-prompt detection on
    each frame independently, and saves masks co-located with extracted
    images as ``mask{frame_idx:08d}.png``.

    Args:
        dataset_root: path to output_dir/dataset/ produced by extract_sable.
        seg_cfg: segmentation config (SegmentationConfig instance).
        cameras: list of camera names to process.
        sessionids: optional list of session IDs to process; processes all if None.
        overwrite: if True, overwrite existing .png files.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'dataset_root not found: {dataset_root}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    session_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if sessionids:
        sid_set = set(sessionids)
        session_dirs = [p for p in session_dirs if p.name in sid_set]
    if not session_dirs:
        raise RuntimeError(f'no session directories found under {dataset_root}')

    log_step(f'loading SAM3 image model on {device}', level='info')
    model, processor = load_sam3_image_model(device)

    total_saved = 0

    for session_dir in session_dirs:
        for cam in cameras:
            cam_dir = session_dir / cam
            if not cam_dir.is_dir():
                log_step(
                    f'skipping {session_dir.name}/{cam}: directory not found',
                    level='debug',
                )
                continue

            entries = _load_session_pairs(session_dir, cam, overwrite=overwrite)
            if not entries:
                log_step(
                    f'skipping {session_dir.name}/{cam}: no frames to process',
                    level='debug',
                )
                continue

            log_step(
                f'SAM3 segment session={session_dir.name} camera={cam} '
                f'frames={len(entries)}',
                level='info',
            )

            for frame_idx, img_path in entries:
                with Image.open(img_path) as img:
                    frame = np.asarray(img.convert('RGB'), dtype=np.uint8)
                mask = segment_image_with_text_prompt(
                    frame,
                    model=model,
                    processor=processor,
                    device=device,
                    text_prompt=seg_cfg.text_prompt,
                    num_object=seg_cfg.num_objects,
                    threshold=seg_cfg.threshold,
                )
                out_path = cam_dir / f'mask{frame_idx:08d}.png'
                cv2.imwrite(str(out_path), mask)
                total_saved += 1

            log_step(
                f'saved session={session_dir.name} camera={cam} count={len(entries)}',
                level='info',
            )

    log_step(f'SAM3 precompute complete: total_saved={total_saved}', level='info')
