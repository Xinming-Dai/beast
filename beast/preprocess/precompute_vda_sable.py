"""VDA depth precomputation for SABLE IBL stereo dataset.

Reads extracted frames from pair_metadata.json and saves per-frame VDA depth
maps alongside the images as vda{n:08d}.npy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from beast.logging import log_step
from beast.models.model_utils.utils_vda import load_frozen_video_depth_anything
from beast.preprocess.config_sable import VDAConfig
from beast.preprocess.extraction_sable import _CAMERA_VIDEO_SUBDIR_TMPL


def _resolve_device(device: str) -> str:
    """Resolve device string, replacing 'auto' with cuda or cpu.

    Args:
        device: requested device string.

    Returns:
        resolved device string.
    """
    requested = str(device).strip().lower()
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return requested


def _load_session_pairs(
    session_dir: Path,
    cam: str,
    *,
    overwrite: bool,
) -> list[tuple[int, Path]]:
    """Read pair_metadata.json and return (source_frame_index, image_path) pairs.

    Skips frames whose output depth file already exists when ``overwrite`` is False.

    Args:
        session_dir: session directory containing pair_metadata.json.
        cam: camera name (e.g. 'left').
        overwrite: if False, skip frames with existing vda*.npy.

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
            depth_path = session_dir / cam / f'vda{frame_idx:08d}.npy'
            if depth_path.exists():
                continue
        seen[frame_idx] = img_path

    return sorted(seen.items())


def _load_rgb_frames(paths: list[Path]) -> np.ndarray:
    """Load a list of image paths as a uint8 NHWC array.

    Args:
        paths: image file paths.

    Returns:
        uint8 array of shape [N, H, W, 3].
    """
    frames = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(np.asarray(img.convert('RGB'), dtype=np.uint8))
    return np.stack(frames, axis=0)


def run_vda_precompute(
    dataset_root: Path,
    vda_cfg: VDAConfig,
    cameras: list[str],
    *,
    sessionids: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Precompute VDA depth maps for all sessions in a SABLE dataset root.

    Reads pair_metadata.json per session, runs VDA inference per camera, and
    saves depth maps co-located with extracted images as ``vda{frame_idx:08d}.npy``.

    Args:
        dataset_root: path to output_dir/dataset/ produced by extract_sable.
        vda_cfg: VDA config (VDAConfig instance).
        cameras: list of camera names to process.
        sessionids: optional list of session IDs to process; processes all if None.
        overwrite: if True, overwrite existing .npy files.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'dataset_root not found: {dataset_root}')

    device = _resolve_device(str(vda_cfg.device))
    input_size = int(vda_cfg.input_size)
    fp32 = bool(vda_cfg.fp32)

    session_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if sessionids:
        sid_set = set(sessionids)
        session_dirs = [p for p in session_dirs if p.name in sid_set]
    if not session_dirs:
        raise RuntimeError(f'no session directories found under {dataset_root}')

    log_step(
        f'loading VDA model (encoder={vda_cfg.encoder}) on {device}',
        level='info',
    )
    vda_cfg_dict = {
        'encoder': vda_cfg.encoder,
        'checkpoint_path': vda_cfg.checkpoint_path,
        'metric': False,
    }
    model = load_frozen_video_depth_anything(vda_cfg_dict).to(device)

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

            frame_indices = [idx for idx, _ in entries]
            frame_paths = [path for _, path in entries]

            log_step(
                f'VDA infer session={session_dir.name} camera={cam} '
                f'frames={len(frame_paths)}',
                level='info',
            )

            frames_np = _load_rgb_frames(frame_paths)

            with torch.no_grad():
                depth_maps, _ = model.infer_video_depth(
                    frames_np,
                    target_fps=1.0,
                    input_size=input_size,
                    device=device,
                    fp32=fp32,
                )

            if depth_maps.shape[0] != len(frame_indices):
                raise RuntimeError(
                    f'VDA returned {depth_maps.shape[0]} depth maps for '
                    f'{len(frame_indices)} frames '
                    f'(session={session_dir.name}, camera={cam})'
                )

            for frame_idx, depth_map in zip(frame_indices, depth_maps):
                out_path = cam_dir / f'vda{frame_idx:08d}.npy'
                np.save(out_path, np.asarray(depth_map, dtype=np.float32))
                total_saved += 1

            log_step(
                f'saved session={session_dir.name} camera={cam} '
                f'count={len(frame_indices)}',
                level='info',
            )

    log_step(f'VDA precompute complete: total_saved={total_saved}', level='info')
