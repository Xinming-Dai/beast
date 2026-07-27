#!/usr/bin/env python3
"""Precompute VDA depth maps for the 4b00 IBL session in the format expected by
beast.data.sable_dataset.IBLTwoViewDataset.

Output path layout (matches IBLTwoViewDataset._load_vda_depth):
    {cache_root}/{session_id}/{camera_name}/{frame_idx:06d}.npy

Where:
    cache_root   = /data/jqh/Datasets/beast3d-data/sable_ibl_4b00/vda_cache
    session_id   = 4b00df29-3769-43be-bb40-128b1cba6d35
    camera_name  = 'left' or 'right'
    frame_idx    = source_frame_index (8-digit numeric extracted from img{N:08d}.png)

VDA checkpoint:
    /home/jqh/NeuralWorkshops/third_party/VDA/checkpoints/video_depth_anything_vitb.pth
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# beast repo on sys.path so we can import the frozen VDA loader
BEAST_ROOT = "/cephfs/jinqihang/SABLE/beast"
sys.path.insert(0, BEAST_ROOT)

from beast.models.model_utils.utils_vda import load_frozen_video_depth_anything  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults for the 4b00 smoke setup
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path("/cephfs/jinqihang/SABLE/datasets/beast3d-data/sable_ibl_4b00")
DEFAULT_CACHE_ROOT = Path("/cephfs/jinqihang/SABLE/datasets/beast3d-data/sable_ibl_4b00/vda_cache")
DEFAULT_SESSION_ID = "4b00df29-3769-43be-bb40-128b1cba6d35"
DEFAULT_VDA_CKPT = "/cephfs/jinqihang/SABLE/third_party/VDA/checkpoints/video_depth_anything_vitb.pth"
DEFAULT_VDA_REPO = "/cephfs/jinqihang/SABLE/third_party/VDA"
DEFAULT_DEVICE = "cuda:1"  # stays out of GPU 0 (occupied by another user)

# Map IBL filesystem layout -> IBLTwoViewDataset camera names
CAMERA_DIRS = {
    "left": "leftCamera.video/_iblrig_leftCamera.downsampled.{}",
    "right": "rightCamera.video/_iblrig_rightCamera.downsampled.{}",
}

IMG_PATTERN = re.compile(r"img(0[0-9]+)\.png$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--session-id", type=str, default=DEFAULT_SESSION_ID)
    p.add_argument("--vda-ckpt", type=str, default=DEFAULT_VDA_CKPT)
    p.add_argument("--vda-repo", type=str, default=DEFAULT_VDA_REPO)
    p.add_argument("--encoder", type=str, default="vitb", choices=["vits", "vitb", "vitl"])
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p.add_argument("--input-size", type=int, default=518)
    p.add_argument("--overwrite", action="store_true", help="Recompute npy files even if present")
    p.add_argument("--cameras", type=str, default="left,right",
                   help="Comma-separated list of cameras to process (default: left,right)")
    p.add_argument("--frame-limit", type=int, default=0,
                   help="If >0, only process this many frames per camera (debug).")
    return p.parse_args()


def list_frames(image_root: Path, session_id: str, camera: str) -> list[tuple[int, Path]]:
    """Return sorted (frame_idx, png_path) for one camera."""
    subdir = CAMERA_DIRS[camera].format(session_id)
    cam_dir = image_root / subdir
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"camera dir not found: {cam_dir}")
    entries: list[tuple[int, Path]] = []
    for png in cam_dir.glob("img*.png"):
        m = IMG_PATTERN.search(png.name)
        if m is None:
            continue
        idx = int(m.group(1))
        entries.append((idx, png))
    entries.sort(key=lambda t: t[0])
    return entries


def load_rgb(paths: list[Path]) -> np.ndarray:
    """Stack HxWx3 uint8 frames into [N, H, W, 3]."""
    out = np.empty((len(paths), 0, 0, 3), dtype=np.uint8)
    rows = []
    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        rows.append(arr)
    return np.stack(rows, axis=0)


def main() -> int:
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    args.cache_root.mkdir(parents=True, exist_ok=True)

    print(f"[precompute] dataset_root = {args.dataset_root}")
    print(f"[precompute] cache_root   = {args.cache_root}")
    print(f"[precompute] session_id   = {args.session_id}")
    print(f"[precompute] cameras      = {cameras}")
    print(f"[precompute] device       = {args.device}")
    print(f"[precompute] encoder      = {args.encoder}")
    print(f"[precompute] input_size   = {args.input_size}")

    # Load frozen VDA model
    vda_cfg = {
        "encoder": args.encoder,
        "checkpoint_path": args.vda_ckpt,
        "repo_root": args.vda_repo,
        "metric": False,
    }
    print(f"[precompute] loading VDA model ...")
    model = load_frozen_video_depth_anything(vda_cfg).to(args.device)
    model.eval()

    total_saved = 0
    total_skipped = 0
    t_start = time.time()

    for camera in cameras:
        cam_dir = args.cache_root / args.session_id / camera
        cam_dir.mkdir(parents=True, exist_ok=True)

        frames = list_frames(args.dataset_root, args.session_id, camera)
        if args.frame_limit > 0:
            frames = frames[: args.frame_limit]
        print(f"[precompute] camera={camera}: {len(frames)} frames to consider")

        # Pre-filter by cache
        todo: list[tuple[int, Path]] = []
        for idx, path in frames:
            out_path = cam_dir / f"{idx:06d}.npy"  # 6-digit, matches IBLTwoViewDataset._load_vda_depth
            if out_path.exists() and not args.overwrite:
                total_skipped += 1
                continue
            todo.append((idx, path))
        print(f"[precompute] camera={camera}: {len(todo)} to compute, {total_skipped} cached")

        if not todo:
            continue

        # Process in batches to keep VRAM bounded (VDA needs ~3 GB on ViT-B).
        batch_size = 32
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start : batch_start + batch_size]
            frame_indices = [idx for idx, _ in batch]
            frame_paths = [path for _, path in batch]
            t0 = time.time()
            frames_np = load_rgb(frame_paths)
            with torch.no_grad():
                depth_maps, _ = model.infer_video_depth(
                    frames_np,
                    target_fps=1.0,
                    input_size=args.input_size,
                    device=args.device,
                    fp32=False,
                )
            if depth_maps.shape[0] != len(batch):
                raise RuntimeError(
                    f"VDA returned {depth_maps.shape[0]} depth maps for {len(batch)} frames"
                )
            for idx, depth_map in zip(frame_indices, depth_maps):
                out_path = cam_dir / f"{idx:06d}.npy"
                np.save(out_path, np.asarray(depth_map, dtype=np.float32))
                total_saved += 1
            torch.cuda.synchronize(args.device)
            dt = time.time() - t0
            print(
                f"[precompute] {camera}: {batch_start + len(batch)}/{len(todo)} "
                f"({dt:.2f}s, {len(batch) / dt:.1f} fps)",
                flush=True,
            )

    total_dt = time.time() - t_start
    print(f"[precompute] DONE: saved={total_saved} skipped={total_skipped} "
          f"total_time={total_dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
