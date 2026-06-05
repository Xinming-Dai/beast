#!/usr/bin/env python3
"""
Run LP3D inference on Cheese3D L/R images using the predict_frame API.

Usage:
    python run_lp3d_cheese3d_inference.py \
        --ckpt-dir /home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/mvt_3d_loss_450_0 \
        --data-root /home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam/cheese3d_cam \
        --output-root /home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_lp_inference \
        --sessions 20231031_B20_chew_bl_000 \
        --device cuda

The LP model was trained on 256x256 images. Cheese3D images are 320x256.
We resize to 256x256 for inference, then scale coordinates back to 320x256.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Add lightning-pose to path
_LP_ROOT = Path("/home/jqh/NeuralWorkshops/lightning-pose")
sys.path.insert(0, str(_LP_ROOT))

# Cheese3D image is 320x256, LP model expects 256x256
_CHEESE3D_W = 320
_CHEESE3D_H = 256
_LP_W = 256
_LP_H = 256
_SCALE_X = _LP_W / _CHEESE3D_W  # 0.8 (320 -> 256)
_SCALE_Y = _LP_H / _CHEESE3D_H  # 1.0 (256 -> 256)
# Inverse scale for coordinates: LP output coords * (320/256) = original coords
_INV_SCALE_X = _CHEESE3D_W / _LP_W  # 1.25
_INV_SCALE_Y = _CHEESE3D_H / _LP_H  # 1.0

# Sessions with LP predictions from Lenny
_ALL_SESSIONS = [
    "20231031_B20_chew_bl_000",
    "20231031_B20_chew_temperature_000",
    "20231031_B21_chew_bl_000",
    "20231031_B21_chew_temperature_000",
    "20231031_B26_chew_bl_000",
    "20231031_B26_chew_temperature_000",
    "20231031_B31_chew_bl_000",
    "20231031_B31_chew_temperature_000",
    "20231031_B6_chew_bl_000",
    "20231031_B6_chew_temperature_000",
    "20250523_B1_ephys-record_awake_000",
]


def load_model(ckpt_dir: Path, device: str = "cuda") -> Any:
    """Load LP3D model from checkpoint."""
    from lightning_pose.api import load_model_from_checkpoint
    from omegaconf import OmegaConf

    config_path = ckpt_dir / "config.yaml"
    ckpt_file = ckpt_dir / "tb_logs" / "test_model" / "version_0" / "checkpoints"

    # Find best ckpt
    ckpts = list(ckpt_file.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found in {ckpt_file}")
    ckpt_file = sorted(ckpts)[-1]

    print(f"Loading config from {config_path}")
    print(f"Loading checkpoint from {ckpt_file}")
    cfg = OmegaConf.load(config_path)

    print("Building model...")
    model = load_model_from_checkpoint(
        cfg=cfg,
        ckpt_file=str(ckpt_file),
        eval=True,
        skip_data_module=True,
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model


def preprocess_frame(img: Image.Image) -> np.ndarray:
    """Convert PIL image to LP model input (256x256 RGB)."""
    img = img.resize((_LP_W, _LP_H), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    # LP expects RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def infer_frame(model: Any, frame_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Run inference on a single frame."""
    return model.predict_frame(frame_rgb)


def run_inference_on_session(
    model: Any,
    session_dir: Path,
    session_id: str,
    cameras: list[str],
    output_root: Path,
    device: str,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Run LP inference on all L/R frames of one session."""
    rows_out = []
    start = time.perf_counter()

    # Collect all image files per camera
    camera_files: dict[str, dict[int, Path]] = {}
    for cam in cameras:
        cam_dir = session_dir / cam
        if not cam_dir.exists():
            continue
        files: dict[int, Path] = {}
        for f in cam_dir.iterdir():
            if f.suffix == ".png" and f.stem.startswith("img"):
                try:
                    idx = int(f.stem.replace("img", ""))
                    files[idx] = f
                except ValueError:
                    pass
        camera_files[cam] = files

    # Find common frame indices across cameras
    if not camera_files:
        return rows_out

    common_indices = None
    for cam_files in camera_files.values():
        indices = set(cam_files.keys())
        common_indices = indices if common_indices is None else common_indices & indices

    if not common_indices:
        print(f"[warn] No common frames for {session_id}")
        return rows_out

    print(
        f"[info] {session_id}: {len(common_indices)} common frames across {cameras}",
        flush=True,
    )

    # Process each camera's frames and save as CSV
    for cam in cameras:
        cam_files = camera_files.get(cam, {})
        out_csv = output_root / session_id / f"predictions_{cam}.csv"
        if out_csv.exists() and not overwrite:
            print(f"[info] {cam}: CSV exists, skipping ({out_csv})")
            rows_out.append(
                {
                    "session_id": session_id,
                    "camera": cam,
                    "status": "skipped_existing",
                }
            )
            continue

        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # Keypoints from model config
        cfg_path = Path("/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/mvt_3d_loss_450_0/config.yaml")
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(cfg_path)
        kp_names = list(cfg.data.keypoint_names)
        num_kp = len(kp_names)

        # Write CSV header
        # Format: scorer, bodyparts (x,y,lh × N), coords headers
        header0 = ["scorer"] + ["heatmap_multiview_transformer_tracker"] * (num_kp * 3)
        header1 = ["bodyparts"] + kp_names * 3
        header2 = ["coords"] + (["x", "y", "likelihood"] * num_kp)

        sorted_indices = sorted(common_indices)
        csv_rows = [header0, header1, header2]

        t0 = time.perf_counter()
        count = 0
        for idx in sorted_indices:
            img_path = cam_files[idx]
            img = Image.open(img_path).convert("RGB")
            frame_rgb = preprocess_frame(img)
            result = infer_frame(model, frame_rgb)

            keypoints = result["keypoints"]  # (N, 2) in LP space (256x256)
            confidence = result["confidence"]  # (N,)

            # Scale back to Cheese3D original space (320x256)
            keypoints[:, 0] *= _INV_SCALE_X
            keypoints[:, 1] *= _INV_SCALE_Y

            row = [img_path.name]
            for ki in range(num_kp):
                if ki < len(keypoints):
                    row.extend(
                        [
                            f"{keypoints[ki, 0]:.8f}",
                            f"{keypoints[ki, 1]:.8f}",
                            f"{confidence[ki]:.16f}",
                        ]
                    )
                else:
                    row.extend(["", "", ""])
            csv_rows.append(row)
            count += 1

        elapsed = time.perf_counter() - t0
        fps = count / elapsed if elapsed > 0 else 0
        print(f"[info] {cam}: {count} frames in {elapsed:.1f}s ({fps:.1f} fps)", flush=True)

        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)

        rows_out.append(
            {
                "session_id": session_id,
                "camera": cam,
                "status": "written",
                "csv_path": str(out_csv),
                "frames": count,
                "fps": round(fps, 1),
            }
        )

    total = time.perf_counter() - start
    print(f"[info] {session_id}: done in {total:.1f}s", flush=True)
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Specific sessions (default: all 11 sessions)",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["L", "R"],
        help="Camera names (default: L R)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sessions = args.sessions or _ALL_SESSIONS
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    print(f"CKPT: {args.ckpt_dir}")
    print(f"Data: {data_root}")
    print(f"Output: {output_root}")
    print(f"Sessions: {sessions}")
    print(f"Cameras: {args.cameras}")
    print(f"Device: {args.device}")
    print()

    model = load_model(args.ckpt_dir, args.device)

    all_rows = []
    for session_id in sessions:
        session_dir = data_root / session_id
        if not session_dir.exists():
            print(f"[warn] Session not found: {session_dir}")
            continue
        rows = run_inference_on_session(
            model=model,
            session_dir=session_dir,
            session_id=session_id,
            cameras=args.cameras,
            output_root=output_root,
            device=args.device,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)

    summary_path = output_root / "inference_summary.json"
    summary_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
