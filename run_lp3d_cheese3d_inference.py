"""
LP3D inference on Cheese3D using real 6-view PNG inputs.

Environment setup (required):
    HF_HOME=/tmp/lp_hf_cache
    HF_ENDPOINT=https://hf-mirror.com
    TRANSFORMERS_OFFLINE=1
    NUMBA_DISABLE_JIT=1
    torch.backends.cudnn.enabled=False

    Usage:
    python run_lp3d_cheese3d_inference.py \
        --ckpt_dir /data/jqh/pretrained_checkpoints/E-RayZer-private/checkpoints/mvt_3d_loss_450_0 \
        --data_root /data/jqh/Datasets/E-RayZer-private/data/cheese3d_cam/cheese3d_cam \
        --output_dir /data/jqh/Outputs/beast/outputs/lp_cheese3d_preds \
        --sessions 20231031_B20_chew_bl_000
"""

import argparse
import os
import re
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/data/jqh/pretrained_checkpoints/E-RayZer-private/checkpoints")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from lightning_pose.api import Model


IMAGE_RE = re.compile(r'^img(?P<idx>\d+)\.png$')


def parse_args():
    p = argparse.ArgumentParser(description="LP3D inference on Cheese3D real 6-view PNG inputs")
    p.add_argument("--ckpt_dir", required=True, type=Path,
                   help="Path to mvt_3d_loss_450_0 checkpoint directory")
    p.add_argument("--data_root", required=True, type=Path,
                   help="Path to cheese3d_cam/cheese3d_cam root")
    p.add_argument("--output_dir", default="/data/jqh/Outputs/beast/outputs/lp_cheese3d_preds", type=Path)
    p.add_argument("--sessions", nargs="+",
                   help="Session folders to process (default: all)")
    p.add_argument("--img_size", type=int, nargs=2, default=[320, 256],
                   help="Original image size W H (default: 320 256)")
    p.add_argument("--resize_h", type=int, default=256,
                   help="LP model resize height (default: 256)")
    p.add_argument("--resize_w", type=int, default=256,
                   help="LP model resize width (default: 256)")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size (images per view per forward pass, default: 1)")
    return p.parse_args()


def preprocess_view_image(
    img: np.ndarray,
    *,
    resize_h: int,
    resize_w: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    img256 = Image.fromarray(img).resize((resize_w, resize_h), Image.BILINEAR)
    arr = np.array(img256, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))


def parse_frame_index(path: Path) -> int | None:
    match = IMAGE_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("idx"))


def collect_png_frames(view_dir: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for path in sorted(view_dir.glob("img*.png")):
        frame_idx = parse_frame_index(path)
        if frame_idx is not None:
            frames[frame_idx] = path
    return frames


def collect_session_frame_map(session_dir: Path, view_names: list[str]) -> tuple[dict[str, dict[int, Path]], list[int]]:
    frames_by_view: dict[str, dict[int, Path]] = {}
    common_indices: set[int] | None = None
    for view_name in view_names:
        view_dir = session_dir / view_name
        if not view_dir.exists():
            raise FileNotFoundError(f"Missing required LP3D view directory: {view_dir}")
        frames = collect_png_frames(view_dir)
        if not frames:
            raise FileNotFoundError(f"No PNG frames found in required LP3D view directory: {view_dir}")
        frames_by_view[view_name] = frames
        frame_indices = set(frames)
        common_indices = frame_indices if common_indices is None else common_indices & frame_indices

    if common_indices is None:
        return frames_by_view, []
    return frames_by_view, sorted(common_indices)


def build_6view_batch(
    frame_paths_by_view: dict[str, Path],
    *,
    view_names: list[str],
    resize_h: int,
    resize_w: int,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cuda",
) -> torch.Tensor:
    """Build a (1, 6, 3, H, W) tensor using real images in checkpoint view order."""
    view_tensors = []
    for view_name in view_names:
        img = np.array(Image.open(frame_paths_by_view[view_name]).convert("RGB"))
        view_tensors.append(
            torch.from_numpy(
                preprocess_view_image(
                    img,
                    resize_h=resize_h,
                    resize_w=resize_w,
                    mean=mean,
                    std=std,
                )
            )
        )

    tensor = torch.stack(view_tensors, dim=0).unsqueeze(0)
    return tensor.to(device)


def main():
    args = parse_args()

    torch.backends.cudnn.enabled = False

    print(f"Loading LP model from {args.ckpt_dir}...")
    t0 = time.perf_counter()
    model = Model.from_dir(str(args.ckpt_dir))
    model._load()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s  |  device: cuda")

    num_kp = model.model.num_keypoints
    view_names = list(model.cfg.data.view_names)
    num_views = len(view_names)
    L_idx = view_names.index("L")
    R_idx = view_names.index("R")
    print(f"Views: {view_names}  |  L_idx={L_idx}, R_idx={R_idx}  |  num_keypoints={num_kp}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    orig_w, orig_h = args.img_size
    scale_x = orig_w / args.resize_w
    scale_y = orig_h / args.resize_h

    if args.sessions:
        sessions = args.sessions
    else:
        sessions = sorted([d.name for d in args.data_root.iterdir() if d.is_dir()])
    print(f"Sessions to process: {sessions}")

    model.model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    for session in sessions:
        sess_dir = args.data_root / session
        if not sess_dir.exists():
            print(f"WARNING: {sess_dir} not found, skipping")
            continue

        try:
            frames_by_view, common_indices = collect_session_frame_map(sess_dir, view_names)
        except FileNotFoundError as exc:
            print(f"WARNING: {exc}; skipping session {session}")
            continue

        if not common_indices:
            print(f"WARNING: No shared frame indices across required views in {sess_dir}, skipping")
            continue

        sess_out = args.output_dir / session
        sess_out.mkdir(exist_ok=True)
        csv_L_out = sess_out / "predictions_L.csv"
        csv_R_out = sess_out / "predictions_R.csv"

        records_L = []
        records_R = []

        for frame_idx_num in tqdm(common_indices, desc=session, unit="frame"):
            frame_name = f"img{frame_idx_num:08d}"
            frame_paths_by_view = {
                view_name: frames_by_view[view_name][frame_idx_num]
                for view_name in view_names
            }

            batch = build_6view_batch(
                frame_paths_by_view,
                view_names=view_names,
                resize_h=args.resize_h,
                resize_w=args.resize_w,
                mean=mean,
                std=std,
            )

            batch_dict = {
                "images": batch,
                "keypoints": torch.zeros(1, num_kp * num_views * 2, device="cuda"),
                "bbox": torch.tensor([[0, 0, args.resize_h, args.resize_w]], device="cuda").repeat(1, num_views),
                "idxs": torch.zeros(1, dtype=torch.long, device="cuda"),
                "heatmaps": torch.zeros(1, num_kp, 1, 1, device="cuda"),
            }

            with torch.inference_mode():
                pred_heatmaps = model.model.forward(batch_dict)

            kp_pred, conf = model.model.head.run_subpixelmaxima(pred_heatmaps)
            kp_reshaped = kp_pred.view(1, num_views, num_kp, 2)
            conf_reshaped = conf.view(1, num_views, num_kp)

            kp_L = kp_reshaped[0, L_idx].cpu().numpy()
            kp_R = kp_reshaped[0, R_idx].cpu().numpy()
            c_L = conf_reshaped[0, L_idx].cpu().numpy()
            c_R = conf_reshaped[0, R_idx].cpu().numpy()

            kp_L[:, 0] *= scale_x
            kp_L[:, 1] *= scale_y
            kp_R[:, 0] *= scale_x
            kp_R[:, 1] *= scale_y

            row_L = [frame_name] + [v for pt in kp_L for v in pt] + list(c_L)
            row_R = [frame_name] + [v for pt in kp_R for v in pt] + list(c_R)
            records_L.append(row_L)
            records_R.append(row_R)

        kp_names = model.cfg.data.keypoint_names
        L_header = ["frame_id"] + [f"{n}_{c}" for n in kp_names for c in ["x", "y"]] + list(kp_names)
        R_header = ["frame_id"] + [f"{n}_{c}" for n in kp_names for c in ["x", "y"]] + list(kp_names)

        df_L = pd.DataFrame(records_L, columns=L_header)
        df_R = pd.DataFrame(records_R, columns=R_header)
        df_L.to_csv(csv_L_out, index=False)
        df_R.to_csv(csv_R_out, index=False)

        nan_L = df_L.drop(columns="frame_id").isna().sum().sum()
        nan_R = df_R.drop(columns="frame_id").isna().sum().sum()
        print(f"  {session}: {len(common_indices)} shared 6-view frames  |  nan_count: L={nan_L}, R={nan_R}")
        print(f"    -> {csv_L_out}")
        print(f"    -> {csv_R_out}")
        total_frames += len(common_indices)

    print(f"\nDone. Total frames processed: {total_frames}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
