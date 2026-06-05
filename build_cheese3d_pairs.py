"""
Build Cheese3D opencv_cameras_pairs.txt from LP3D predictions directory.

Creates:
  {output_dir}/sessions/{session}/pairs/opencv_cameras_pair_{frame_idx:06d}.json

And outputs:
  {output_dir}/opencv_cameras_pairs.txt  (list of JSON paths)

Usage:
    python build_cheese3d_pairs.py \
        --data_root /home/jqh/NeuralWorkshops/E-RayZer-private/data/cheese3d_cam/cheese3d_cam \
        --pred_csv_dir /tmp/lp_cheese3d_preds \
        --output_dir /tmp/cheese3d_ibl_pairs \
        --sessions 20231031_B20_chew_bl_000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, type=Path)
    p.add_argument("--pred_csv_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--sessions", nargs="+",
                   help="Sessions to process (default: all found in data_root)")
    p.add_argument("--img_w", type=int, default=320)
    p.add_argument("--img_h", type=int, default=256)
    p.add_argument("--fx", type=float, default=260.0,
                   help="Focal length x (in pixels)")
    p.add_argument("--fy", type=float, default=260.0,
                   help="Focal length y (in pixels)")
    p.add_argument("--cx", type=float, default=None,
                   help="Principal point x (default: img_w/2)")
    p.add_argument("--cy", type=float, default=None,
                   help="Principal point y (default: img_h/2)")
    return p.parse_args()


def identity_matrix():
    return [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]


def build_session_pairs(args, session: str) -> list[Path]:
    """Build all L/R pair JSONs for one session. Returns list of JSON paths."""
    sess_dir = args.data_root / session
    L_dir = sess_dir / "L"
    R_dir = sess_dir / "R"

    if not L_dir.exists() or not R_dir.exists():
        print(f"WARNING: L/R dirs not found for {session}, skipping")
        return []

    # Get frames from predictions CSVs
    csv_L = args.pred_csv_dir / session / "predictions_L.csv"
    if not csv_L.exists():
        print(f"WARNING: {csv_L} not found, skipping")
        return []

    df = pd.read_csv(csv_L, usecols=["frame_id"])
    frame_ids = df["frame_id"].tolist()

    out_root = args.output_dir / "sessions" / session / "pairs"
    out_root.mkdir(parents=True, exist_ok=True)

    cx = args.cx if args.cx is not None else args.img_w / 2
    cy = args.cy if args.cy is not None else args.img_h / 2

    pair_paths = []
    for frame_idx, frame_id in enumerate(tqdm(frame_ids, desc=f"{session} pairs", unit="frame")):
        pair_dir = out_root / f"opencv_cameras_pair_{frame_idx:06d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Image paths: absolute (SABLE resolves from JSON dir, absolute is fine)
        img_L = str(L_dir / f"{frame_id}.png")
        img_R = str(R_dir / f"{frame_id}.png")

        pair_data = {
            "scene_name": session,
            "session_id": session,
            "pair_id": frame_idx,
            "query_time_sec": 0.0,
            "source_format": "cheese3d_lp3d",
            "camera_toml": str(sess_dir / "calibration.toml"),
            "frames": [
                {
                    "camera_name": "left",
                    "file_path": img_L,
                    "timestamp_sec": 0.0,
                    "source_frame_index": frame_idx,
                    "fx": args.fx,
                    "fy": args.fy,
                    "cx": cx,
                    "cy": cy,
                    "w2c": identity_matrix(),
                    "actual_timestamp_sec": 0.0,
                },
                {
                    "camera_name": "right",
                    "file_path": img_R,
                    "timestamp_sec": 0.0,
                    "source_frame_index": frame_idx,
                    "fx": args.fx,
                    "fy": args.fy,
                    "cx": cx,
                    "cy": cy,
                    "w2c": identity_matrix(),
                    "actual_timestamp_sec": 0.0,
                },
            ],
            "metadata": {
                "backend": "lp3d_cheese3d_320x256",
                "source": "LP3D predictions",
                "split": "train",
            },
        }

        json_path = pair_dir / "opencv_cameras_pair.json"
        with open(json_path, "w") as f:
            json.dump(pair_data, f)

        pair_paths.append(json_path)

    return pair_paths


def main():
    args = parse_args()

    if args.sessions:
        sessions = args.sessions
    else:
        sessions = sorted([d.name for d in args.data_root.iterdir()
                          if d.is_dir() and (d / "L").exists() and (d / "R").exists()])

    print(f"Sessions: {sessions}")
    print(f"Output: {args.output_dir}")

    all_paths = []
    for session in sessions:
        paths = build_session_pairs(args, session)
        all_paths.extend(paths)

    # Write pairs.txt
    pairs_txt = args.output_dir / "opencv_cameras_pairs.txt"
    with open(pairs_txt, "w") as f:
        for p in sorted(all_paths):
            f.write(str(p) + "\n")

    print(f"\nTotal pairs: {len(all_paths)}")
    print(f"Pairs list: {pairs_txt}")
    print(f"First path: {all_paths[0] if all_paths else 'NONE'}")


if __name__ == "__main__":
    main()
