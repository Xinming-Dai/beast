#!/usr/bin/env python3
"""
Batch convert LP3D CSV predictions -> litpose_matches.npz for all sessions.
"""

import argparse
from pathlib import Path

from convert_csv_to_litpose_cache import convert_session


def parse_args():
    p = argparse.ArgumentParser(description="Batch CSV -> litpose_matches.npz converter")
    p.add_argument("--pred_csv_dir", required=True, type=Path,
                   help="Dir containing session subdirs with predictions_{L,R}.csv")
    p.add_argument("--output_root", required=True, type=Path,
                   help="Root of SABLE correspondence cache")
    p.add_argument("--min_confidence", type=float, default=0.8)
    p.add_argument("--orig_w", type=int, default=320)
    p.add_argument("--orig_h", type=int, default=256)
    p.add_argument("--keypoints", type=str, default=None,
                   help="Comma-separated keypoint names to include (default: all)")
    p.add_argument("--keypoint_variant", type=str, default="all28",
                   choices=["all28", "rigidHead", "dynamicHighConf"])
    p.add_argument("--dynamic_min_confidence", type=float, default=0.95)
    p.add_argument("--backend", type=str, default="lp3d_cheese3d_320x256")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    csv_dir = Path(args.pred_csv_dir)
    output_root = Path(args.output_root)

    sessions = sorted([
        d.name for d in csv_dir.iterdir()
        if d.is_dir() and (d / "predictions_L.csv").exists() and (d / "predictions_R.csv").exists()
    ])

    if not sessions:
        print(f"No sessions found in {csv_dir}")
        return

    print(f"Processing {len(sessions)} sessions:")
    for session in sessions:
        print(f"  - {session}")
    print()

    for session in sessions:
        print(f"Session: {session}")
        convert_session(csv_dir, session, output_root, args)


if __name__ == "__main__":
    main()
