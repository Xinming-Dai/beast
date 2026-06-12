#!/usr/bin/env python3
"""
Validate Stage 1 Cheese3D correspondence cache against dataset contracts.

Checks:
- selected_frames parsing on real session files
- cache bundle existence for selected dataset frames
- coordinate / shape sanity for litpose_matches bundles
- dataset-side rescaling from 320x256 to image_size
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from beast.data.cheese3d_dataset import (
    Cheese3DDataset,
    _resolve_dataset_dir,
    build_frame_index,
    load_selected_frame_indices,
)
from beast.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Cheese3D Stage 1 cache contract")
    parser.add_argument("--config", type=Path, required=True, help="Path to SABLE Cheese3D config")
    parser.add_argument("--cache_root", type=Path, required=True, help="Root containing session/pair_xxxxxx/litpose_matches.npz")
    parser.add_argument("--sessions", nargs="*", default=None, help="Optional session ids to validate")
    parser.add_argument("--sample_limit", type=int, default=32, help="Max bundles to inspect deeply")
    return parser.parse_args()


def summarize_npz(npz_path: Path, image_size: int) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    left_xy = np.asarray(data["left_xy"], dtype=np.float32)
    right_xy = np.asarray(data["right_xy"], dtype=np.float32)
    confidence = np.asarray(data["confidence"], dtype=np.float32)
    labels = np.asarray(data["labels"])
    left_orig_w = float(data["left_orig_w"])
    left_orig_h = float(data["left_orig_h"])
    right_orig_w = float(data["right_orig_w"])
    right_orig_h = float(data["right_orig_h"])
    metadata_json = data.get("metadata_json")
    metadata = {}
    if metadata_json is not None:
        metadata = json.loads(str(np.asarray(metadata_json).item()))

    if left_xy.shape != right_xy.shape:
        raise ValueError(f"Shape mismatch in {npz_path}: left={left_xy.shape} right={right_xy.shape}")
    if left_xy.shape[0] != confidence.shape[0] or left_xy.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Length mismatch in {npz_path}: pts={left_xy.shape[0]} conf={confidence.shape[0]} labels={labels.shape[0]}"
        )
    if left_xy.ndim != 2 or left_xy.shape[-1] != 2:
        raise ValueError(f"Expected Nx2 coordinates in {npz_path}, got {left_xy.shape}")
    if not np.isfinite(left_xy).all() or not np.isfinite(right_xy).all() or not np.isfinite(confidence).all():
        raise ValueError(f"Non-finite values found in {npz_path}")
    if ((confidence < 0.0) | (confidence > 1.0)).any():
        raise ValueError(f"Confidence outside [0,1] in {npz_path}")
    if left_xy.size:
        if (left_xy[:, 0] < 0).any() or (left_xy[:, 0] > left_orig_w).any() or (left_xy[:, 1] < 0).any() or (left_xy[:, 1] > left_orig_h).any():
            raise ValueError(f"Left raw coordinates out of range in {npz_path}")
        if (right_xy[:, 0] < 0).any() or (right_xy[:, 0] > right_orig_w).any() or (right_xy[:, 1] < 0).any() or (right_xy[:, 1] > right_orig_h).any():
            raise ValueError(f"Right raw coordinates out of range in {npz_path}")

    scaled_left = left_xy.copy()
    scaled_right = right_xy.copy()
    if scaled_left.size:
        scaled_left[:, 0] *= image_size / left_orig_w
        scaled_left[:, 1] *= image_size / left_orig_h
        scaled_right[:, 0] *= image_size / right_orig_w
        scaled_right[:, 1] *= image_size / right_orig_h
        if (scaled_left[:, 0] < 0).any() or (scaled_left[:, 0] > image_size).any() or (scaled_left[:, 1] < 0).any() or (scaled_left[:, 1] > image_size).any():
            raise ValueError(f"Left scaled coordinates out of range in {npz_path}")
        if (scaled_right[:, 0] < 0).any() or (scaled_right[:, 0] > image_size).any() or (scaled_right[:, 1] < 0).any() or (scaled_right[:, 1] > image_size).any():
            raise ValueError(f"Right scaled coordinates out of range in {npz_path}")

    return {
        "npz_path": str(npz_path),
        "num_points": int(left_xy.shape[0]),
        "left_min": left_xy.min(axis=0).tolist() if left_xy.size else None,
        "left_max": left_xy.max(axis=0).tolist() if left_xy.size else None,
        "right_min": right_xy.min(axis=0).tolist() if right_xy.size else None,
        "right_max": right_xy.max(axis=0).tolist() if right_xy.size else None,
        "scaled_left_max": scaled_left.max(axis=0).tolist() if scaled_left.size else None,
        "scaled_right_max": scaled_right.max(axis=0).tolist() if scaled_right.size else None,
        "mean_confidence": float(confidence.mean()) if confidence.size else 0.0,
        "split": metadata.get("split"),
        "variant": metadata.get("keypoint_variant"),
        "left_orig_w": left_orig_w,
        "left_orig_h": left_orig_h,
    }


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    config["model"]["merge_pcd"]["correspondence_cache_root"] = str(args.cache_root)
    if args.sessions:
        config["training"]["sessions"] = list(args.sessions)

    dataset = Cheese3DDataset(config)
    dataset_root = Path(config["training"]["dataset_path"])
    dataset_dir = _resolve_dataset_dir(dataset_root)
    views = config["training"]["views"]
    sessions = config["training"].get("sessions")

    records, summary = build_frame_index(
        root=dataset_root,
        views=views,
        sessions=sessions,
        start_frame=int(config["training"].get("start_frame", 0)),
        frame_step=int(config["training"].get("frame_step", 1)),
        max_frames_per_session=config["training"].get("max_frames_per_session"),
        require_masks=not bool(config["training"].get("allow_missing_masks", False)),
    )

    inspected_sessions = sessions or list(summary["records_per_session"].keys())
    print("Selected-frame parser check:")
    for session in inspected_sessions:
        session_dir = dataset_dir / session
        selected = load_selected_frame_indices(session_dir)
        print(f"  {session}: selected_frames={len(selected)} first5={sorted(selected)[:5]}")

    print("\nDataset summary:")
    print(f"  Records built: {len(records)}")
    print(f"  Dataset length: {len(dataset)}")
    print(f"  Image size: {dataset.image_size}")

    missing = []
    inspected = []
    empty_count = 0
    min_points = None
    max_points = 0
    min_required_points = int(config["model"]["merge_pcd"].get("num_points", 3))
    max_empty_ratio = float(config["training"].get("max_empty_bundle_ratio", 0.0))

    for record in records:
        session_id = record["session_id"]
        frame_idx = int(record["frame_idx"])
        npz_path = args.cache_root / session_id / f"pair_{frame_idx:06d}" / "litpose_matches.npz"
        if not npz_path.exists():
            missing.append(str(npz_path))
            continue
        if len(inspected) < args.sample_limit:
            stats = summarize_npz(npz_path, dataset.image_size)
            inspected.append(stats)
        data = np.load(npz_path, allow_pickle=True)
        n = int(np.asarray(data["left_xy"]).shape[0])
        empty_count += int(n == 0)
        min_points = n if min_points is None else min(min_points, n)
        max_points = max(max_points, n)

    if missing:
        print("\nMissing bundles:")
        for path in missing[:20]:
            print(f"  {path}")
        raise FileNotFoundError(f"Missing {len(missing)} cache bundles referenced by dataset")

    print("\nBundle coverage:")
    print(f"  Referenced records: {len(records)}")
    print(f"  Empty bundles: {empty_count}")
    print(f"  Min points/frame: {min_points}")
    print(f"  Max points/frame: {max_points}")

    if min_points is None:
        raise ValueError("No cache bundles were inspected")
    empty_ratio = empty_count / len(records) if records else 0.0
    if min_points < min_required_points:
        raise ValueError(
            f"Minimum correspondences per frame too low: {min_points} < required {min_required_points}"
        )
    if empty_ratio > max_empty_ratio:
        raise ValueError(
            f"Empty-bundle ratio too high: {empty_ratio:.4f} > allowed {max_empty_ratio:.4f}"
        )

    print("\nSample bundle stats:")
    for item in inspected[:10]:
        print(
            f"  {item['npz_path']}: K={item['num_points']} conf={item['mean_confidence']:.4f} "
            f"split={item['split']} variant={item['variant']} "
            f"left_max={item['left_max']} scaled_left_max={item['scaled_left_max']}"
        )

    sample = dataset[0]
    print("\nDataset sample tensors:")
    print(f"  image: {tuple(sample['image'].shape)}")
    print(f"  leftcamera_xy: {tuple(sample['leftcamera_xy'].shape)}")
    print(f"  rightcamera_xy: {tuple(sample['rightcamera_xy'].shape)}")
    print(f"  confidence: {tuple(sample['confidence'].shape)}")
    if sample["leftcamera_xy"].numel() > 0:
        print(f"  leftcamera_xy max: {sample['leftcamera_xy'].max(dim=0).values.tolist()}")
        print(f"  rightcamera_xy max: {sample['rightcamera_xy'].max(dim=0).values.tolist()}")

    print("\nStage 1 dataset/cache contract validation passed.")


if __name__ == "__main__":
    main()
