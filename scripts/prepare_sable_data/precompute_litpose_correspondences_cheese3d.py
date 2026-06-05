#!/usr/bin/env python3
"""
Build LitPose correspondence bundles for Cheese3D sessions from Lenny's LP predictions.

Input: E-RayZer-private/checkpoints/mvt_3d_loss_450_0/predictions_{L,R}.csv
Output: outputs under correspondence_cache_root/

Cheese3D specifics (vs IBL rig):
  - Images are 320x256 (square-ish, NOT 256x320 L / 320x256 R)
  - LP model was trained on 256x256 resized images
  - Scale: x *= 320/256 = 1.25, y *= 256/256 = 1.0
  - L and R cameras are synchronized (same frame indices)
  - Correspondence is via frame index matching
  - 28 keypoints (but only a subset have good likelihood)

For BEAST/SABLE training we use L and R as the stereo pair.
The correspondence_cache_root directory structure:
  {cache_root}/{session_id}/pair_{pair_idx:06d}/litpose_matches.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Cheese3D: LP predictions are in 256x256, images are 320x256
_LP_RESIZE_W = 256
_LP_RESIZE_H = 256
_CHEESE3D_ORIG_W = 320
_CHEESE3D_ORIG_H = 256

_SCALE_X = _CHEESE3D_ORIG_W / _LP_RESIZE_W  # 1.25
_SCALE_Y = _CHEESE3D_ORIG_H / _LP_RESIZE_H  # 1.0

_BUNDLE_FILENAME = "litpose_matches.npz"

# Sessions covered by Lenny's model
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


def _parse_session_from_path(path_str: str) -> tuple[str, str]:
    """Extract session_id and frame_idx from a prediction CSV path.

    e.g. 'labeled-data/20231031_B20_chew_bl_000_08-38-25_L/img00000676.png'
    Returns: ('20231031_B20_chew_bl_000', 676)
    """
    fname = path_str.split("/")[-1]  # img00000676.png
    frame_idx = int(re.search(r"img(\d+)", fname).group(1))

    for suffix in ["_L", "_R", "_BC", "_TC", "_TL", "_TR"]:
        if suffix in path_str:
            idx = path_str.rfind(suffix)
            rest = path_str[:idx]
            # Remove timestamp HH-MM-SS from end
            m = re.match(r"(.+)_(\d{2}-\d{2}-\d{2})$", rest)
            if m:
                session = m.group(1)
            else:
                session = rest
            # Strip 'labeled-data/' prefix if present
            session = session.replace("labeled-data/", "")
            return session, frame_idx

    raise ValueError(f"Cannot parse session from path: {path_str}")


def _load_predictions(csv_path: Path, session_id: str) -> dict[int, list[str]]:
    """Load a predictions CSV, return {frame_idx: row_values} for a specific session.

    row_values are the full CSV row (including scorer column).
    """
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    if len(rows) < 4:
        return {}

    frame_map: dict[int, list[str]] = {}
    bodyparts_flat = rows[1][1:]
    expected = len(bodyparts_flat)

    for r in rows[3:]:
        if not r:
            continue
        try:
            session, frame_idx = _parse_session_from_path(r[0])
        except (ValueError, IndexError):
            continue

        # Filter: only keep rows from the target session
        if session != session_id:
            continue

        vals = r[1:1 + expected]
        if len(vals) < expected:
            continue

        if frame_idx not in frame_map:
            frame_map[frame_idx] = vals

    return frame_map


def _extract_keypoints_from_row(
    row_vals: list[str],
    bodyparts_flat: list[str],
    keypoint_names: list[str],
    min_likelihood: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract (x, y, confidence, labels) from a CSV row."""
    # Build keypoint name -> column start
    kp_starts: dict[str, int] = {}
    for i in range(0, len(bodyparts_flat), 3):
        name = bodyparts_flat[i]
        if name and name in keypoint_names and name not in kp_starts:
            kp_starts[name] = i

    left_list: list[tuple[float, float]] = []
    right_list: list[tuple[float, float]] = []
    conf_list: list[float] = []
    labels: list[str] = []

    for name in keypoint_names:
        start = kp_starts.get(name)
        if start is None:
            continue
        try:
            x = float(row_vals[start])
            y = float(row_vals[start + 1])
            lh = float(row_vals[start + 2])
        except (ValueError, IndexError):
            continue
        if not (lh >= min_likelihood and x > 0 and y > 0):
            continue
        left_list.append((x, y))
        conf_list.append(lh)
        labels.append(name)

    if not left_list:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32), []

    # left_xy and right_xy are both in LP space (256x256) initially
    left_xy = np.asarray(left_list, dtype=np.float32)
    # Rescale: 256x256 -> 320x256
    left_xy[:, 0] *= _SCALE_X
    left_xy[:, 1] *= _SCALE_Y
    right_xy = left_xy.copy()  # Will be overwritten by right camera data

    return left_xy, right_xy, np.asarray(conf_list, dtype=np.float32), labels


def _build_bodyparts_flat(csv_path: Path) -> list[str]:
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    if len(rows) < 2:
        return []
    return rows[1][1:]


def _build_correspondences_for_session(
    lp_dir: Path,
    session_id: str,
    keypoint_names: list[str],
    min_likelihood: float,
    output_root: Path,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Build correspondence bundles for one Cheese3D session.

    Finds L/R pairs with matching frame indices and creates NPZ bundles.
    """
    rows_out: list[dict[str, Any]] = []

    # Load L and R predictions
    l_csv = lp_dir / f"predictions_L.csv"
    r_csv = lp_dir / f"predictions_R.csv"

    if not l_csv.exists() or not r_csv.exists():
        print(f"[warn] Missing predictions CSV for session={session_id}", file=sys.stderr)
        return rows_out

    print(f"[info] Loading {session_id}...")
    t0 = time.perf_counter()

    # Load bodyparts
    bp_l = _build_bodyparts_flat(l_csv)
    bp_r = _build_bodyparts_flat(r_csv)

    if bp_l != bp_r:
        print(f"[warn] Bodyparts mismatch L vs R for {session_id}", file=sys.stderr)
        return rows_out

    # Load frame maps (per-session filtering applied)
    l_map = _load_predictions(l_csv, session_id)
    r_map = _load_predictions(r_csv, session_id)

    print(
        f"[info] {session_id}: L={len(l_map)} frames, R={len(r_map)} frames "
        f"elapsed={time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # Find shared frame indices
    shared_frames = sorted(set(l_map.keys()) & set(r_map.keys()))
    print(f"[info] {session_id}: {len(shared_frames)} shared L/R pairs", flush=True)

    if not shared_frames:
        return rows_out

    # Sort frames for consistent pair_idx ordering
    shared_frames.sort()

    # Build kp_starts for keypoint extraction
    kp_starts: dict[str, int] = {}
    for i in range(0, len(bp_l), 3):
        name = bp_l[i]
        if name and name in keypoint_names and name not in kp_starts:
            kp_starts[name] = i

    for frame_idx in shared_frames:
        # IMPORTANT: use actual frame_idx as pair dir name so dataset lookup succeeds
        out_dir = output_root / session_id / f"pair_{int(frame_idx):06d}"
        out_path = out_dir / _BUNDLE_FILENAME

        if out_path.exists() and not overwrite:
            rows_out.append(
                {
                    "session_id": session_id,
                    "frame_idx": int(frame_idx),
                    "status": "skipped_existing",
                    "bundle_path": str(out_path),
                }
            )
            continue

        l_vals = l_map[frame_idx]
        r_vals = r_map[frame_idx]

        # Extract keypoints for L camera
        l_kp_list: list[tuple[float, float]] = []
        l_conf_list: list[float] = []
        l_labels: list[str] = []

        for name in keypoint_names:
            start = kp_starts.get(name)
            if start is None:
                continue
            try:
                x = float(l_vals[start])
                y = float(l_vals[start + 1])
                lh = float(l_vals[start + 2])
            except (ValueError, IndexError):
                continue
            if lh < min_likelihood or x <= 0 or y <= 0:
                continue
            l_kp_list.append((x, y))
            l_conf_list.append(lh)
            l_labels.append(name)

        # Extract keypoints for R camera
        r_kp_list: list[tuple[float, float]] = []
        r_conf_list: list[float] = []

        for name in keypoint_names:
            start = kp_starts.get(name)
            if start is None:
                continue
            try:
                x = float(r_vals[start])
                y = float(r_vals[start + 1])
                lh = float(r_vals[start + 2])
            except (ValueError, IndexError):
                continue
            if lh < min_likelihood or x <= 0 or y <= 0:
                continue
            r_kp_list.append((x, y))
            r_conf_list.append(lh)

        if len(l_kp_list) < 3:
            rows_out.append(
                {
                    "session_id": session_id,
                    "frame_idx": int(frame_idx),
                    "status": "insufficient_keypoints",
                }
            )
            continue

        # Build arrays: both cameras in LP 256x256 space -> rescale to 320x256
        left_xy = np.asarray(l_kp_list, dtype=np.float32)
        left_xy[:, 0] *= _SCALE_X
        left_xy[:, 1] *= _SCALE_Y

        right_xy = np.asarray(r_kp_list, dtype=np.float32)
        right_xy[:, 0] *= _SCALE_X
        right_xy[:, 1] *= _SCALE_Y

        # Use minimum confidence across cameras
        min_conf = np.minimum(
            np.asarray(l_conf_list, dtype=np.float32),
            np.asarray(r_conf_list, dtype=np.float32)[: len(l_conf_list)],
        )

        # Ensure same length (use first N)
        n = min(len(left_xy), len(right_xy))
        left_xy = left_xy[:n]
        right_xy = right_xy[:n]
        min_conf = min_conf[:n]
        labels = l_labels[:n]

        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "backend": "lenny_lp3d_cheese3d",
            "session_id": session_id,
            "frame_idx": int(frame_idx),
            "source": str(lp_dir),
            "keypoints": list(keypoint_names),
            "left_orig_space": f"lp_256x256_rescaled_to_{_CHEESE3D_ORIG_W}x{_CHEESE3D_ORIG_H}",
            "right_orig_space": f"lp_256x256_rescaled_to_{_CHEESE3D_ORIG_W}x{_CHEESE3D_ORIG_H}",
            "scale_factors": {"x": _SCALE_X, "y": _SCALE_Y},
            "confidence_rule": "min(left_likelihood, right_likelihood) per keypoint",
        }

        np.savez_compressed(
            out_path,
            left_xy=left_xy,
            right_xy=right_xy,
            confidence=min_conf,
            labels=np.asarray(labels, dtype="<U32"),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            left_orig_w=np.asarray(_LP_RESIZE_W, dtype=np.float32),
            left_orig_h=np.asarray(_LP_RESIZE_H, dtype=np.float32),
            right_orig_w=np.asarray(_LP_RESIZE_W, dtype=np.float32),
            right_orig_h=np.asarray(_LP_RESIZE_H, dtype=np.float32),
        )

        rows_out.append(
            {
                "session_id": session_id,
                "frame_idx": int(frame_idx),
                "status": "written",
                "bundle_path": str(out_path),
                "n": int(n),
            }
        )

    print(f"[info] {session_id}: done, {len(rows_out)} pairs processed", flush=True)
    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lp-dir",
        type=Path,
        default=Path("/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/mvt_3d_loss_450_0"),
        help="Directory with predictions_L.csv, predictions_R.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root for output correspondence bundles",
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default="ear(base)(left),ear(tip)(left),ear(base)(right),ear(tip)(right),lowerlip,upperlip(left),upperlip(right),nose(bottom),nose(tip)",
        help="Comma-separated keypoint names to use for Kabsch",
    )
    parser.add_argument(
        "--min-likelihood",
        type=float,
        default=0.5,
        help="Minimum likelihood threshold (default: 0.5)",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Specific sessions to process (default: all 11 sessions)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bundles",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    lp_dir = args.lp_dir.expanduser().resolve()
    keypoints = [s.strip() for s in str(args.keypoints).split(",") if s.strip()]
    sessions = args.sessions or _ALL_SESSIONS

    print(f"LP dir: {lp_dir}")
    print(f"Output: {output_root}")
    print(f"Keypoints: {keypoints}")
    print(f"Min likelihood: {args.min_likelihood}")
    print(f"Sessions: {sessions}")
    print()

    all_rows: list[dict[str, Any]] = []
    for session_id in sessions:
        rows = _build_correspondences_for_session(
            lp_dir=lp_dir,
            session_id=session_id,
            keypoint_names=keypoints,
            min_likelihood=args.min_likelihood,
            output_root=output_root,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)

    summary = {
        "total": len(all_rows),
        "written": sum(1 for r in all_rows if r["status"] == "written"),
        "skipped": sum(1 for r in all_rows if r["status"] == "skipped_existing"),
        "failed": sum(1 for r in all_rows if r["status"] not in ("written", "skipped_existing")),
    }

    summary_path = output_root / "cheese3d_lp_correspondence_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"Details: {summary_path}")


if __name__ == "__main__":
    main()
