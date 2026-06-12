"""
Convert LP3D predictions CSV (L/R views) into SABLE litpose_matches.npz cache.

Each pair_{frame_idx:06d}/litpose_matches.npz contains:
    left_xy: (K, 2) float32  - original camera-pixel coordinates
    right_xy: (K, 2) float32
    confidence: (K,) float32
    labels: (K,) U32
    metadata_json: str
    left_orig_w/h, right_orig_w/h: scalar float32

The CSV predictions are already written in the original Cheese3D camera coordinate
system by run_lp3d_cheese3d_inference.py. Therefore this converter preserves those
pixel coordinates and only filters / repackages them for SABLE.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


FRAME_RE = re.compile(r"img(\d+)")
RIGID_HEAD_KEYPOINTS = [
    "nose(bottom)",
    "nose(tip)",
    "nose(top)",
    "eye(front)(left)",
    "eye(top)(left)",
    "eye(back)(left)",
    "eye(bottom)(left)",
    "eye(front)(right)",
    "eye(top)(right)",
    "eye(back)(right)",
    "eye(bottom)(right)",
    "ear(base)(left)",
    "ear(top)(left)",
    "ear(tip)(left)",
    "ear(bottom)(left)",
    "ear(base)(right)",
    "ear(top)(right)",
    "ear(tip)(right)",
    "ear(bottom)(right)",
]

EAR_NOSE_5_KEYPOINTS = [
    "nose(tip)",
    "ear(base)(left)",
    "ear(base)(right)",
    "ear(tip)(left)",
    "ear(tip)(right)",
]

HEAD_CORE_7_KEYPOINTS = [
    "nose(bottom)",
    "nose(tip)",
    "nose(top)",
    "ear(base)(left)",
    "ear(tip)(left)",
    "ear(base)(right)",
    "ear(tip)(right)",
]

HEAD_CORE_9_KEYPOINTS = [
    *HEAD_CORE_7_KEYPOINTS,
    "eye(front)(left)",
    "eye(front)(right)",
]

RIGID_EAR_NOSE_11_KEYPOINTS = [
    "nose(bottom)",
    "nose(tip)",
    "nose(top)",
    "ear(base)(left)",
    "ear(top)(left)",
    "ear(tip)(left)",
    "ear(bottom)(left)",
    "ear(base)(right)",
    "ear(top)(right)",
    "ear(tip)(right)",
    "ear(bottom)(right)",
]

RIGID_FACE_13_KEYPOINTS = [
    *RIGID_EAR_NOSE_11_KEYPOINTS,
    "eye(front)(left)",
    "eye(front)(right)",
]

RIGID_FACE_15_KEYPOINTS = [
    *RIGID_FACE_13_KEYPOINTS,
    "eye(back)(left)",
    "eye(back)(right)",
]

RIGID_NO_EYE_BOTTOM_17_KEYPOINTS = [
    k for k in RIGID_HEAD_KEYPOINTS
    if k not in {"eye(bottom)(left)", "eye(bottom)(right)"}
]

KEYPOINT_VARIANTS = {
    "all28": None,
    "rigidHead": RIGID_HEAD_KEYPOINTS,
    "earNose5": EAR_NOSE_5_KEYPOINTS,
    "headCore7": HEAD_CORE_7_KEYPOINTS,
    "headCore9": HEAD_CORE_9_KEYPOINTS,
    "rigidEarNose11": RIGID_EAR_NOSE_11_KEYPOINTS,
    "rigidFace13": RIGID_FACE_13_KEYPOINTS,
    "rigidFace15": RIGID_FACE_15_KEYPOINTS,
    "rigidNoEyeBottom17": RIGID_NO_EYE_BOTTOM_17_KEYPOINTS,
    "dynamicHighConf": None,
}


def parse_args():
    p = argparse.ArgumentParser(description="CSV predictions → litpose_matches.npz")
    p.add_argument("--pred_csv_dir", required=True, type=Path,
                   help="Dir containing predictions_L.csv / predictions_R.csv")
    p.add_argument("--session", required=True,
                   help="Session name (subfolder of pred_csv_dir)")
    p.add_argument("--output_root", required=True, type=Path,
                   help="Root of SABLE correspondence cache")
    p.add_argument("--min_confidence", type=float, default=0.8,
                   help="Filter keypoints below this confidence (default: 0.8)")
    p.add_argument("--orig_w", type=int, default=320,
                   help="Original image width (for metadata, default: 320)")
    p.add_argument("--orig_h", type=int, default=256,
                   help="Original image height (for metadata, default: 256)")
    p.add_argument("--keypoints", type=str, default=None,
                   help="Comma-separated keypoint names to include (default: all)")
    p.add_argument("--keypoint_variant", type=str, default="all28",
                   choices=list(KEYPOINT_VARIANTS.keys()),
                   help="Named keypoint subset / threshold preset")
    p.add_argument("--dynamic_min_confidence", type=float, default=0.95,
                   help="Confidence threshold used for dynamicHighConf variant")
    p.add_argument("--backend", type=str, default="lp3d_cheese3d_320x256",
                   help="Backend label in metadata")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files")
    return p.parse_args()


def _float_array(arr):
    return np.asarray(arr, dtype=np.float32)


def _metadata_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True)


def parse_frame_idx(frame_id: str) -> int:
    match = FRAME_RE.search(str(frame_id))
    if match is None:
        raise ValueError(f"Could not parse frame index from frame_id={frame_id!r}")
    return int(match.group(1))


def determine_keypoint_names(df: pd.DataFrame) -> list[str]:
    x_cols = [c for c in df.columns if c.endswith("_x")]
    return sorted(set(c[:-2] for c in x_cols))


def resolve_keypoint_selection(all_keypoints: list[str], args) -> tuple[list[str], float, str]:
    if args.keypoints:
        requested = [k.strip() for k in args.keypoints.split(",") if k.strip()]
        selected = [k for k in requested if k in all_keypoints]
        return selected, float(args.min_confidence), "explicit"

    if args.keypoint_variant in KEYPOINT_VARIANTS and KEYPOINT_VARIANTS[args.keypoint_variant] is not None:
        selected = [k for k in KEYPOINT_VARIANTS[args.keypoint_variant] if k in all_keypoints]
        return selected, float(args.min_confidence), args.keypoint_variant

    if args.keypoint_variant == "dynamicHighConf":
        return list(all_keypoints), float(args.dynamic_min_confidence), "dynamicHighConf"

    return list(all_keypoints), float(args.min_confidence), "all28"


def save_bundle(output_path: Path, left_xy, right_xy, confidence, labels, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_arr = np.asarray(list(labels), dtype=np.dtype("U64"))
    np.savez_compressed(
        output_path,
        left_xy=_float_array(left_xy),
        right_xy=_float_array(right_xy),
        confidence=np.asarray(confidence, dtype=np.float32),
        labels=label_arr,
        metadata_json=np.asarray(_metadata_json(metadata)),
        left_orig_w=np.float32(metadata["left_orig_w"]),
        left_orig_h=np.float32(metadata["left_orig_h"]),
        right_orig_w=np.float32(metadata["right_orig_w"]),
        right_orig_h=np.float32(metadata["right_orig_h"]),
    )


def build_metadata(session: str, frame_idx: int, labels: list[str], args, min_confidence_used: float, selection_mode: str) -> dict:
    return {
        "session_id": session,
        "pair_idx": frame_idx,
        "frame_idx": frame_idx,
        "keypoints": labels,
        "backend": args.backend,
        "split": "train",
        "confidence_rule": "min(left_likelihood, right_likelihood) per keypoint",
        "left_orig_w": args.orig_w,
        "left_orig_h": args.orig_h,
        "right_orig_w": args.orig_w,
        "right_orig_h": args.orig_h,
        "min_confidence_used": min_confidence_used,
        "keypoint_variant": args.keypoint_variant,
        "keypoint_selection_mode": selection_mode,
        "num_kp_raw": len(labels),
    }


def convert_session(csv_dir: Path, session: str, output_root: Path, args):
    sess_csv_dir = csv_dir / session
    csv_L = sess_csv_dir / "predictions_L.csv"
    csv_R = sess_csv_dir / "predictions_R.csv"

    if not csv_L.exists() or not csv_R.exists():
        raise FileNotFoundError(f"CSV files not found in {sess_csv_dir}")

    df_L = pd.read_csv(csv_L)
    df_R = pd.read_csv(csv_R)
    df_L["_frame_idx"] = df_L["frame_id"].map(parse_frame_idx)
    df_R["_frame_idx"] = df_R["frame_id"].map(parse_frame_idx)

    merged = pd.merge(df_L, df_R, on="_frame_idx", suffixes=("_L", "_R"))
    print(f"  Merged {len(merged)} frames (L={len(df_L)}, R={len(df_R)}, matched={len(merged)})")

    all_keypoints = determine_keypoint_names(df_L)
    kp_names, min_confidence_used, selection_mode = resolve_keypoint_selection(all_keypoints, args)
    print(f"  Keypoints ({len(kp_names)}/{len(all_keypoints)}): {kp_names[:5]}...")
    print(f"  Variant={args.keypoint_variant} selection_mode={selection_mode} min_confidence={min_confidence_used}")

    total_written = 0
    total_filtered = 0
    total_skipped = 0

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"{session}", unit="frame"):
        frame_idx = int(row["_frame_idx"])
        pair_dir = output_root / session / f"pair_{frame_idx:06d}"
        bundle_path = pair_dir / "litpose_matches.npz"

        if bundle_path.exists() and not args.force:
            total_skipped += 1
            continue

        left_list, right_list, conf_list, label_list = [], [], [], []

        for kp in kp_names:
            xl = row.get(f"{kp}_x_L")
            yl = row.get(f"{kp}_y_L")
            xr = row.get(f"{kp}_x_R")
            yr = row.get(f"{kp}_y_R")
            cl = row.get(f"{kp}_L")
            cr = row.get(f"{kp}_R")

            if any(pd.isna(v) for v in [xl, yl, xr, yr, cl, cr]):
                continue

            conf = min(float(cl), float(cr))
            if conf < min_confidence_used:
                total_filtered += 1
                continue

            left_list.append([float(xl), float(yl)])
            right_list.append([float(xr), float(yr)])
            conf_list.append(conf)
            label_list.append(kp)

        metadata = build_metadata(
            session=session,
            frame_idx=frame_idx,
            labels=label_list,
            args=args,
            min_confidence_used=min_confidence_used,
            selection_mode=selection_mode,
        )

        if not left_list:
            save_bundle(
                bundle_path,
                left_xy=np.zeros((0, 2), dtype=np.float32),
                right_xy=np.zeros((0, 2), dtype=np.float32),
                confidence=np.zeros(0, dtype=np.float32),
                labels=[],
                metadata=metadata,
            )
            total_written += 1
            continue

        save_bundle(
            bundle_path,
            left_xy=np.array(left_list, dtype=np.float32),
            right_xy=np.array(right_list, dtype=np.float32),
            confidence=np.array(conf_list, dtype=np.float32),
            labels=label_list,
            metadata=metadata,
        )
        total_written += 1

    print(
        f"  Written: {total_written} bundles | "
        f"Filtered (conf<{min_confidence_used}): {total_filtered} | Skipped: {total_skipped}"
    )


def main():
    args = parse_args()

    csv_dir = Path(args.pred_csv_dir)
    output_root = Path(args.output_root)

    session = args.session
    print(f"Converting session: {session}")
    print(f"  CSV dir: {csv_dir / session}")
    print(f"  Output root: {output_root}")

    convert_session(csv_dir, session, output_root, args)

    bundles = list((output_root / session).rglob("litpose_matches.npz"))
    non_empty = []
    for bundle in bundles:
        data = np.load(bundle, allow_pickle=True)
        if data["left_xy"].shape[0] > 0:
            non_empty.append(bundle)
    print("\nSummary:")
    print(f"  Total bundles: {len(bundles)}")
    print(f"  Non-empty bundles: {len(non_empty)}")
    if non_empty:
        sample = np.load(non_empty[0], allow_pickle=True)
        print(f"  Sample K: {sample['left_xy'].shape[0]}")
        print(f"  Sample labels: {list(sample['labels'][:5])}")
        print(f"  Sample mean conf: {sample['confidence'].mean():.4f}")
    print(f"  Output: {output_root / session}")


if __name__ == "__main__":
    main()
