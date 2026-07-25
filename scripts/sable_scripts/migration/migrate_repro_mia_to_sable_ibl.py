#!/usr/bin/env python3
"""Migrate repro_mia_4b00 to the SABLE IBL filesystem layout.

Source layout (repro_mia_4b00):
    {SRC}/processed/precached_video/{EID}/{train,val,test}/camera_{left,right}/img_frame_{N:06d}.png
    {SRC}/processed/precached_video/{EID}/pair_metadata.json
    {SRC}/litpose_correspondences/processed_correspondences/{EID}/pair_{N:06d}/litpose_matches.npz

Target layout (sable_ibl_4b00):
    {DST}/leftCamera.video/_iblrig_leftCamera.downsampled.{EID}/img{N:08d}.png
    {DST}/rightCamera.video/_iblrig_rightCamera.downsampled.{EID}/img{N:08d}.png
    {DST}/litpose_correspondences/processed_correspondences/{EID}/correspondences{pair_idx:08d}.npz

Notes:
- The target image filename uses metadata's `left_source_frame_index` (not the old N value).
- LitPose bundles are copied (not moved) so the original cache remains recoverable.
- depth cache is intentionally NOT migrated; online VDA computes depth internally.
- Output:
    migration_manifest.json: rename mapping + sha256 spot-checks for 5 random images
                             and 5 random bundles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

EID = "4b00df29-3769-43be-bb40-128b1cba6d35"
SRC_DEFAULT = Path("/data/jqh/Datasets/E-RayZer-private/data/repro_mia_4b00")
DST_DEFAULT = Path("/data/jqh/Datasets/beast3d-data/sable_ibl_4b00")
N_SPOT_CHECK = 5


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def migrate_images(
    src_root: Path,
    dst_root: Path,
    eid: str,
    pairs: list[dict],
) -> tuple[list[dict], list[str]]:
    """Copy each (left, right) pair image into the SABLE IBL flat layout.

    Returns:
        image_records: per-pair list of {pair_idx, left_dst, right_dst,
                       left_source_frame_index, split, left_src_sha256}.
        skipped: list of pair_idx whose source image was not found.
    """
    left_dst_root = dst_root / "leftCamera.video" / f"_iblrig_leftCamera.downsampled.{eid}"
    right_dst_root = dst_root / "rightCamera.video" / f"_iblrig_rightCamera.downsampled.{eid}"
    left_dst_root.mkdir(parents=True, exist_ok=True)
    right_dst_root.mkdir(parents=True, exist_ok=True)

    # pair_metadata.json stores paths relative to processed/precached_video/{eid}/
    # (i.e. "train/camera_left/img_frame_000000.png"), but our SRC root is the
    # top-level repro_mia_4b00/ directory. Build the actual prefix here.
    precache_root = src_root / "processed" / "precached_video" / eid

    image_records: list[dict] = []
    skipped: list[str] = []
    for p in pairs:
        pair_idx = int(p["pair_idx"])
        split = p.get("split", "train")
        src_left_rel = p["left_path"]
        src_right_rel = p["right_path"]
        src_left = precache_root / src_left_rel
        src_right = precache_root / src_right_rel
        if not src_left.exists() or not src_right.exists():
            skipped.append(str(pair_idx))
            continue

        src_idx = int(p["left_source_frame_index"])
        dst_left = left_dst_root / f"img{src_idx:08d}.png"
        dst_right = right_dst_root / f"img{src_idx:08d}.png"

        # copy if missing OR if source is newer; copy2 preserves mtime.
        if not dst_left.exists():
            shutil.copy2(src_left, dst_left)
        if not dst_right.exists():
            shutil.copy2(src_right, dst_right)

        image_records.append(
            {
                "pair_idx": pair_idx,
                "split": split,
                "left_source_frame_index": src_idx,
                "right_source_frame_index": int(p["right_source_frame_index"]),
                "src_left": str(src_left),
                "src_right": str(src_right),
                "dst_left": str(dst_left),
                "dst_right": str(dst_right),
            }
        )
    return image_records, skipped


def migrate_correspondences(
    src_root: Path,
    dst_root: Path,
    eid: str,
) -> tuple[int, list[str]]:
    """Copy each pair's LitPose bundle.

    The source layout is `{eid}/pair_NNNNNN/litpose_matches.npz` and the
    destination is `{eid}/correspondences{pair_idx:08d}.npz`.

    Returns:
        bundle_count: number of bundles copied.
        skipped: list of pair_idx whose source bundle was missing.
    """
    src_lp_root = src_root / "litpose_correspondences" / "processed_correspondences" / eid
    dst_lp_root = dst_root / "litpose_correspondences" / "processed_correspondences" / eid
    dst_lp_root.mkdir(parents=True, exist_ok=True)

    bundle_count = 0
    skipped: list[str] = []
    if not src_lp_root.is_dir():
        return bundle_count, ["<src_lp_root_missing>"]

    for old_dir in sorted(src_lp_root.glob("pair_*")):
        pair_idx_str = old_dir.name.removeprefix("pair_")
        try:
            pair_idx = int(pair_idx_str)
        except ValueError:
            skipped.append(old_dir.name)
            continue
        src_npz = old_dir / "litpose_matches.npz"
        if not src_npz.exists():
            skipped.append(str(pair_idx))
            continue
        dst_npz = dst_lp_root / f"correspondences{pair_idx:08d}.npz"
        if not dst_npz.exists():
            shutil.copy2(src_npz, dst_npz)
        bundle_count += 1
    return bundle_count, skipped


def verify_spot_checks(
    src_root: Path,
    image_records: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Spot-check `n` random pairs: source vs destination sha256 (left image)."""
    if not image_records:
        return []
    sample = rng.sample(image_records, min(n, len(image_records)))
    spot_checks: list[dict] = []
    for rec in sample:
        src_sha = sha256_of(Path(rec["src_left"]))
        dst_sha = sha256_of(Path(rec["dst_left"]))
        spot_checks.append(
            {
                "pair_idx": rec["pair_idx"],
                "split": rec["split"],
                "left_source_frame_index": rec["left_source_frame_index"],
                "src_left": rec["src_left"],
                "dst_left": rec["dst_left"],
                "src_sha256": src_sha,
                "dst_sha256": dst_sha,
                "match": src_sha == dst_sha,
            }
        )
    return spot_checks


def verify_correspondence_spot_checks(
    src_root: Path,
    dst_root: Path,
    eid: str,
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Spot-check `n` random correspondence bundles for valid keys/shapes."""
    src_lp_root = src_root / "litpose_correspondences" / "processed_correspondences" / eid
    dst_lp_root = dst_root / "litpose_correspondences" / "processed_correspondences" / eid

    pair_dirs = sorted(src_lp_root.glob("pair_*"))
    if not pair_dirs:
        return []
    sample = rng.sample(pair_dirs, min(n, len(pair_dirs)))
    spot_checks: list[dict] = []
    for old_dir in sample:
        pair_idx = int(old_dir.name.removeprefix("pair_"))
        src_npz = old_dir / "litpose_matches.npz"
        dst_npz = dst_lp_root / f"correspondences{pair_idx:08d}.npz"
        if not dst_npz.exists():
            spot_checks.append(
                {
                    "pair_idx": pair_idx,
                    "found": False,
                }
            )
            continue
        try:
            import numpy as np  # local import to keep CLI fast

            payload = np.load(str(dst_npz), allow_pickle=True)
            left_xy = np.asarray(payload["left_xy"], dtype=np.float32)
            right_xy = np.asarray(payload["right_xy"], dtype=np.float32)
            confidence = np.asarray(payload["confidence"], dtype=np.float32)
            shape_ok = (
                left_xy.shape == right_xy.shape
                and left_xy.ndim == 2
                and left_xy.shape[1] == 2
                and confidence.shape == (left_xy.shape[0],)
            )
            finite_ok = bool(
                (left_xy.size and np.isfinite(left_xy).all())
                and (right_xy.size and np.isfinite(right_xy).all())
                and (confidence.size and np.isfinite(confidence).all())
            )
            spot_checks.append(
                {
                    "pair_idx": pair_idx,
                    "found": True,
                    "src": str(src_npz),
                    "dst": str(dst_npz),
                    "shape": list(left_xy.shape),
                    "shape_ok": bool(shape_ok),
                    "finite": bool(finite_ok),
                }
            )
        except Exception as exc:
            spot_checks.append(
                {
                    "pair_idx": pair_idx,
                    "found": True,
                    "error": str(exc),
                }
            )
    return spot_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, default=SRC_DEFAULT, help="Source root (repro_mia_4b00)."
    )
    parser.add_argument(
        "--dst", type=Path, default=DST_DEFAULT, help="Destination root (sable_ibl_4b00)."
    )
    parser.add_argument("--eid", type=str, default=EID, help="Session ID.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for spot-checks.")
    parser.add_argument(
        "--n-spot-checks", type=int, default=N_SPOT_CHECK, help="Number of spot-check pairs."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_root = args.src
    dst_root = args.dst
    eid = args.eid

    meta_path = src_root / "processed" / "precached_video" / eid / "pair_metadata.json"
    if not meta_path.exists():
        print(f"ERROR: pair_metadata.json not found at {meta_path}", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text())
    pairs = meta["pairs"]
    print(f"Found {len(pairs)} pairs in metadata.")

    print("Migrating images...")
    image_records, skipped_imgs = migrate_images(src_root, dst_root, eid, pairs)
    print(f"  images: {len(image_records)} copied, {len(skipped_imgs)} skipped")
    if skipped_imgs:
        print(f"  WARNING: skipped image pair_idx={skipped_imgs[:10]}")

    print("Migrating LitPose bundles...")
    bundle_count, skipped_bundles = migrate_correspondences(src_root, dst_root, eid)
    print(f"  bundles: {bundle_count} copied, {len(skipped_bundles)} skipped")

    rng = random.Random(args.seed)
    print(f"Verifying {args.n_spot_checks} image spot-checks (sha256)...")
    img_spot_checks = verify_spot_checks(src_root, image_records, args.n_spot_checks, rng)
    bad = [c for c in img_spot_checks if not c.get("match", False)]
    if bad:
        print(f"  ERROR: {len(bad)}/{len(img_spot_checks)} image sha256 mismatches!")
        for c in bad:
            print(f"    pair_idx={c['pair_idx']}")
        return 2
    print(f"  all {len(img_spot_checks)} image spot-checks PASS")

    print(f"Verifying {args.n_spot_checks} correspondence bundle spot-checks...")
    bundle_spot_checks = verify_correspondence_spot_checks(
        src_root, dst_root, eid, args.n_spot_checks, rng
    )
    bad_b = [
        c
        for c in bundle_spot_checks
        if not (c.get("found") and c.get("shape_ok") and c.get("finite"))
    ]
    if bad_b:
        print(f"  ERROR: {len(bad_b)}/{len(bundle_spot_checks)} correspondence issues!")
        for c in bad_b:
            print(f"    pair_idx={c.get('pair_idx')}: {c}")
        return 3
    print(f"  all {len(bundle_spot_checks)} bundle spot-checks PASS")

    manifest = {
        "eid": eid,
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "num_pairs": len(pairs),
        "num_images_migrated": len(image_records),
        "num_bundles_migrated": bundle_count,
        "skipped_images": skipped_imgs,
        "skipped_bundles": skipped_bundles,
        "image_spot_checks": img_spot_checks,
        "bundle_spot_checks": bundle_spot_checks,
        "depth_cache_migrated": False,
        "note": (
            "depth cache is intentionally NOT migrated; online VDA computes "
            "depth inside Sable.forward. Source vda_depth_cache/ is left "
            "untouched on disk."
        ),
    }
    manifest_path = dst_root / "migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
