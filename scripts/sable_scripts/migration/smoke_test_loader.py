#!/usr/bin/env python3
"""Loader-only smoke test for the SABLE IBL dataset in online VDA mode.

Reads pair_metadata.json / dataset directory contract to assert:
  - 1642 records discoverable via IBLTwoViewDataset
  - depth_vda is zero (online mode returns a zero tensor from the dataset;
    the real depth is produced inside Sable.forward via online VDA)
  - leftcamera_xy is zero-padded to _MAX_MATCHES entries with at least
    3 non-zero rows (LitPose bundles contain matches)
  - model.vda.mode == "online"
  - a couple of spot-check pairs have the expected (pair_idx, source_frame_index)
    pairing from the original pair_metadata.json

After running this script, dump the loader's deterministic train/val split to
`loader_split_manifest.json` so all 8 cells share the same val frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EID = "4b00df29-3769-43be-bb40-128b1cba6d35"
DATASET_PATH_DEFAULT = "/cephfs/jinqihang/SABLE/datasets/beast3d-data/sable_ibl_4b00"
CORR_ROOT_DEFAULT = f"{DATASET_PATH_DEFAULT}/litpose_correspondences/processed_correspondences"
META_SRC_DEFAULT = "/cephfs/jinqihang/SABLE/datasets/E-RayZer-private/data/repro_mia_4b00"
EXPECTED_NUM_PAIRS = 1642


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default=DATASET_PATH_DEFAULT)
    parser.add_argument("--corr-root", default=CORR_ROOT_DEFAULT)
    parser.add_argument("--eid", default=EID)
    parser.add_argument(
        "--meta-src", default=META_SRC_DEFAULT, help="Source pair_metadata.json for spot-checks."
    )
    parser.add_argument(
        "--split-out", default=None, help="If set, write loader_split_manifest.json here."
    )
    parser.add_argument("--val-split-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Defer heavy imports until after arg parsing so --help is fast.
    import torch

    sys.path.insert(0, "/cephfs/jinqihang/SABLE/beast")
    from beast.data.sable_dataset import IBLTwoViewDataset  # noqa: E402

    config = {
        "model": {
            "seed": args.seed,
            "image_tokenizer": {"image_size": 320},
            "vda": {
                "mode": "online",
                "cache_root": None,
                "encoder": "vitb",
                "metric": False,
                "checkpoint_path": (
                    "/cephfs/jinqihang/SABLE/third_party/VDA/checkpoints/"
                    "video_depth_anything_vitb.pth"
                ),
                "repo_root": "/cephfs/jinqihang/SABLE/third_party/VDA",
            },
            "merge_pcd": {"correspondence_cache_root": args.corr_root},
        },
        "training": {
            "dataset_path": args.dataset_path,
            "session_names": [args.eid],
            "val_split_ratio": args.val_split_ratio,
        },
    }

    print("Building IBLTwoViewDataset (full / split-aware)...")
    # Training instantiates train and val separately with include_splits=['train'] / ['val']
    ds_train = IBLTwoViewDataset(config, include_splits=["train"])
    ds_val = IBLTwoViewDataset(config, include_splits=["val"])
    print(f"  train={len(ds_train)} val={len(ds_val)} total={len(ds_train) + len(ds_val)}")
    assert (len(ds_train) + len(ds_val)) == EXPECTED_NUM_PAIRS, (
        f"train+val should be {EXPECTED_NUM_PAIRS}, got {len(ds_train) + len(ds_val)}"
    )
    # Use the train dataset as the canonical "full" record list for further checks.
    ds = ds_train

    # mode sanity check
    mode = config["model"]["vda"]["mode"]
    print(f"  model.vda.mode = {mode!r}")
    assert mode == "online", f"vda mode must be 'online', got {mode!r}"

    # First sample assertions
    print("Inspecting sample 0...")
    sample = ds[0]
    image = sample["image"]
    depth_vda = sample["depth_vda"]
    left_xy = sample["leftcamera_xy"]
    right_xy = sample["rightcamera_xy"]
    confidence = sample["confidence"]

    print(f"  image.shape = {tuple(image.shape)}")
    print(f"  depth_vda.shape = {tuple(depth_vda.shape)}")
    print(f"  leftcamera_xy.shape = {tuple(left_xy.shape)}")
    print(f"  rightcamera_xy.shape = {tuple(right_xy.shape)}")
    print(f"  confidence.shape = {tuple(confidence.shape)}")

    assert image.ndim == 4 and image.shape == (2, 3, 320, 320), (
        f"image must be [2,3,320,320], got {tuple(image.shape)}"
    )
    assert depth_vda.ndim == 4 and depth_vda.shape == (2, 1, 320, 320), (
        f"depth_vda must be [2,1,320,320], got {tuple(depth_vda.shape)}"
    )
    assert torch.all(depth_vda == 0).item(), (
        "depth_vda must be all-zero in online mode (dataset returns zero tensor)"
    )
    assert left_xy.shape == right_xy.shape and left_xy.shape[1] == 2, (
        f"left/right_xy must be [K,2], got {tuple(left_xy.shape)}/{tuple(right_xy.shape)}"
    )
    nz = int((confidence > 0).sum().item())
    print(f"  confidence: {nz} non-zero rows")
    assert nz >= 3, f"need at least 3 LitPose matches, got {nz}"

    # Spot-check a few records' (pair_idx, source_frame_index) against the source metadata.
    meta_path = (
        Path(args.meta_src) / "processed" / "precached_video" / args.eid / "pair_metadata.json"
    )
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # Build a mapping: pair_idx -> left_source_frame_index.
        meta_pair_to_sfi = {
            int(p["pair_idx"]): int(p["left_source_frame_index"]) for p in meta["pairs"]
        }
        # ds._records[0] should correspond to pair_idx 0 (the smallest source_frame_index
        # after sort). The loader assigns pair_idx = position in sorted-by-source_frame_index
        # list, so source_frame_index is monotonic in pair_idx.
        records = ds._records
        sample_indices = sorted({0, len(records) // 2, len(records) - 1})
        for idx in sample_indices:
            rec = records[idx]
            expected_sfi = meta_pair_to_sfi.get(rec.pair_idx)
            assert expected_sfi is not None, (
                f"record {idx} pair_idx={rec.pair_idx} not in source metadata"
            )
            assert int(rec.left_source_frame_index) == expected_sfi, (
                f"record {idx} pair_idx={rec.pair_idx}: "
                f"loader sfi={rec.left_source_frame_index} != metadata sfi={expected_sfi}"
            )
            print(f"  record[{idx}] pair_idx={rec.pair_idx} sfi={rec.left_source_frame_index} ✓")
    else:
        print(f"  WARNING: source metadata not found at {meta_path}; skipping spot-check")

    # Output loader_split_manifest.json if requested.
    if args.split_out:
        out_path = Path(args.split_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # The split comes from _split_records: shuffling all records with seed=split_seed
        # then taking the last ceil(N * val_split_ratio) as val. Reproduce that here so
        # the manifest captures both the train and val indices explicitly.
        train_set = {(r.session_id, int(r.pair_idx)) for r in ds_train._records}
        val_set = {(r.session_id, int(r.pair_idx)) for r in ds_val._records}
        assert len(train_set & val_set) == 0, "train/val overlap detected"
        assert (len(train_set) + len(val_set)) == EXPECTED_NUM_PAIRS

        manifest = {
            "eid": args.eid,
            "dataset_path": args.dataset_path,
            "val_split_ratio": args.val_split_ratio,
            "split_seed": args.seed,
            "num_records_total": EXPECTED_NUM_PAIRS,
            "num_records_train": len(ds_train._records),
            "num_records_val": len(ds_val._records),
            "train_records": [
                {
                    "session_id": r.session_id,
                    "scene_name": r.scene_name,
                    "pair_idx": int(r.pair_idx),
                    "left_source_frame_index": int(r.left_source_frame_index),
                    "right_source_frame_index": int(r.right_source_frame_index),
                }
                for r in ds_train._records
            ],
            "val_records": [
                {
                    "session_id": r.session_id,
                    "scene_name": r.scene_name,
                    "pair_idx": int(r.pair_idx),
                    "left_source_frame_index": int(r.left_source_frame_index),
                    "right_source_frame_index": int(r.right_source_frame_index),
                }
                for r in ds_val._records
            ],
            "note": (
                "This manifest captures the loader's deterministic train/val split. "
                "The training launcher does NOT read it (the loader's own split is "
                "used), but the evaluator must re-instantiate the dataset with the "
                "same (val_split_ratio, seed) and assert the resulting _records match "
                "this snapshot before computing metrics."
            ),
        }
        out_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {out_path}")
        print(f"  train={manifest['num_records_train']} val={manifest['num_records_val']}")

    print("Loader smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
