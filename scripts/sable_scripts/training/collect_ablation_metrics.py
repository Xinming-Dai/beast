#!/usr/bin/env python3
"""Collect held-out evaluation metrics for all 8 cells of the loss-weighting
ablation.

This is Phase 7 of the loss-weighting ablation plan. It computes per-cell
metrics on all 164 pairs in the validation split frozen by
loader_split_manifest.json, using a unified evaluation protocol:

- All loss weights are forced to 1.0 during evaluation so the raw metrics
  (PSNR, SSIM, L_recon, L_percept, L_geom) are directly comparable across cells.
- The training weights are reported separately in the "weighted" column.
- The periodic step-10000 checkpoint is resolved exactly; best checkpoints and
  arbitrary fallbacks are rejected.
- Four manifest-pinned frames are used only for the qualitative grid. The
  loader's full deterministic validation split is reproduced and validated
  before any metric is computed.

Outputs (under <out_dir>/):
- metrics.csv                   per-cell raw metrics + training weights
- metrics.json                  same data in JSON
- vis/<cell>/<pair_idx>.png     GT vs predicted render per frame
- vis/grid_4x9.png              4 val frames x 9 cols (GT + 8 cells)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = "/cephfs/jinqihang/SABLE/beast"
PROJECT_ROOT = "/cephfs/jinqihang/SABLE"

# Frozen contract (same as run_one_cell.sh)
EID = "4b00df29-3769-43be-bb40-128b1cba6d35"
DATASET_PATH = "/cephfs/jinqihang/SABLE/datasets/beast3d-data/sable_ibl_4b00"
CORR_ROOT = f"{DATASET_PATH}/litpose_correspondences/processed_correspondences"
VDA_CKPT = "/cephfs/jinqihang/SABLE/third_party/VDA/checkpoints/video_depth_anything_vitb.pth"
VDA_REPO_ROOT = "/cephfs/jinqihang/SABLE/third_party/VDA"
VGG_SRC = "/cephfs/jinqihang/SABLE/ckpt/imagenet-vgg-verydeep-19.mat"
BASE_CONFIG = f"{REPO_ROOT}/configs/sable/sable_ibl3d.yaml"

# 8 cells (name, train_weights)
CELLS = [
    ("default", 1.0, 0.3, 1.0),
    ("no-percept", 1.0, 0.0, 1.0),
    ("low-percept", 1.0, 0.1, 1.0),
    ("high-percept", 1.0, 1.0, 1.0),
    ("low-recon", 0.5, 0.3, 1.0),
    ("high-recon", 2.0, 0.3, 1.0),
    ("no-geom", 1.0, 0.3, 0.0),
    ("high-geom", 1.0, 0.3, 2.0),
]

# Four pre-training qualitative frames: evenly spaced positions in the sorted
# 164-pair frozen validation manifest. They are used only for the visual grid.
EVAL_PAIR_IDX = [1, 501, 1064, 1599]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/cephfs/jinqihang/SABLE/outputs/loss_weighting")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/cephfs/jinqihang/SABLE/outputs/loss_weighting/eval")
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=Path(f"{DATASET_PATH}/loader_split_manifest.json")
    )
    parser.add_argument(
        "--ckpt-step", type=int, default=10000, help="Exact periodic checkpoint step to evaluate."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=4,
        help="Number of fixed val frames to use for the grid.",
    )
    parser.add_argument(
        "--max-eval-frames",
        type=int,
        default=0,
        help="Limit quantitative frames for evaluator smoke tests; 0 means all.",
    )
    parser.add_argument(
        "--include-lpips",
        action="store_true",
        help="Also compute LPIPS; requires the optional lpips package.",
    )
    parser.add_argument("--no-render-visuals", action="store_true")
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help="Comma-separated cell names to evaluate (default: all 8).",
    )
    return parser.parse_args()


def load_config_overrides() -> dict:
    """Build the training config with the frozen online-VDA contract."""
    import yaml  # noqa: PLC0415

    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)

    config["model"]["vda"] = {
        "enabled": True,
        "mode": "online",
        "cache_root": None,
        "encoder": "vitb",
        "metric": False,
        "checkpoint_path": VDA_CKPT,
        "repo_root": VDA_REPO_ROOT,
        "debug_save": False,
        "debug_dir": None,
    }
    config["model"]["merge_pcd"]["correspondence_cache_root"] = CORR_ROOT
    config["training"]["dataset_path"] = DATASET_PATH
    config["training"]["session_names"] = [EID]
    config["training"]["reset_training_state"] = True
    config["training"]["save_visuals"] = False
    config["training"]["save_pointclouds"] = False
    config["training"]["save_camera_pointcloud_scene"] = False
    config["training"]["num_workers"] = 0
    return config


def validate_split_against_manifest(
    ds_val, manifest_path: Path, eval_pair_idx: list[int]
) -> list[int]:
    val_records = ds_val._records
    val_set = {(r.session_id, int(r.pair_idx)) for r in val_records}

    if not manifest_path.exists():
        print(f"WARNING: split manifest not found at {manifest_path}; skipping manifest check")
        return [p for p in eval_pair_idx if (EID, p) in val_set]

    manifest = json.loads(manifest_path.read_text())
    expected_val = {(r["session_id"], int(r["pair_idx"])) for r in manifest["val_records"]}
    if val_set != expected_val:
        only_in_runtime = val_set - expected_val
        only_in_manifest = expected_val - val_set
        runtime_sample = list(only_in_runtime)[:5]
        manifest_sample = list(only_in_manifest)[:5]
        print(
            f"ERROR: loader val split does not match manifest!\n"
            f"  only in runtime: {len(only_in_runtime)} pairs (first 5: {runtime_sample})\n"
            f"  only in manifest: {len(only_in_manifest)} pairs (first 5: {manifest_sample})"
        )
        raise RuntimeError("val split mismatch with manifest; aborting evaluation")

    chosen = []
    for p in eval_pair_idx:
        if (EID, p) not in val_set:
            raise RuntimeError(f"eval pair_idx {p} not in val set")
        chosen.append(p)
    return chosen


def build_val_dataset(config: dict):
    from beast.data.sable_dataset import IBLTwoViewDataset  # noqa: PLC0415

    return IBLTwoViewDataset(config, include_splits=["val"])


def build_single_batch(ds_val, pair_idx: int, micro_batch: int = 1):
    """Build a micro-batch collated tensor containing the given pair_idx."""
    from beast.data.sable_dataset import collate_with_correspondence_padding  # noqa: PLC0415

    target_idx = None
    for idx, rec in enumerate(ds_val._records):
        if int(rec.pair_idx) == pair_idx:
            target_idx = idx
            break
    if target_idx is None:
        raise RuntimeError(f"pair_idx {pair_idx} not in val records")
    n = len(ds_val._records)
    indices = [(target_idx + i) % n for i in range(micro_batch)]
    samples = [ds_val[i] for i in indices]
    return collate_with_correspondence_padding(samples)


def resolve_periodic_checkpoint(cell_dir: Path, step: int) -> Path:
    """Return the unique periodic checkpoint for ``step``.

    Validation-best checkpoints are intentionally excluded because selecting
    them would use a different criterion for each loss-weight cell.
    """
    all_ckpts = sorted(cell_dir.rglob("*.ckpt"))
    periodic = [path for path in all_ckpts if "best" not in path.name.lower()]
    step_token = re.compile(rf"(?:^|[^0-9])0*{int(step)}(?:[^0-9]|$)")
    matches = [path for path in periodic if step_token.search(path.stem)]
    if len(matches) != 1:
        found = "\n    ".join(str(path) for path in all_ckpts) or "<none>"
        raise RuntimeError(
            f"expected exactly one non-best step-{step} checkpoint under {cell_dir}, "
            f"found {len(matches)}; all checkpoints:\n    {found}"
        )
    return matches[0]


def evaluate_cell(
    cell_dir: Path,
    cell_name: str,
    l2_w: float,
    p_w: float,
    g_w: float,
    config: dict,
    ds_val,
    metric_pair_idx: list[int],
    visual_pair_idx: list[int],
    out_dir: Path,
    device: str,
    ckpt_step: int,
    no_render_visuals: bool,
    include_lpips: bool,
) -> dict:
    """Load one cell's fixed-step checkpoint and evaluate the frozen split."""
    import torch  # noqa: PLC0415

    from beast.models.sable import Sable  # noqa: PLC0415
    from beast.sable_encoding_decoding.render.metrics import (  # noqa: PLC0415
        _psnr_per_image,
        _ssim_per_image,
    )

    ckpt_path = resolve_periodic_checkpoint(cell_dir, ckpt_step)
    print(f"  loading {ckpt_path}")

    # Force eval weights to 1.0 for raw, unweighted metrics.
    eval_config = {**config, "training": {**config["training"]}}
    eval_config["training"]["l2_loss_weight"] = 1.0
    eval_config["training"]["perceptual_loss_weight"] = 1.0
    eval_config["training"]["gs_reg_loss_weight"] = 1.0
    eval_config["training"]["lpips_loss_weight"] = 1.0 if include_lpips else 0.0

    model = Sable.load_from_checkpoint(
        str(ckpt_path),
        config=eval_config,
        strict=False,
    )
    model = model.to(device)
    model.eval()

    metrics = {
        "cell": cell_name,
        "ckpt_path": str(ckpt_path),
        "train_l2_loss_weight": l2_w,
        "train_perceptual_loss_weight": p_w,
        "train_gs_reg_loss_weight": g_w,
        "status": "ok",
        "checkpoint_step": ckpt_step,
        "num_eval_pairs": len(metric_pair_idx),
        "lpips_status": "computed" if include_lpips else "not_computed",
    }
    per_frame = {}
    visual_set = set(visual_pair_idx)

    for pair_idx in metric_pair_idx:
        batch = build_single_batch(ds_val, pair_idx, micro_batch=1)
        batch = {
            key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()
        }
        with torch.no_grad():
            out = model.get_model_outputs(batch)
            result = model.loss_computer(
                out["render"],
                out["target_image"],
                out["xyz_norm"],
                out["xyz_init_norm"],
                out.get("pixel_mask"),
                out.get("gaussian_mask"),
            )

        rendering = out["render"]
        target = out["target_image"]
        psnr_block = _psnr_per_image(rendering, target).detach().cpu().numpy()
        ssim_block = _ssim_per_image(rendering, target).detach().cpu().numpy()
        values = {
            "l2_loss": float(result.l2_loss.detach()),
            "psnr": float(psnr_block.mean()),
            "ssim": float(ssim_block.mean()),
            "perceptual_loss": float(result.perceptual_loss.detach()),
            "gs_reg_loss": float(result.gs_reg_loss.detach()),
        }
        if include_lpips:
            values["lpips_loss"] = float(result.lpips_loss.detach())
        if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
            raise RuntimeError(
                f"non-finite metric for cell={cell_name}, pair_idx={pair_idx}: {values}"
            )
        values["weighted_loss"] = (
            l2_w * values["l2_loss"]
            + p_w * values["perceptual_loss"]
            + g_w * values["gs_reg_loss"]
        )
        per_frame[int(pair_idx)] = values

        if not no_render_visuals and pair_idx in visual_set:
            save_compare_visual(
                rendering,
                target,
                pair_idx=int(pair_idx),
                out_dir=out_dir / "vis" / cell_name,
            )

    if len(per_frame) != len(metric_pair_idx):
        raise RuntimeError(
            f"incomplete evaluation for {cell_name}: {len(per_frame)}/{len(metric_pair_idx)} pairs"
        )
    good = list(per_frame.values())
    metrics["avg_l2_loss"] = float(sum(m["l2_loss"] for m in good) / len(good))
    metrics["avg_psnr"] = float(sum(m["psnr"] for m in good) / len(good))
    metrics["avg_ssim"] = float(sum(m["ssim"] for m in good) / len(good))
    metrics["avg_perceptual_loss"] = float(sum(m["perceptual_loss"] for m in good) / len(good))
    metrics["avg_gs_reg_loss"] = float(sum(m["gs_reg_loss"] for m in good) / len(good))
    if include_lpips:
        metrics["avg_lpips_loss"] = float(sum(m["lpips_loss"] for m in good) / len(good))
    metrics["avg_weighted_loss"] = float(sum(m["weighted_loss"] for m in good) / len(good))
    metrics["per_frame"] = per_frame
    return metrics


def save_compare_visual(rendering, target, pair_idx: int, out_dir: Path) -> None:
    import torch  # noqa: PLC0415
    from torchvision.utils import save_image  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    r = rendering.detach().float().cpu().clamp(0.0, 1.0)
    t = target.detach().float().cpu().clamp(0.0, 1.0)
    r = r[0, 0]
    t = t[0, 0]
    side = torch.cat([t, r], dim=2)
    save_image(side, str(out_dir / f"pair_{pair_idx:08d}.png"))


def make_grid_visual(out_dir: Path, eval_pair_idx: list[int], cells: list[str]) -> None:
    """Stitch per-cell per-frame PNGs into a 4x9 grid (4 frames x 9 cols = GT+8cells)."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("PIL not available; skipping grid_4x9.png")
        return
    rows = []
    for pair_idx in eval_pair_idx:
        cols = []
        # GT column: load from cell_default/pair_*.png left half.
        gt_path = out_dir / "vis" / "default" / f"pair_{pair_idx:08d}.png"
        if not gt_path.exists():
            cols.append(Image.new("RGB", (320, 320), (0, 0, 0)))
        else:
            img = Image.open(gt_path).convert("RGB")
            w, h = img.size
            cols.append(img.crop((0, 0, w // 2, h)))
        for cell in cells:
            p = out_dir / "vis" / cell / f"pair_{pair_idx:08d}.png"
            if not p.exists():
                cols.append(Image.new("RGB", (320, 320), (0, 0, 0)))
                continue
            img = Image.open(p).convert("RGB")
            w, h = img.size
            cols.append(img.crop((w // 2, 0, w, h)))
        row = Image.new("RGB", (sum(c.size[0] for c in cols), cols[0].size[1]))
        x = 0
        for c in cols:
            row.paste(c, (x, 0))
            x += c.size[0]
        rows.append(row)
    if not rows:
        return
    grid = Image.new("RGB", (rows[0].size[0], sum(r.size[1] for r in rows)))
    y = 0
    for r in rows:
        grid.paste(r, (0, y))
        y += r.size[1]
    grid.save(out_dir / "grid_4x9.png")
    print(f"Saved {out_dir / 'grid_4x9.png'}")


def main() -> int:
    args = parse_args()
    sys.path.insert(0, REPO_ROOT)

    # Match run_one_cell.sh so torch extension cache keys do not drift between
    # training and evaluation.
    if Path("/usr/bin/gcc").is_file():
        os.environ.setdefault("CC", "/usr/bin/gcc")
        os.environ.setdefault("CXX", "/usr/bin/g++")
    if "10.0" in os.environ.get("TORCH_CUDA_ARCH_LIST", ""):
        os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    args.root = args.root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.split_manifest = args.split_manifest.expanduser().resolve()
    if args.include_lpips and importlib.util.find_spec("lpips") is None:
        raise RuntimeError(
            "--include-lpips was requested, but the optional lpips package is not installed"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    metric_dir = args.out / "metric_checkpoint"
    metric_dir.mkdir(parents=True, exist_ok=True)
    vgg_link = metric_dir / "imagenet-vgg-verydeep-19.mat"
    if (vgg_link.exists() or vgg_link.is_symlink()) and vgg_link.resolve() != Path(
        VGG_SRC
    ).resolve():
        raise RuntimeError(f"unexpected VGG file at {vgg_link}: resolves to {vgg_link.resolve()}")
    if not vgg_link.exists() and not vgg_link.is_symlink():
        vgg_link.symlink_to(VGG_SRC)
    os.chdir(args.out)

    config = load_config_overrides()
    print("Building val dataset...")
    ds_val = build_val_dataset(config)
    print(f"  val size = {len(ds_val)}")

    visual_pair_idx = validate_split_against_manifest(ds_val, args.split_manifest, EVAL_PAIR_IDX)
    visual_pair_idx = visual_pair_idx[: args.sample_frames]
    metric_pair_idx = [int(record.pair_idx) for record in ds_val._records]
    if args.max_eval_frames > 0:
        metric_pair_idx = metric_pair_idx[: args.max_eval_frames]
    print(f"  quantitative pairs: {len(metric_pair_idx)}")
    print(f"  qualitative pair_idx (manifest-validated): {visual_pair_idx}")

    all_metrics = []
    cells_to_run = [c for c in CELLS if (not args.cells or c[0] in args.cells.split(","))]
    for name, l2_w, p_w, g_w in cells_to_run:
        cell_dir = args.root / f"cell_{name}"
        if not cell_dir.exists():
            raise FileNotFoundError(f"cell dir missing: {cell_dir}")
        print(f"\n=== Cell: {name} (l2={l2_w} p={p_w} g={g_w}) ===")
        m = evaluate_cell(
            cell_dir=cell_dir,
            cell_name=name,
            l2_w=l2_w,
            p_w=p_w,
            g_w=g_w,
            config=config,
            ds_val=ds_val,
            metric_pair_idx=metric_pair_idx,
            visual_pair_idx=visual_pair_idx,
            out_dir=args.out,
            device=args.device,
            ckpt_step=args.ckpt_step,
            no_render_visuals=args.no_render_visuals,
            include_lpips=args.include_lpips,
        )
        all_metrics.append(m)
        print(
            f"  -> {m.get('status', '?')}; "
            f"avg_psnr={m.get('avg_psnr', float('nan')):.3f} "
            f"avg_ssim={m.get('avg_ssim', float('nan')):.4f} "
            f"avg_l2={m.get('avg_l2_loss', float('nan')):.5f}"
        )

    metrics_csv = args.out / "metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cell",
                "status",
                "ckpt_path",
                "checkpoint_step",
                "num_eval_pairs",
                "lpips_status",
                "train_l2",
                "train_perceptual",
                "train_gs_reg",
                "avg_l2_loss",
                "avg_psnr",
                "avg_ssim",
                "avg_perceptual_loss",
                "avg_gs_reg_loss",
                "avg_lpips_loss",
                "avg_weighted_loss",
            ]
        )
        for m in all_metrics:
            writer.writerow(
                [
                    m.get("cell", ""),
                    m.get("status", ""),
                    m.get("ckpt_path", ""),
                    m.get("checkpoint_step", ""),
                    m.get("num_eval_pairs", ""),
                    m.get("lpips_status", ""),
                    m.get("train_l2_loss_weight", ""),
                    m.get("train_perceptual_loss_weight", ""),
                    m.get("train_gs_reg_loss_weight", ""),
                    m.get("avg_l2_loss", ""),
                    m.get("avg_psnr", ""),
                    m.get("avg_ssim", ""),
                    m.get("avg_perceptual_loss", ""),
                    m.get("avg_gs_reg_loss", ""),
                    m.get("avg_lpips_loss", ""),
                    m.get("avg_weighted_loss", ""),
                ]
            )
    print(f"\nWrote {metrics_csv}")

    metrics_json = args.out / "metrics.json"
    metrics_json.write_text(
        json.dumps(
            {
                "quantitative_pair_idx": metric_pair_idx,
                "qualitative_pair_idx": visual_pair_idx,
                "checkpoint_step": args.ckpt_step,
                "lpips_status": "computed" if args.include_lpips else "not_computed",
                "cells": all_metrics,
            },
            indent=2,
        )
    )
    print(f"Wrote {metrics_json}")

    if not args.no_render_visuals:
        make_grid_visual(args.out, visual_pair_idx, [c[0] for c in cells_to_run])

    return 0


if __name__ == "__main__":
    sys.exit(main())
