#!/usr/bin/env python3
"""Full-model smoke test for online VDA + Sable + renderer + perceptual.

Steps:
1. Load base config sable_ibl3d.yaml + apply online-VDA overrides + small batch.
2. Instantiate Sable, load init weights from erayzer_dl3dv.pt.
3. Build a batch of [B=12, 2, 3, 320, 320] from the loaders (no .npy calib,
   so the model uses its learned pose predictor).
4. Forward + backward + optimizer step.
5. Record peak VRAM via torch.cuda.max_memory_allocated() and elapsed time.
6. Assertions:
   - Peak VRAM <= 70 GiB (H100 80GB - 10 GiB headroom for CUDA kernels).
   - All loss components finite.
   - Gradients finite.
   - Parameters updated.

If OOM at batch=24, reduce micro-batch and increase grad_accum proportionally
to keep effective batch=24. The chosen micro-batch is printed and MUST be
echoed by all 8 cells' run_one_cell.sh invocations.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = "/cephfs/jinqihang/SABLE/beast"
PROJECT_ROOT = "/cephfs/jinqihang/SABLE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/cephfs/jinqihang/SABLE/outputs/loss_weighting/_smoke/online_vda_full_model"
        ),
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default="/cephfs/jinqihang/SABLE/ckpt/E-RayZer-private/checkpoints/erayzer_dl3dv.pt",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="/cephfs/jinqihang/SABLE/datasets/beast3d-data/sable_ibl_4b00"
    )
    parser.add_argument("--eid", type=str, default="4b00df29-3769-43be-bb40-128b1cba6d35")
    parser.add_argument(
        "--vda-ckpt",
        type=str,
        default="/cephfs/jinqihang/SABLE/third_party/VDA/checkpoints/video_depth_anything_vitb.pth",
    )
    parser.add_argument(
        "--vda-repo", type=str, default="/cephfs/jinqihang/SABLE/third_party/VDA"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--max-fwdbwd",
        type=int,
        default=2,
        help="Number of micro-batches to run (default 2, one optimizer step).",
    )
    parser.add_argument(
        "--allow-oom",
        action="store_true",
        help="If set, exit gracefully on OOM instead of asserting.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Load base sable_ibl3d.yaml and apply overrides."""
    sys.path.insert(0, REPO_ROOT)
    import yaml  # noqa: PLC0415

    with open(Path(REPO_ROOT) / "configs" / "sable" / "sable_ibl3d.yaml") as f:
        config = yaml.safe_load(f)

    # --- frozen online-VDA contract ---
    config["model"]["vda"] = {
        "enabled": True,
        "mode": "online",
        "cache_root": None,
        "encoder": "vitb",
        "metric": False,
        "checkpoint_path": args.vda_ckpt,
        "repo_root": args.vda_repo,
        "debug_save": False,
        "debug_dir": None,
    }
    config["model"]["merge_pcd"]["correspondence_cache_root"] = (
        f"{args.dataset_path}/litpose_correspondences/processed_correspondences"
    )

    # --- training overrides ---
    config["training"]["dataset_path"] = args.dataset_path
    config["training"]["session_names"] = [args.eid]
    config["training"]["batch_size_per_gpu"] = args.micro_batch
    config["training"]["grad_accum_steps"] = args.grad_accum
    config["training"]["max_fwdbwd_passes"] = 1
    config["training"]["val_every"] = 1
    config["training"]["checkpoint_every"] = 1
    config["training"]["checkpoint_dir"] = str(args.out)
    config["training"]["resume_ckpt"] = args.init_ckpt
    config["training"]["reset_training_state"] = True
    config["training"]["num_workers"] = 0
    config["training"]["print_every"] = 1
    config["training"]["checkpoint_steps"] = []  # no step list (uses checkpoint_every)
    config["training"]["save_visuals"] = False
    config["training"]["save_pointclouds"] = False
    config["training"]["save_camera_pointcloud_scene"] = False

    # This script uses a manual optimizer rather than the Lightning scheduler,
    # but keep the config valid if it is later routed through Trainer.
    config["optimizer"]["warmup"] = 2

    return config


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Symlink VGG weights into the CWD so LossComputer can find them.
    vgg_src = (
        "/cephfs/jinqihang/SABLE/ckpt/imagenet-vgg-verydeep-19.mat"
    )
    here = Path.cwd()
    here.joinpath("metric_checkpoint").mkdir(parents=True, exist_ok=True)
    vgg_dst = here / "metric_checkpoint" / "imagenet-vgg-verydeep-19.mat"
    if not vgg_dst.exists():
        os.symlink(vgg_src, vgg_dst)

    print("Building config...", flush=True)
    config = build_config(args)

    print("Building Sable model...", flush=True)
    from beast.models.sable import Sable  # noqa: PLC0415

    model = Sable(config)
    if args.init_ckpt and Path(args.init_ckpt).exists():
        print(f"Loading init ckpt: {args.init_ckpt}", flush=True)
        raw = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw.get("model", raw))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  WARNING missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"  WARNING unexpected keys: {len(unexpected)}", flush=True)
    model = model.to(args.device)
    model.train()

    if args.max_fwdbwd < args.grad_accum:
        raise ValueError(
            "--max-fwdbwd must be at least --grad-accum so the smoke test "
            "executes and verifies an optimizer step"
        )

    # Use AdamW with the configured lr/wd as the optimizer.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["optimizer"]["lr"]),
        betas=(float(config["optimizer"]["beta1"]), float(config["optimizer"]["beta2"])),
        weight_decay=float(config["optimizer"]["wd"]),
    )
    tracked_name, tracked_param = next(
        (name, param)
        for name, param in reversed(list(model.named_parameters()))
        if param.requires_grad
    )
    tracked_before = tracked_param.detach().clone()

    # Build dataset for one batch.
    print("Building dataset & dataloader...", flush=True)
    from beast.data.sable_dataset import (  # noqa: PLC0415
        IBLTwoViewDataset,
        collate_with_correspondence_padding,
    )

    train_ds = IBLTwoViewDataset(config, include_splits=["train"])
    print(f"  train_ds len={len(train_ds)}", flush=True)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_correspondence_padding,
        drop_last=True,
    )

    # Reset peak memory tracker before timing.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(args.device)

    print(
        f"Running {args.max_fwdbwd} fwdbwd pass(es) at micro-batch={args.micro_batch}...",
        flush=True,
    )
    t0 = time.time()
    peak_vram_gib = None
    oom = False
    optimizer_steps = 0
    component_history = []
    grad_finite = True
    saw_gradient = False
    nan_grads = []
    try:
        for step, batch in enumerate(loader):
            if step >= args.max_fwdbwd:
                break
            batch = {
                k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            outputs = model.get_model_outputs(batch)
            result = model.loss_computer(
                outputs["render"],
                outputs["target_image"],
                outputs["xyz_norm"],
                outputs["xyz_init_norm"],
                outputs.get("pixel_mask"),
                outputs.get("gaussian_mask"),
            )
            components = {
                "total": float(result.loss.detach()),
                "l2": float(result.l2_loss.detach()),
                "perceptual": float(result.perceptual_loss.detach()),
                "geom": float(result.gs_reg_loss.detach()),
            }
            if not all(
                torch.isfinite(value).item()
                for value in (
                    result.loss,
                    result.l2_loss,
                    result.perceptual_loss,
                    result.gs_reg_loss,
                )
            ):
                raise RuntimeError(f"non-finite loss component: {components}")
            component_history.append(components)
            (result.loss / args.grad_accum).backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                saw_gradient = True
                if not torch.isfinite(param.grad).all().item():
                    grad_finite = False
                    if name not in nan_grads:
                        nan_grads.append(name)
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            print(f"  step {step}: loss={components['total']:.4f}", flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        print(f"OOM caught: {exc}", flush=True)
        oom = True
    finally:
        elapsed = time.time() - t0
        peak_vram_gib = torch.cuda.max_memory_allocated(args.device) / 1024**3

    # Check parameter finiteness.
    params_finite = True
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all().item():
            params_finite = False
            print(f"  non-finite param: {n}", flush=True)
    params_updated = not torch.equal(tracked_before, tracked_param.detach())

    print("\nResults:", flush=True)
    print(f"  elapsed = {elapsed:.2f}s", flush=True)
    print(f"  peak VRAM = {peak_vram_gib:.2f} GiB", flush=True)
    print(f"  gradient finite = {grad_finite}", flush=True)
    print(f"  observed gradient = {saw_gradient}", flush=True)
    print(f"  parameters finite = {params_finite}", flush=True)
    print(f"  optimizer steps = {optimizer_steps}", flush=True)
    print(f"  tracked parameter updated ({tracked_name}) = {params_updated}", flush=True)
    if nan_grads:
        print(f"  non-finite grads: {nan_grads[:5]} ...", flush=True)

    result = {
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.micro_batch * args.grad_accum,
        "elapsed_sec": elapsed,
        "peak_vram_gib": peak_vram_gib,
        "grad_finite": grad_finite,
        "saw_gradient": saw_gradient,
        "params_finite": params_finite,
        "params_updated": params_updated,
        "tracked_parameter": tracked_name,
        "optimizer_steps": optimizer_steps,
        "loss_components": component_history,
        "nan_grads": nan_grads,
        "oom": oom,
        "nan_grad_count": len(nan_grads),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "device_total_gib": torch.cuda.get_device_properties(args.device).total_memory / 1024**3,
    }
    out_json = args.out / "smoke_result.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"  wrote {out_json}", flush=True)

    if oom:
        if args.allow_oom:
            print("OOM (allowed)", flush=True)
            return 0
        print("RESULT: OOM — reduce micro-batch and re-run with --allow-oom", flush=True)
        return 2
    if (
        not grad_finite
        or not saw_gradient
        or not params_finite
        or not params_updated
        or optimizer_steps < 1
    ):
        print("RESULT: gradient/parameter update check failed", flush=True)
        return 3
    if peak_vram_gib > 70.0:
        print(f"RESULT: peak VRAM {peak_vram_gib:.2f} GiB > 70 GiB limit", flush=True)
        return 4
    print("RESULT: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
