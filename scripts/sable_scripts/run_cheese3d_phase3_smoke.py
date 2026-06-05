#!/usr/bin/env python3
"""
Stage 1 smoke / validation launcher for Cheese3D LP3D correspondences.

This script focuses on geometry-contract validation and Kabsch-path activation.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

HF_DINO_DIR = "/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m"
os.environ["HF_HOME"] = HF_DINO_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["PATH"] = "/home/jqh/miniconda3/envs/neuro/bin:" + os.environ.get("PATH", "")
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Cheese3D LP3D Kabsch validation")
    parser.add_argument("--config", default="beast/configs/sable_cheese3d_lp3d.yaml",
                        help="Path to config file relative to workspace root")
    parser.add_argument("--session", default="20231031_B20_chew_bl_000")
    parser.add_argument("--correspondence_cache", default=None,
                        help="Path to LP3D correspondence cache (required for validation/smoke)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a 4-step Stage 1 smoke after validation")
    parser.add_argument(
        "--smoke_mode",
        choices=("clean", "debug"),
        default="debug",
        help=(
            "Smoke success criterion: 'clean' verifies non-zero, finite gs_reg_loss; "
            "'debug' verifies debug_pcd/batch_000/ PLY and overlay artifacts."
        ),
    )
    parser.add_argument("--smoke_output_dir", default="./outputs/cheese3d_stage1_single_session/smoke",
                        help="Output directory for smoke artifacts relative to workspace root")
    parser.add_argument("--sample_limit", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from beast.io import load_config
    from beast.models.sable import Sable
    from beast.train_sable import train_sable

    print(f"torch: {torch.__version__} cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cache_root_arg = args.correspondence_cache
    if cache_root_arg is None:
        raise ValueError("--correspondence_cache is required; point it to a freshly generated Stage 1 cache")
    cache_root = Path(cache_root_arg)
    if not cache_root.exists():
        raise FileNotFoundError(f"Correspondence cache not found: {cache_root}")

    session_dir = cache_root / args.session
    if session_dir.exists():
        bundles = list(session_dir.rglob("litpose_matches.npz"))
        non_empty = [b for b in bundles if b.stat().st_size > 1000]
        print(f"Cache summary for {args.session}: total={len(bundles)} non_empty={len(non_empty)}")

    config_path = (REPO_ROOT.parent / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    validator_script = REPO_ROOT / "scripts" / "sable_scripts" / "validate_cheese3d_stage1_cache.py"
    validate_cmd = [
        str(Path(os.environ["PATH"].split(":")[0]) / "python"),
        str(validator_script),
        "--config", str(config_path),
        "--cache_root", str(cache_root),
        "--sessions", args.session,
        "--sample_limit", str(args.sample_limit),
    ]
    import subprocess
    subprocess.run(validate_cmd, check=True)

    config = load_config(str(config_path))
    config["training"]["sessions"] = [args.session]
    config["training"]["max_frames_per_session"] = 4 if args.smoke else config["training"].get("max_frames_per_session")
    config["training"]["num_workers"] = 0
    config["training"]["val_every"] = 0
    config["training"]["save_visuals"] = False

    smoke_output_dir = (REPO_ROOT / args.smoke_output_dir).resolve()
    if args.smoke:
        config["training"]["max_fwdbwd_passes"] = 4
        config["training"]["checkpoint_dir"] = str(smoke_output_dir)
    else:
        checkpoint_dir = Path(config["training"]["checkpoint_dir"]).expanduser()
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = (REPO_ROOT / checkpoint_dir).resolve()
        else:
            checkpoint_dir = checkpoint_dir.resolve()
        config["training"]["checkpoint_dir"] = str(checkpoint_dir)

    config["model"]["merge_pcd"]["correspondence_cache_root"] = str(cache_root)
    config["model"]["merge_pcd"]["debug_merged_pcd"] = args.smoke_mode == "debug"
    config["model"]["vda"]["mode"] = "online"

    output_dir = Path(config["training"]["checkpoint_dir"])
    print(f"Output: {output_dir}")
    print(f"init_gs: {config['model']['gaussians']['init_gs']}")
    print(f"correspondence_mode: {config['training'].get('correspondence_mode')}")
    print(f"smoke_mode: {args.smoke_mode}")
    print(f"debug_merged_pcd: {config['model']['merge_pcd'].get('debug_merged_pcd')}")
    if args.smoke:
        if args.smoke_mode == "clean":
            print("Smoke success criterion: verify non-zero, finite gs_reg_loss.")
        else:
            print("Smoke success criterion: verify debug_pcd/batch_000/ PLY and overlay artifacts.")

    print("Initializing Sable model...")
    model = Sable(config)
    print("Model initialized successfully.")

    if args.smoke:
        train_sable(config, model, output_dir=str(output_dir))
        if args.smoke_mode == "clean":
            print("Clean smoke finished. Verify non-zero, finite gs_reg_loss in the 4-step run.")
        else:
            print("Debug smoke finished. Verify debug_pcd/batch_000/ PLY and overlay artifacts.")
    else:
        print("Validation completed without training. Re-run with --smoke to execute the 4-step Stage 1 smoke.")


if __name__ == "__main__":
    main()
