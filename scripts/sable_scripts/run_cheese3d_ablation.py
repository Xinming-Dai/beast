#!/usr/bin/env python3
"""
Phase 2 Ablation runner — shared launcher.

Usage:
    python run_cheese3d_ablation.py configs/sable_cheese3d_ablation_baseline.yaml
    python run_cheese3d_ablation.py configs/sable_cheese3d_ablation_pseudokabsch.yaml

Outputs go to the config's checkpoint_dir.
"""
import os
import sys
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

HF_DINO_DIR = "/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m"
os.environ["HF_HOME"] = HF_DINO_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["PATH"] = "/home/jqh/miniconda3/envs/neuro/bin:" + os.environ.get("PATH", "")
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Ablation runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    import torch
    from beast.io import load_config
    from beast.models.sable import Sable
    from beast.train_sable import train_sable

    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config_path = os.path.join(REPO_ROOT, args.config)
    config = load_config(config_path)
    output_dir = os.path.join(REPO_ROOT, config["training"]["checkpoint_dir"])

    print(f"\nConfig : {config_path}")
    print(f"Output : {output_dir}")
    print(f"init_gs: {config['model']['gaussians']['init_gs']}")
    print(f"debug_merged_pcd: {config['model']['merge_pcd']['debug_merged_pcd']}")
    print(f"max_fwdbwd_passes: {config['training']['max_fwdbwd_passes']}")
    print(f"HF_HOME: {os.environ['HF_HOME']}\n")

    model = Sable(config)
    train_sable(config, model, output_dir=output_dir)

    print("\n[Ablation run] Done.")


if __name__ == "__main__":
    main()
