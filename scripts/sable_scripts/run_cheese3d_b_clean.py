#!/usr/bin/env python3
"""
B-clean: Cheese3D mask-bbox pseudo-correspondence — clean run.
Validates real Kabsch branch (gs_reg > 0, no xyz=xyz_init shortcut).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Environment
HF_DINO_DIR = "/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m"
os.environ["HF_HOME"] = HF_DINO_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["PATH"] = "/home/jqh/miniconda3/envs/neuro/bin:" + os.environ.get("PATH", "")
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba-cache"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-cache"


def main() -> None:
    import torch
    from beast.io import load_config
    from beast.models.sable import Sable
    from beast.train_sable import train_sable

    print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config_path = os.path.join(REPO_ROOT, "configs/sable_cheese3d_b_clean.yaml")
    output_dir = os.path.join(REPO_ROOT, "outputs/cheese3d_phase2_b_clean")

    print(f"\nConfig : {config_path}")
    print(f"Output : {output_dir}")
    print(f"HF_HOME: {os.environ['HF_HOME']}\n")

    config = load_config(config_path)
    model = Sable(config)
    train_sable(config, model, output_dir=output_dir)

    print("\n[B-clean] Done.")


if __name__ == "__main__":
    main()
