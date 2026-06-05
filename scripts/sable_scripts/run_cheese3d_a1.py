#!/usr/bin/env python3
"""
A1: VDA online depth sanity check.
Validates: Cheese3D + VDA online depth → non-zero, non-NaN depth maps.

Key checks after run:
  ls outputs/cheese3d_phase1_a1/
  # Expect: vda_rgb_orig.png, vda_rgb_stretched_518.png,
  #         vda_depth_gray_518.png, vda_depth_turbo_518.png
  python -c "
    from PIL import Image; import numpy as np
    d = np.array(Image.open('outputs/cheese3d_phase1_a1/vda_depth_gray_518.png'))
    print('depth shape:', d.shape, 'min/max/mean:', d.min(), d.max(), d.mean())
    assert d.max() > 0, 'ERROR: depth is all zeros'
    print('PASS: depth is non-zero')
  "
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# ── Environment ────────────────────────────────────────────────────────────────
HF_DINO_DIR = "/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m"
os.environ["HF_HOME"] = HF_DINO_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
os.environ["PATH"] = "/home/jqh/miniconda3/envs/neuro/bin:" + os.environ.get("PATH", "")

import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Config + Model ──────────────────────────────────────────────────────────
from beast.io import load_config
from beast.models.sable import Sable
from beast.train_sable import train_sable

CONFIG = os.path.join(REPO_ROOT, "configs/sable_cheese3d_a1.yaml")
OUTPUT = os.path.join(REPO_ROOT, "outputs/cheese3d_phase1_a1")

print(f"\nConfig : {CONFIG}")
print(f"Output : {OUTPUT}")
print(f"HF_HOME: {os.environ['HF_HOME']}\n")

config = load_config(CONFIG)
model = Sable(config)
train_sable(config, model, output_dir=OUTPUT)

print("\n[A1] Done.")
