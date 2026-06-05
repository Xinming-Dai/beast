#!/usr/bin/env python3
"""
A0 smoke test launcher: Cheese3D zero-shot with precomputed VDA (zero depth).
Validates: Cheese3D data loading → SABLE forward → renderer → loss (no crash/NaN).
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# ── Environment ────────────────────────────────────────────────────────────────
HF_DINO_DIR = "/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m"
os.environ["HF_HOME"] = HF_DINO_DIR
os.environ["HF_HUB_OFFLINE"] = "1"   # offline only, no network fallback
os.environ["PYTHONUNBUFFERED"] = "1"
# Blackwell GPU workaround; sm_86 for RTX 3090
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

CONFIG = os.path.join(REPO_ROOT, "configs/sable_cheese3d.yaml")
OUTPUT = os.path.join(REPO_ROOT, "outputs/cheese3d_phase1_a0_full")

print(f"\nConfig : {CONFIG}")
print(f"Output : {OUTPUT}")
print(f"HF_HOME: {os.environ['HF_HOME']}\n")

config = load_config(CONFIG)
model = Sable(config)
train_sable(config, model, output_dir=OUTPUT)

print("\n[A0] Done.")
