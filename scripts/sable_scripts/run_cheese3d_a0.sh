#!/bin/bash
# A0 smoke test launcher for Cheese3D zero-shot evaluation
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Paths ────────────────────────────────────────────────────────────────────
export HF_HOME="/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m"
export PYTHONUNBUFFERED=1

# Blackwell GPU workaround
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

CONFIG="${1:-$REPO_ROOT/configs/sable_cheese3d.yaml}"
shift || true

cd "$REPO_ROOT"

echo "============================================"
echo "A0: Cheese3D Phase 1 smoke test"
echo "HF_HOME: $HF_HOME"
echo "Config: $CONFIG"
echo "============================================"

# Verify GPU
python - <<'PY'
import torch
print(f"torch: {torch.__version__}  cuda: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

echo "[$(date)] Launching..."
/home/jqh/miniconda3/envs/neuro/bin/python -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from beast.train_sable import train_sable_from_config
train_sable_from_config('$CONFIG', output_dir='$REPO_ROOT/outputs/cheese3d_phase1_a0', overrides={\$*})
" 2>&1

echo "[$(date)] Done."
