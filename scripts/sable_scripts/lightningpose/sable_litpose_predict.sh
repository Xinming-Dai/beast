#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH -t 0-00:30:00
#SBATCH -J litpose
#SBATCH -o scripts/sable_scripts/lightningpose/litpose_predict_sable_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
# LitPose prediction uses a separate Lightning Pose env (default `lp`); override with LP_CONDA_ENV.
conda activate "${LP_CONDA_ENV:-lp}"

# Resolve repo root without hardcoding a user account: explicit BEAST_REPO override wins,
# then the sbatch submit dir (run `sbatch` from the repo root), then this script's location.
if [ -z "${BEAST_REPO:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        BEAST_REPO="$SLURM_SUBMIT_DIR"
    else
        BEAST_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    fi
fi
ROOT="${ROOT:-/work/hdd/bfsr/xdai3/IBL-2view}"
LIGHTNING_POSE_MODEL_DIR="${LIGHTNING_POSE_MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/lightning_pose/multiview_transformer_3235_0}"
SCRIPT="${BEAST_REPO}/beast/preprocess/sable/run_litpose_predict_sable.py"

SESSION_IDS=(
  # b03fbc44-3d8e-4a6c-8a50-5ea3498568e0
  # b196a2ad-511b-4e90-ac99-b5a29ad25c22
  # b22f694e-4a34-4142-ab9d-2556c3487086
  # d0ea3148-948d-4817-94f8-dcaf2342bbbe
  # d23a44ef-1402-4ed7-97f5-47e9a7a504d9
  # dda5fc59-f09a-4256-9fb5-66c67667a466
  # e45481fa-be22-4365-972c-e7404ed8ab5a
  # e535fb62-e245-4a48-b119-88ce62a6fe67
)

# to skip labeled overlay videos, append: -- --skip_viz

PYTHONPATH="${BEAST_REPO}:${PYTHONPATH}" python -u "${SCRIPT}" \
  --root "${ROOT}" \
  --model-dir "${LIGHTNING_POSE_MODEL_DIR}" \
  --session-ids e535fb62-e245-4a48-b119-88ce62a6fe67 \
  --skip-existing

conda deactivate
