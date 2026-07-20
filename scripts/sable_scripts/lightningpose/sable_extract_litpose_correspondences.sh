#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="correspondences"
#SBATCH --partition=cpu
#SBATCH -c 4
#SBATCH -t 1:00:00
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH -o scripts/sable_scripts/lightningpose/precompute_litpose_correspondences_%j.log

exec 2>&1
source ~/.bashrc
conda activate "${CONDA_ENV:-sable}"

# Resolve repo root without hardcoding a user account: explicit REPO_ROOT override wins,
# then the sbatch submit dir (run `sbatch` from the repo root), then this script's location.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    fi
fi
CONFIG="${CONFIG:-$REPO_ROOT/configs/multiview/extraction_pipeline_sable.yaml}"
SCRIPT="${REPO_ROOT}/beast/preprocess/sable/precompute_litpose_correspondences.py"

echo "=== executing ==="
python "${SCRIPT}" \
    --config "${CONFIG}" \
    --overwrite
