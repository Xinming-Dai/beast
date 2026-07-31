#!/bin/bash
#SBATCH -A bfsx-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-03:59:00
#SBATCH -J sam3_segment_eval
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/segmentation_masks/sam3_segment_eval_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
conda activate beast

BEAST_REPO=/u/xdai3/project3d/SBALE_repo/beast
CONFIG=/u/xdai3/project3d/SBALE_repo/beast/configs/multiview/extraction_pipeline_sable.yaml
SCRIPT=/u/xdai3/project3d/SBALE_repo/beast/beast/preprocess/sable/precompute_sam3_masks_eval.py

FRAMES_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/eval
OUTPUT_ROOT=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames_for_eyz/eval

# only these two sessions are segmented; add/remove UUIDs here to change scope
SESSION_IDS=(
    # "f312aaec-3b6f-44b3-86b4-3a0c119c0438"
    "4b00df29-3769-43be-bb40-128b1cba6d35"
)

PYTHONPATH="${BEAST_REPO}:${PYTHONPATH}" python -u "${SCRIPT}" \
  --config "${CONFIG}" \
  --frames-dir "${FRAMES_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --eids "${SESSION_IDS[@]}" \
  --split test

conda deactivate
