#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="correspondences"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH --mem=10G
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/lightningpose/precompute_litpose_correspondences_%j.log

exec 2>&1
source ~/.bashrc

CONFIG=/u/xdai3/project3d/SBALE_repo/beast/configs/multiview/extraction_pipeline_sable.yaml
SCRIPT=/u/xdai3/project3d/SBALE_repo/beast/beast/preprocess/sable/ibl/sable_extract_litpose_correspondences.py

SESSION_IDS=(
    "3e6a97d3-3991-49e2-b346-6948cb4580fb",
    "5dcee0eb-b34d-4652-acc3-d10afc6eae68"
)

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting litpose correspondences for sessions: ${SESSION_IDS[*]}"

echo "=== executing ==="
python "${SCRIPT}" \
    --config "${CONFIG}" \
    --layout eval \
    --input-dir /work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/eval \
    --litpose-root /work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames_for_eyz/eval/litpose_correspondences \
    --output-root /work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames_for_eyz/eval \
    --eids "${SESSION_IDS[@]}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."