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
SCRIPT=/u/xdai3/project3d/SBALE_repo/beast/beast/preprocess/sable/sable_extract_litpose_correspondences.py

echo "=== executing ==="
python "${SCRIPT}" \
    --config "${CONFIG}" \
    --input-dir /work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/pretrain \
    --overwrite
