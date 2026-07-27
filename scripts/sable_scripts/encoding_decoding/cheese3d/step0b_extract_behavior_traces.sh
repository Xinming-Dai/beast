#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-01:00:00
#SBATCH -J behavior_traces
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/cheese3d/step0b_extract_behavior_traces_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Bins Cheese3D TL/TR Lightning Pose keypoints onto the same trials as
# extract_cheese3d_neural_data.py (via frame_manifest.json), for a raw-behavior encoding
# baseline. Must run step0_extract_neural_data.sh first. Pure numpy/csv — cheap CPU job.
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,EID=...,MIN_LIKELIHOOD=... \
#     scripts/sable_scripts/encoding_decoding/cheese3d/step0b_extract_behavior_traces.sh
EID="${EID:-20250523_B1_ephys-record_awake_000}"
FRAME_MANIFEST="${FRAME_MANIFEST:-/work/hdd/bfsr/xdai3/cheese3d_neural/neural_data/${EID}/frame_manifest.json}"
LP_CSV_TL="${LP_CSV_TL:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/lightning_pose/outputs_cheese-3d_test_450_LP3D/mvt_3d_loss_450_0/video_preds/${EID}_TL_18-24-03.csv}"
LP_CSV_TR="${LP_CSV_TR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/lightning_pose/outputs_cheese-3d_test_450_LP3D/mvt_3d_loss_450_0/video_preds/${EID}_TR_18-24-03.csv}"
BEHAVIOR_OUTPUT_DIR="${BEHAVIOR_OUTPUT_DIR:-/work/hdd/bfsr/xdai3/cheese3d_neural/behavior_data}"
MIN_LIKELIHOOD="${MIN_LIKELIHOOD:-0.0}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Cheese3D behavior traces for eid=$EID"

python -m beast.preprocess.cheese3d.extract_cheese3d_behavior_traces \
    --frame-manifest "$FRAME_MANIFEST" \
    --lp-csv-tl "$LP_CSV_TL" \
    --lp-csv-tr "$LP_CSV_TR" \
    --eid "$EID" \
    --behavior-output-dir "$BEHAVIOR_OUTPUT_DIR" \
    --min-likelihood "$MIN_LIKELIHOOD"

conda deactivate
