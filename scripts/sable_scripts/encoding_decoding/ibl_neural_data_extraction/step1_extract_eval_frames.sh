#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem 7G
#SBATCH -t 0-04:00:00
#SBATCH -J extract_sable_eval_frames
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/ibl_neural_data_extraction/step1_extract_eval_frames_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Extracts the exact eval-layout frames (left/right cameras by default) for the trials chosen
# by step0, from the raw IBL two-view session videos, via a single sequential OpenCV decode
# pass per camera.
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,EID=... \
#     scripts/sable_scripts/encoding_decoding/ibl_neural_data_extraction/step1_extract_eval_frames.sh
EID="${EID:-3e6a97d3-3991-49e2-b346-6948cb4580fb}"
NEURAL_DATA_DIR="${NEURAL_DATA_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"
RAW_VIDEO_DIR="${RAW_VIDEO_DIR:-/work/hdd/bfsr/xdai3/IBL-2view}"
EVAL_FRAMES_OUTPUT_DIR="${EVAL_FRAMES_OUTPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/eval}"
CAMERAS="${CAMERAS:-left right}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting IBL eval frames for eid=$EID"

python -m beast.preprocess.sable.extract_sable_eval_frames \
    --eid "$EID" \
    --neural-data-dir "$NEURAL_DATA_DIR" \
    --raw-video-dir "$RAW_VIDEO_DIR" \
    --output-dir "$EVAL_FRAMES_OUTPUT_DIR" \
    --cameras $CAMERAS

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job Done."
conda deactivate
