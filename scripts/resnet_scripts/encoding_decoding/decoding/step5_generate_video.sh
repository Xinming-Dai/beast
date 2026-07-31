#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH -t 0-00:15:00
#SBATCH -J resnet_generate_vids
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/resnet_scripts/encoding_decoding/decoding/step5_generate_video_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 5: stitch a folder of decoded frames (from step4_decode_resnet_latents.sh) into an MP4.
# Reused unchanged from the beast/Sable pipeline — it only operates on a directory of
# rendered-frame images.
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=20505763
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/resnet_ae_152/$EID/$JOB_ID}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
INPUT_DIR="${INPUT_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/decode_saved_latents/decoded}"
OUTPUT="${OUTPUT:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/decode_saved_latents/video.mp4}"
FPS="${FPS:-24}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Generating video from decoded frames in $INPUT_DIR"

python -m beast.sable_encoding_decoding.video.video_generator \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT" \
    --fps "$FPS"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
