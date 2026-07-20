#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0-01:59:00
#SBATCH -J sable_generate_video
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/generate_video_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,INPUT_DIR=...,OUTPUT=... scripts/sable_scripts/encoding_decoding/generate_video.sh
INPUT_DIR="${INPUT_DIR:-<PATH_TO_RENDERED_FRAMES_DIR>}"
OUTPUT="${OUTPUT:-$INPUT_DIR/video.mp4}"
FPS="${FPS:-24}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Generating video from frames in $INPUT_DIR"

python -m beast.sable_encoding_decoding.video.video_generator \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT" \
    --fps "$FPS"

conda deactivate
