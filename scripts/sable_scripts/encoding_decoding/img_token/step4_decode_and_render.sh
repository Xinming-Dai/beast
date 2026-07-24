#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-01:00:00
#SBATCH -J decode_and_render
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step4_decode_and_render_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 4 feeds (real or decoded) img tokens through SABLE's decoder to reconstruct images, save point clouds, and/or compute PSNR/SSIM.
# Requires step3's unprojected tokens (or raw step0 img tokens) as Z_SOURCE; produces
# rendered frames under OUT_DIR, consumed by step5_generate_video.sh.
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,Z_SOURCE=...,MODEL_ROOT=...,DATASET_PATH=... \
#     scripts/sable_scripts/encoding_decoding/img_token/step4_decode_and_render.sh
Z_SOURCE="${Z_SOURCE:-<PATH_TO_IMG_TOKENS_NPZ_OR_DIR>}"     # single .npz or a directory of img_tokens*.npz
MODEL_ROOT="${MODEL_ROOT:-<MODEL_ROOT>}"                    # dir with config.yaml + *best.ckpt
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/render_out}"
DATASET_PATH="${DATASET_PATH:-<PATH_TO_DATASET>}"
VDA_CACHE_ROOT="${VDA_CACHE_ROOT:-}"                        # optional override
CORRESPONDENCE_CACHE_ROOT="${CORRESPONDENCE_CACHE_ROOT:-}"  # optional override

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding + rendering img tokens from $Z_SOURCE"

ARGS=(
    --z-source "$Z_SOURCE"
    --out-dir "$OUT_DIR"
    --model-dir "$MODEL_ROOT"
    --dataset-path "$DATASET_PATH"
)
[ -n "$VDA_CACHE_ROOT" ] && ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")

python -m beast.sable_encoding_decoding.render.decode_and_render "${ARGS[@]}"

conda deactivate
