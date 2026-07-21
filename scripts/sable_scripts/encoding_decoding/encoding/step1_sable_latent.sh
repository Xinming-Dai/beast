#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-11:59:00
#SBATCH -J sable_latent_extraction
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step1_sable_latent_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Extracts per-pair Sable latent tensors (frame_z / depth_z / combined_z / img_tokens) from a
# trained checkpoint, for downstream neural encoding/decoding. Analogous to E-RayZer's
# scripts/mia/erz_dino/encoding/step1_erz_latent.sh + src/inference.py latent-export flags
# (--return-cat-z, --return-combined-all-z, --return-img-tokens); this repo only supports the
# "legacy" per-pair .npy resume layout (safe to rerun after a killed job: already-completed
# batches are skipped, unless RESUME=false).
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,MODEL_DIR=...,DATASET_PATH=... \
#     scripts/sable_scripts/encoding_decoding/encoding/step1_sable_latent.sh
MODEL_DIR="${MODEL_DIR:-<MODEL_ROOT>}"                      # dir with config.yaml + *best.ckpt
DATASET_PATH="${DATASET_PATH:-<PATH_TO_DATASET>}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_DIR/latents}"
SPLITS="${SPLITS:-train val}"
LATENT_FLAGS="${LATENT_FLAGS:---return-all-z --return-img-tokens}"   # or e.g. "--return-frame-z"
RESUME="${RESUME:-true}"                                    # set to "false" to force recompute
VDA_CACHE_ROOT="${VDA_CACHE_ROOT:-}"                        # optional override
CORRESPONDENCE_CACHE_ROOT="${CORRESPONDENCE_CACHE_ROOT:-}"  # optional override

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Sable latents -> $OUTPUT_DIR"

ARGS=(
    --model "$MODEL_DIR"
    --input "$DATASET_PATH"
    --output "$OUTPUT_DIR"
    --extract-latents
    --splits $SPLITS
)
[ -n "$VDA_CACHE_ROOT" ] && ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")
[ "$RESUME" = "false" ] && ARGS+=(--no-resume)

beast predict "${ARGS[@]}" $LATENT_FLAGS

conda deactivate
