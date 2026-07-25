#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-5:59:00
#SBATCH -J extract_cheese3d_img_tokens
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/cheese3d/step2_extract_img_tokens_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Extracts Sable latents (frame_z/dino_z/combined_z/img_tokens) for the Cheese3D ephys session's
# eval-layout frames (step1's output), using a checkpoint trained from
# configs/sable/sable_cheese3d_ephys_session.yaml. Deliberately does NOT pass
# --vda-cache-root/--correspondence-cache-root: unlike IBLTwoViewDataset,
# Cheese3DDataset computes VDA depth online and uses a fixed 3-point correspondence set, so
# those caches don't apply here (see docs/sable/cheese3d_dataset.md).
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,MODEL_DIR=...,EID=... \
#     scripts/sable_scripts/encoding_decoding/cheese3d/step2_extract_img_tokens.sh
EID="${EID:-20250523_B1_ephys-record_awake_000}"
MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/cheese3d/20156493}"                        # dir with config.yaml + *best.ckpt
EVAL_FRAMES_DIR="${EVAL_FRAMES_DIR:-/work/hdd/bfsr/xdai3/cheese3d_neural/eval_frames}"
OUTPUT_DIR="$MODEL_DIR/latents"
LATENT_FLAGS="${LATENT_FLAGS:---return-all-z --return-img-tokens}"   # or e.g. "--return-frame-z"
RESUME="${RESUME:-true}"                                             # set to "false" to force recompute
BATCH_SIZE="${BATCH_SIZE:-64}"                                       # override training.batch_size_per_gpu
DISABLE_INIT_GS="${DISABLE_INIT_GS:-true}"                          # set to "true" to skip VDA-depth Gaussian init/reg at inference

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Sable latents for eid=$EID -> $OUTPUT_DIR"

ARGS=(
    --model "$MODEL_DIR"
    --input "$EVAL_FRAMES_DIR"
    --output "$OUTPUT_DIR"
    --session-names "$EID"
    --extract-latents
)
[ "$RESUME" = "false" ] && ARGS+=(--no-resume)
[ -n "$BATCH_SIZE" ] && ARGS+=(--latent-batch-size "$BATCH_SIZE")
[ "$DISABLE_INIT_GS" = "true" ] && ARGS+=(--disable-init-gs)

beast predict "${ARGS[@]}" $LATENT_FLAGS

conda deactivate
