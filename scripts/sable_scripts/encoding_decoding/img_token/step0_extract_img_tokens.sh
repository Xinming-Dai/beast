#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-5:59:00
#SBATCH -J extract_img_tokens
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step0_extract_img_tokens_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 0 (prerequisite) per docs/sable/neural_encoding_decoding.md: extracts only the
# per-patch img_tokens latent needed by this img_token/ pipeline (step1_run_pca_and_save.sh
# onward). For the full-latent variant (frame/dino/cat), see encoding/step1_sable_latent.sh.
JOB_ID="20284092"
EIDS="${EIDS:72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"                                            # space-separated session IDs (eids); default: use training config's session_names

MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_multisession/$JOB_ID}"
DATASET_BASE=/work/hdd/bfsr/xdai3/IBL_data/synchronized
DATASET_PATH="${DATASET_PATH:-$DATASET_BASE/extracted_frames/eval}"
VDA_CACHE_ROOT=$DATASET_BASE/extracted_frames_for_eyz/eval/depth_map
CORRESPONDENCE_CACHE_ROOT=$DATASET_BASE/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences                      # dir with config.yaml + *best.ckpt
OUTPUT_DIR=$MODEL_DIR/latents
LATENT_FLAGS="${LATENT_FLAGS:---return-img-tokens}"                  # img-tokens-only by default
RESUME="${RESUME:-true}"                                             # set to "false" to force recompute
BATCH_SIZE="${BATCH_SIZE:-64}"                                       # override training.batch_size_per_gpu

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Sable img tokens -> $OUTPUT_DIR"

ARGS=(
    --model "$MODEL_DIR"
    --input "$DATASET_PATH"
    --output "$OUTPUT_DIR"
    --extract-latents
)
[ -n "$EIDS" ] && ARGS+=(--session-names $EIDS)
[ -n "$VDA_CACHE_ROOT" ] && ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")
[ "$RESUME" = "false" ] && ARGS+=(--no-resume)
[ -n "$BATCH_SIZE" ] && ARGS+=(--latent-batch-size "$BATCH_SIZE")

beast predict "${ARGS[@]}" $LATENT_FLAGS

conda deactivate
