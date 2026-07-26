#!/bin/bash
#SBATCH -A bezq-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-11:59:00
#SBATCH -J extraction
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step1_sable_latent_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

JOB_ID="20014553"
EIDS="${EIDS:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"                                           # space-separated session IDs (eids); default: use training config's session_names

MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/$EIDS/$JOB_ID}"
DATASET_BASE=/work/hdd/bfsr/xdai3/IBL_data/synchronized
DATASET_PATH="${DATASET_PATH:-$DATASET_BASE/extracted_frames/eval}"
VDA_CACHE_ROOT=$DATASET_BASE/extracted_frames_for_eyz/eval/depth_map
CORRESPONDENCE_CACHE_ROOT=$DATASET_BASE/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences                      # dir with config.yaml + *best.ckpt
OUTPUT_DIR=$MODEL_DIR/latents
LATENT_FLAGS="${LATENT_FLAGS:---return-all-z --return-img-tokens}"   # or e.g. "--return-frame-z"
RESUME="${RESUME:-true}"                                             # set to "false" to force recompute
BATCH_SIZE="${BATCH_SIZE:-64}"                                       # override training.batch_size_per_gpu

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Sable latents -> $OUTPUT_DIR"

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
