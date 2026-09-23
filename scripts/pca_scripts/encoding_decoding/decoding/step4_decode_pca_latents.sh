#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 0-00:59:00
#SBATCH -J pca_decode_latents
#SBATCH -o /u/xdai3/project3d/SABLE_repo_3/beast/scripts/pca_scripts/encoding_decoding/decoding/step4_decode_pca_latents_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# EID="${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}"
# EID="${EID:-72cb5550-43b4-4ef0-add5-e4adfdfb5e02}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=21810401
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/restricted_sessions/$JOB_ID}"
SPLIT="${SPLIT:-test}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
ESTIMATED_DIR="${ESTIMATED_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/$SPLIT}"
DATASET_BASE="${DATASET_BASE:-/work/hdd/bfsr/xdai3/IBL_data/synchronized}"
TARGET_LEFT="${TARGET_LEFT:-$DATASET_BASE/extracted_frames/eval/leftCamera.video/_iblrig_leftCamera.downsampled.$EID}"
TARGET_RIGHT="${TARGET_RIGHT:-$DATASET_BASE/extracted_frames/eval/rightCamera.video/_iblrig_rightCamera.downsampled.$EID}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/decode_saved_latents}"
USE_MASK=true
SEGMENTATION_ROOT="${SEGMENTATION_ROOT:-$DATASET_BASE/extracted_frames_for_eyz/eval}"
METRICS_ONLY=false
NEURAL_TRIAL_INDEX="${NEURAL_TRIAL_INDEX:-0}"
METRICS_NPZ="${METRICS_NPZ:-$OUT_DIR/psnr_ssim_metrics_1session.npz}"

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding PCA estimated frame latents from $ESTIMATED_DIR"

ARGS=(
    --model-dir "$MODEL_DIR"
    --estimated-dir "$ESTIMATED_DIR"
    --target-frame-mapping-left "$TARGET_LEFT"
    --target-frame-mapping-right "$TARGET_RIGHT"
    --out-dir "$OUT_DIR"
    --batch-size 60
    --image-size 320
)
[ -n "$NEURAL_TRIAL_INDEX" ] && ARGS+=(--neural-trial-index "$NEURAL_TRIAL_INDEX")
[ -n "$METRICS_NPZ" ] && ARGS+=(--metrics-npz "$METRICS_NPZ")
[ "$USE_MASK" = true ] && ARGS+=(--use-segmentation-mask --segmentation-root "$SEGMENTATION_ROOT" --eid "$EID")
[ "$METRICS_ONLY" = true ] && ARGS+=(--metrics-only)

python -m beast.sable_encoding_decoding.pca.decode_pca_latents "${ARGS[@]}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
