#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:59:00
#SBATCH -J resnet_decode_latents
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/resnet_scripts/encoding_decoding/decoding/step4_decode_resnet_latents_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 4: decode step3's neurally-estimated resnet frame latents back into reconstructed
# frames, via beast.sable_encoding_decoding.resnet.decode_resnet_latents. Unlike beast's ViT-MAE
# decode (step4_decode_tokens.sh), there is no ids_restore/patch-grid to carry through: the
# ResnetAutoencoder decodes the flat 768-dim latent directly (latents_to_decoder -> decoder), and
# the camera axis is already a plain leading dim of size 2 in the saved latents (no shard-layout
# merge/unmerge needed). Produces rendered frames under OUT_DIR, consumed by
# step5_generate_video.sh.
EID="${EID:-f312aaec-3b6f-44b3-86b4-3a0c119c0438}"
JOB_ID=20668654
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/resnet_ae_152/$EID/$JOB_ID}"
SPLIT="${SPLIT:-test}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
ESTIMATED_DIR="${ESTIMATED_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/$SPLIT}"
DATASET_BASE="${DATASET_BASE:-/work/hdd/bfsr/xdai3/IBL_data/synchronized}"
TARGET_LEFT="${TARGET_LEFT:-$DATASET_BASE/extracted_frames/eval/leftCamera.video/_iblrig_leftCamera.downsampled.$EID}"
TARGET_RIGHT="${TARGET_RIGHT:-$DATASET_BASE/extracted_frames/eval/rightCamera.video/_iblrig_rightCamera.downsampled.$EID}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/decode_saved_latents}"
USE_MASK=true
SEGMENTATION_ROOT="${SEGMENTATION_ROOT:-$DATASET_BASE/extracted_frames_for_eyz/eval}"
METRICS_ONLY="${METRICS_ONLY:-true}"

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding resnet estimated frame latents from $ESTIMATED_DIR"

ARGS=(
    --model-dir "$MODEL_DIR"
    --estimated-dir "$ESTIMATED_DIR"
    --target-frame-mapping-left "$TARGET_LEFT"
    --target-frame-mapping-right "$TARGET_RIGHT"
    --out-dir "$OUT_DIR"
)
[ "$USE_MASK" = true ] && ARGS+=(--use-segmentation-mask --segmentation-root "$SEGMENTATION_ROOT" --eid "$EID")
[ "$METRICS_ONLY" = true ] && ARGS+=(--metrics-only)

python -m beast.sable_encoding_decoding.resnet.decode_resnet_latents "${ARGS[@]}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
