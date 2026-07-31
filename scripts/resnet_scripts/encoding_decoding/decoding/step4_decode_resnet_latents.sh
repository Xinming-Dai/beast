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
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=20505763
SPLIT="${SPLIT:-test}"
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/resnet_ae_152/$EID/$JOB_ID}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
ESTIMATED_DIR="${ESTIMATED_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/$SPLIT}"
DATASET_BASE="${DATASET_BASE:-/work/hdd/bfsr/xdai3/IBL_data/synchronized}"
TARGET_LEFT="${TARGET_LEFT:-$DATASET_BASE/extracted_frames/eval/leftCamera.video/_iblrig_leftCamera.downsampled.$EID}"
TARGET_RIGHT="${TARGET_RIGHT:-$DATASET_BASE/extracted_frames/eval/rightCamera.video/_iblrig_rightCamera.downsampled.$EID}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/decode_saved_latents}"

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding resnet estimated frame latents from $ESTIMATED_DIR"

python -m beast.sable_encoding_decoding.resnet.decode_resnet_latents \
    --model-dir "$MODEL_DIR" \
    --estimated-dir "$ESTIMATED_DIR" \
    --target-frame-mapping-left "$TARGET_LEFT" \
    --target-frame-mapping-right "$TARGET_RIGHT" \
    --out-dir "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
