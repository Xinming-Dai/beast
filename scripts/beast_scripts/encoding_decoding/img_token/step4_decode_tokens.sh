#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:59:00
#SBATCH -J beast_decode_tokens
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/encoding_decoding/img_token/step4_decode_tokens_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 4: decode step3's neurally-estimated beast img_tokens back into reconstructed frames, via
# beast.sable_encoding_decoding.img_token.decode_beast_tokens (the beast analog of Sable's
# camera/Gaussian-splat decode_and_render.py — beast has no camera geometry, so this just runs
# the model's own ViT-MAE decoder + unpatchify). Produces rendered frames under OUT_DIR, consumed
# by step5_generate_video.sh.
#
# Uses "estimated mode" (--estimated-dir): decodes step3's img_tokens_estimated_neuraltrial*.npz
# directly. Those files carry no ids_restore of their own, so --ids-restore-sidecar points at the
# img_tokens_camera_parameters.npz sidecar step1_run_pca_and_save.sh already writes
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=20505751
SPLIT="${SPLIT:-test}"
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/beast_vit_large/$EID/$JOB_ID}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
ESTIMATED_DIR="${ESTIMATED_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/$SPLIT}"
IDS_RESTORE_SIDECAR="${IDS_RESTORE_SIDECAR:-$MODEL_ROOT/img_tokens_camera_parameters.npz}"
DATASET_BASE="${DATASET_BASE:-/work/hdd/bfsr/xdai3/IBL_data/synchronized}"
TARGET_LEFT="${TARGET_LEFT:-$DATASET_BASE/extracted_frames/eval/leftCamera.video/_iblrig_leftCamera.downsampled.$EID}"
TARGET_RIGHT="${TARGET_RIGHT:-$DATASET_BASE/extracted_frames/eval/rightCamera.video/_iblrig_rightCamera.downsampled.$EID}"
OUT_DIR="${OUT_DIR:-$MODEL_ROOT/img_tokens_compressed_estimated/$EID/decode_saved_latents}"

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding beast estimated img_tokens from $ESTIMATED_DIR"

python -m beast.sable_encoding_decoding.img_token.decode_beast_tokens \
    --model-dir "$MODEL_DIR" \
    --estimated-dir "$ESTIMATED_DIR" \
    --ids-restore-sidecar "$IDS_RESTORE_SIDECAR" \
    --target-frame-mapping-left "$TARGET_LEFT" \
    --target-frame-mapping-right "$TARGET_RIGHT" \
    --out-dir "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
