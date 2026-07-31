#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:20:00
#SBATCH -J finetune_decoder
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step4b_finetune_decoder_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 4b finetunes SABLE's image-token decoder (image_token_decoder + upsampler + renderer)
# on the val split's neural-decoded (CNN-predicted) img tokens against real images, then scores
# PSNR/SSIM on the test split with the finetuned weights. Mirrors the original E-RayZer
# step5_erayzer_decoder_finetune.sh. 
MODEL_ROOT=/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_multisession/20503395                   # dir with config.yaml + *best.ckpt
EID=4b00df29-3769-43be-bb40-128b1cba6d35
LR="1e-4"

SUBDIR=latents/img_tokens_compressed/$EID
ESTIMATED_ROOT="$MODEL_ROOT/$SUBDIR/img_tokens_compressed_estimated/$EID"
CAMERA_NPZ="$MODEL_ROOT/$SUBDIR/img_tokens_camera_parameters.npz"
PRECACHED_VIDEO_ROOT="/work/hdd/bfsr/xdai3/IBL_data/synchronized"
USE_MASK=true
SEGMENTATION_ROOT="$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval"

# finetune decoder on val split for one epoch
FINETUNE_DIR="$MODEL_ROOT/$SUBDIR/img_tokens_compressed_estimated/$EID/finetuned_decoder_${LR}"
OUT_DIR="$FINETUNE_DIR/decode_saved_latents"
mkdir -p "$FINETUNE_DIR" "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Finetuning decoder on val split for eid=$EID"

FINETUNE_ARGS=(
    --z-source "$ESTIMATED_ROOT/val"
    --camera-npz "$CAMERA_NPZ"
    --out-dir "$OUT_DIR"
    --model-dir "$MODEL_ROOT"
    --dataset-path "$PRECACHED_VIDEO_ROOT/extracted_frames/eval"
    --vda-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/depth_map"
    --correspondence-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences"
    --batch-size 60
    --include-splits val
    --finetune-ckpt-out "$FINETUNE_DIR/finetuned_decoder_best.ckpt"
    --finetune-lr "$LR"
)
[ -n "$VDA_CACHE_ROOT" ] && FINETUNE_ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && FINETUNE_ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")

python -m beast.sable_encoding_decoding.render.decode_and_render "${FINETUNE_ARGS[@]}"

# decoding and rendering with finetuned decoder on test split
cp "$MODEL_ROOT/config.yaml" "$FINETUNE_DIR/config.yaml"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Scoring test split with finetuned decoder for eid=$EID"

EVAL_ARGS=(
    --z-source "$ESTIMATED_ROOT/test"
    --camera-npz "$CAMERA_NPZ"
    --out-dir "$OUT_DIR"
    --model-dir "$FINETUNE_DIR"
    --dataset-path "$PRECACHED_VIDEO_ROOT/extracted_frames/eval"
    --vda-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/depth_map"
    --correspondence-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences"
    --batch-size 60
    --include-splits test
    --metrics-only
)
[ -n "$VDA_CACHE_ROOT" ] && EVAL_ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && EVAL_ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")
[ "$USE_MASK" = true ] && ARGS+=(--use-segmentation-mask --segmentation-root "$SEGMENTATION_ROOT")

python -m beast.sable_encoding_decoding.render.decode_and_render "${EVAL_ARGS[@]}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done finetuning decoder and scoring test for eid=$EID"
conda deactivate
