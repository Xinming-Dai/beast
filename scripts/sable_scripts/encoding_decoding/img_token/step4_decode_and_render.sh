#!/bin/bash
#SBATCH -A bezq-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:20:00
#SBATCH -J decode_and_render
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step4_decode_and_render_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 4 feeds (real or decoded) img tokens through SABLE's decoder to reconstruct images, save point clouds, and/or compute PSNR/SSIM.
# Requires step3's unprojected tokens (or raw step0 img tokens) as Z_SOURCE; produces
# rendered frames under OUT_DIR, consumed by step5_generate_video.sh.
MODEL_ROOT=/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_multisession/20503395                   # dir with config.yaml + *best.ckpt
EID=4b00df29-3769-43be-bb40-128b1cba6d35

SUBDIR=latents/img_tokens_compressed/$EID
Z_SOURCE="$MODEL_ROOT/$SUBDIR/img_tokens_compressed_estimated/$EID/test"     # single .npz or a directory of img_tokens*.npz
OUT_DIR="$MODEL_ROOT/$SUBDIR/img_tokens_compressed_estimated/$EID/decode_saved_latents"
PRECACHED_VIDEO_ROOT="/work/hdd/bfsr/xdai3/IBL_data/synchronized"
USE_MASK=true
SEGMENTATION_ROOT="$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval"

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding + rendering img tokens from $Z_SOURCE"

ARGS=(
    --z-source "$Z_SOURCE"
    --camera-npz "$MODEL_ROOT/$SUBDIR/img_tokens_camera_parameters.npz"
    --out-dir "$OUT_DIR"
    --model-dir "$MODEL_ROOT"
    --dataset-path "$PRECACHED_VIDEO_ROOT/extracted_frames/eval"
    --vda-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/depth_map"
    --correspondence-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences"
    --batch-size 60
    --include-splits test
    --neural-trial-index 0
    --eid "$EID"
)
[ -n "$VDA_CACHE_ROOT" ] && ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")
[ "$USE_MASK" = true ] && ARGS+=(--use-segmentation-mask --segmentation-root "$SEGMENTATION_ROOT")

python -m beast.sable_encoding_decoding.render.decode_and_render "${ARGS[@]}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done decoding and rendering img tokens for eid=$EID"
conda deactivate
