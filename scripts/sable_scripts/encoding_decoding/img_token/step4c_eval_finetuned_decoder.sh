#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:20:00
#SBATCH -J eval_finetuned_decoder
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step4c_eval_finetuned_decoder_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

# # Pre-build the gsplat CUDA extension before rendering starts. Jobs land on
for _cuda in /usr/local/cuda \
             /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8 \
             /opt/cuda; do
    [ -d "$_cuda" ] && export CUDA_HOME="$_cuda" && break
done
export PATH="${CUDA_HOME:-}/bin:${PATH}"

GSPLAT_VER="$(python -c 'import gsplat; print(gsplat.__version__)' 2>/dev/null || echo unknown)"
GSPLAT_KEY="${GSPLAT_VER}_${TORCH_CUDA_ARCH_LIST}"
GSPLAT_CACHE_DIR="$HOME/.cache/gsplat_build"
mkdir -p "$GSPLAT_CACHE_DIR"
GSPLAT_LOCK_DIR="$GSPLAT_CACHE_DIR/${GSPLAT_KEY}.lock"
GSPLAT_DONE_MARKER="$GSPLAT_CACHE_DIR/${GSPLAT_KEY}.done"

if [ ! -f "$GSPLAT_DONE_MARKER" ]; then
    if mkdir "$GSPLAT_LOCK_DIR" 2>/dev/null; then
        echo "[$(date)] Building gsplat CUDA extension (key: $GSPLAT_KEY)..."
        if python -c "from gsplat.cuda._backend import _C"; then
            touch "$GSPLAT_DONE_MARKER"
        else
            echo "WARNING: gsplat pre-build failed; rendering will retry the build itself"
        fi
        rmdir "$GSPLAT_LOCK_DIR"
    else
        echo "[$(date)] Another job is building the gsplat CUDA extension, waiting..."
        for _ in $(seq 1 120); do
            [ -f "$GSPLAT_DONE_MARKER" ] && break
            [ -d "$GSPLAT_LOCK_DIR" ] || break
            sleep 5
        done
    fi
fi

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 4c scores PSNR/SSIM on the test split using a decoder checkpoint that was already
# finetuned by step4b_finetune_decoder.sh (image_token_decoder + upsampler + renderer), without
# rerunning the finetuning pass.

# EID="${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}"
# EID="${EID:-72cb5550-43b4-4ef0-add5-e4adfdfb5e02}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID="${JOB_ID:-21047248}"
MODEL_ROOT="/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID"                  # dir with config.yaml + *best.ckpt
MODEL_DIR="/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID"                 # dir with the base model's config.yaml (fallback source for FINETUNE_DIR/config.yaml)
LR="${LR:-1e-4}"

SUBDIR=latents/img_tokens_compressed/$EID
ESTIMATED_ROOT="$MODEL_ROOT/$SUBDIR/img_tokens_compressed_estimated/$EID"
CAMERA_NPZ="$MODEL_ROOT/$SUBDIR/img_tokens_camera_parameters.npz"
PRECACHED_VIDEO_ROOT="/work/hdd/bfsr/xdai3/IBL_data/synchronized"
USE_MASK=false
SEGMENTATION_ROOT="$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval"

# reuse the already-finetuned checkpoint written by step4b_finetune_decoder.sh
FINETUNE_DIR="$MODEL_ROOT/$SUBDIR/img_tokens_compressed_estimated/$EID/finetuned_decoder_${LR}"
OUT_DIR="$FINETUNE_DIR/decode_saved_latents"
mkdir -p "$OUT_DIR"

FINETUNE_CKPT="$FINETUNE_DIR/finetuned_decoder_best.ckpt"
if [ ! -f "$FINETUNE_CKPT" ]; then
    echo "ERROR: no finetuned checkpoint at $FINETUNE_CKPT; run step4b_finetune_decoder.sh first" >&2
    exit 1
fi
[ -f "$FINETUNE_DIR/config.yaml" ] || cp "$MODEL_DIR/config.yaml" "$FINETUNE_DIR/config.yaml"

if [ "$USE_MASK" = true ]; then
    METRICS_NPZ="$OUT_DIR/psnr_ssim_metrics_masked.npz"
else
    METRICS_NPZ="$OUT_DIR/psnr_ssim_metrics_unmasked.npz"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Scoring test split with finetuned decoder for eid=$EID"

EVAL_ARGS=(
    --z-source "$ESTIMATED_ROOT/test"
    --camera-npz "$CAMERA_NPZ"
    --out-dir "$OUT_DIR"
    --model-dir "$FINETUNE_DIR"
    --dataset-path "$PRECACHED_VIDEO_ROOT/extracted_frames/eval"
    --vda-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/depth_map"
    --correspondence-cache-root "$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences"
    --eid "$EID"
    --batch-size 60
    --include-splits test
    --neural-trial-index 0,1,2,3,4,5,6,7,8,9
    --metrics-npz "psnr_ssim_metrics_10sessions.npz"
    # --metrics-npz "$METRICS_NPZ"
    # --metrics-only
)
[ -n "$VDA_CACHE_ROOT" ] && EVAL_ARGS+=(--vda-cache-root "$VDA_CACHE_ROOT")
[ -n "$CORRESPONDENCE_CACHE_ROOT" ] && EVAL_ARGS+=(--correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT")
[ "$USE_MASK" = true ] && EVAL_ARGS+=(--use-segmentation-mask --segmentation-root "$SEGMENTATION_ROOT")

python -m beast.sable_encoding_decoding.render.decode_and_render "${EVAL_ARGS[@]}"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done scoring test with finetuned decoder for eid=$EID"
conda deactivate
