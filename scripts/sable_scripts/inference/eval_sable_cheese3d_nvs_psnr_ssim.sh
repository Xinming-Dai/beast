#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 0-00:30:00
#SBATCH -J sable_cheese3d_nvs_eval
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/inference/eval_sable_cheese3d_nvs_psnr_ssim_%j.log
#SBATCH --export=ALL

# Per-frame PSNR/SSIM of the anchored novel TC view on every held-out frame of the val split,

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
CKPT_ROOT="/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/cheese3d"

# 20156493: 2-view TL/TR model (TC has no predicted pose, so the anchor is derived from the GT rig).
# 20030003_center_view_holdout: TC held out; scored with HELD_OUT_TC=masked (strict holdout).
MODEL_DIRS="${MODEL_DIRS:-$CKPT_ROOT/20156493 $CKPT_ROOT/20030003_center_view_holdout}"
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam}"
# space-separated session names; empty = every session in the saved training config
SESSIONS="${SESSIONS:-}"
SPLIT="${SPLIT:-val}"
# BC is not needed: the similarity fit uses the model's input cameras and TC is the scored view
CAMERAS="${CAMERAS:-TL TR TC}"
# empty = cameras the model took as input (held-out center view excluded)
ALIGN_CAMERAS="${ALIGN_CAMERAS:-}"
# per-model default: <model dir>/nvs_eval_black_bg
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
# none = fp32, bf16 = autocast like training
AMP="${AMP:-none}"
CONTEXT_BATCH="${CONTEXT_BATCH:-1}"
# camera that is placed from the anchor pose and scored
ANCHOR_CAMERA="${ANCHOR_CAMERA:-TC}"
# auto = the model's predicted pose when it has one, derived = GT TC mapped by the TL/TR fit
ANCHOR_SOURCE="${ANCHOR_SOURCE:-auto}"
INTRINSICS="${INTRINSICS:-gt}"
# center_camera_holdout models: masked = strict holdout (tokens zeroed, TC's Gaussians dropped),
# visible = TC left visible, both = each under <output>/tc_*
HELD_OUT_TC="${HELD_OUT_TC:-masked}"
HIDE_HELD_OUT_FROM_ENCODERS="${HIDE_HELD_OUT_FROM_ENCODERS:-0}"
# render background; GT images are not masked
BG_COLOR="${BG_COLOR:-black}"

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

for _cuda in /usr/local/cuda \
             /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8 \
             /opt/cuda; do
    [ -d "$_cuda" ] && export CUDA_HOME="$_cuda" && break
done
export PATH="${CUDA_HOME:-}/bin:${PATH}"
export PYTHONUNBUFFERED=1

cat <<EOF
---------------------------------------
Job name:              ${SLURM_JOB_NAME:-local}
Job ID:                ${SLURM_JOB_ID:-local}
Running on node(s):    ${SLURM_NODELIST:-$(hostname)}
Model dirs:            $MODEL_DIRS
Dataset path:          $DATASET_PATH
Sessions:              ${SESSIONS:-(all sessions in training config)}
Split:                 $SPLIT
Cameras:               $CAMERAS
Align cameras:         ${ALIGN_CAMERAS:-(model input cameras)}
Output root:           ${OUTPUT_ROOT:-(<model dir>/nvs_eval_black_bg)}
AMP / context batch:   $AMP / $CONTEXT_BATCH
Anchor camera:         $ANCHOR_CAMERA (source $ANCHOR_SOURCE)
Held-out TC variants:  $HELD_OUT_TC (hide from encoders: $HIDE_HELD_OUT_FROM_ENCODERS)
Background:            $BG_COLOR
TORCH_CUDA_ARCH_LIST:  $TORCH_CUDA_ARCH_LIST
---------------------------------------
EOF

[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

cd "$REPO_ROOT"

DONE_DIRS=()
for MODEL_DIR in $MODEL_DIRS; do
    echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Scoring every $SPLIT frame for $MODEL_DIR"
    [ -d "$MODEL_DIR" ] || { echo "ERROR: Model dir not found: $MODEL_DIR"; continue; }

    if [ -n "$OUTPUT_ROOT" ]; then
        OUTPUT_DIR="$OUTPUT_ROOT/$(basename "$MODEL_DIR")"
    else
        OUTPUT_DIR="$MODEL_DIR/nvs_eval_black_bg"
    fi

    EVAL_ARGS=(
        --model "$MODEL_DIR"
        --input "$DATASET_PATH"
        --output "$OUTPUT_DIR"
        --split "$SPLIT"
        --cameras $CAMERAS
        --amp "$AMP"
        --context-batch "$CONTEXT_BATCH"
        --intrinsics "$INTRINSICS"
        --anchor-camera "$ANCHOR_CAMERA"
        --anchor-source "$ANCHOR_SOURCE"
        --held-out-tc "$HELD_OUT_TC"
        --bg-color "$BG_COLOR"
    )
    [ "$HIDE_HELD_OUT_FROM_ENCODERS" = "1" ] && EVAL_ARGS+=(--hide-held-out-from-encoders)
    [ -n "$SESSIONS" ]       && EVAL_ARGS+=(--sessions $SESSIONS)
    [ -n "$ALIGN_CAMERAS" ]  && EVAL_ARGS+=(--align-cameras $ALIGN_CAMERAS)

    if python -m beast.sable_encoding_decoding.render.cheese3d_nvs_eval "${EVAL_ARGS[@]}"; then
        DONE_DIRS+=("$OUTPUT_DIR")
    else
        echo "ERROR: NVS eval failed for $MODEL_DIR"
    fi
done

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Done scoring."
if [ "${#DONE_DIRS[@]}" -gt 0 ]; then
    echo "=== Mean and SE over $SPLIT frames, $ANCHOR_CAMERA view, $BG_COLOR background ==="
    python -m beast.sable_encoding_decoding.render.cheese3d_nvs_eval --summarize "${DONE_DIRS[@]}"
fi

conda deactivate
