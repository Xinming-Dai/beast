#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:15:00
#SBATCH -J sable_cheese3d_nvs
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/inference/infer_sable_cheese3d_nvs_%j.log
#SBATCH --export=ALL

# Novel-view synthesis on one frame of each Cheese3D session
# Per session/frame_<idx>/: nvs_grid.png, nvs_metrics.json (PSNR/SSIM, intrinsics, pose error) and glb/ scenes

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
CKPT_ROOT="/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/cheese3d"

# Model dirs contain config.yaml saved during training; checkpoints live under tb_logs/
MODEL_DIRS="${MODEL_DIRS:-$CKPT_ROOT/20030003_center_view_holdout $CKPT_ROOT/22338245_3_view_reconsturction $CKPT_ROOT/20156493}"
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam}"
# space-separated session names; empty = every session in the saved training config
SESSIONS="${SESSIONS:-}"
FRAME_IDX="${FRAME_IDX:-}" # empty = first frame of each session in SPLIT
SPLIT="${SPLIT:-val}"
CAMERAS="${CAMERAS:-TL TR TC BC}"
# empty = cameras the model took as input (held-out center view excluded)
ALIGN_CAMERAS="${ALIGN_CAMERAS:-}"
# per-model default: <model dir>/nvs
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
# none = fp32, bf16 = autocast like training
AMP="${AMP:-none}"
# records collated per forward pass (training validated with 16); the requested frame is sample 0
CONTEXT_BATCH="${CONTEXT_BATCH:-1}"
# how the projected row places cameras: anchor (predicted poses where the model has them, the rest
# placed relative to ANCHOR_CAMERA via the GT rig) or global_fit (one similarity fit of all GT
# cameras); metrics for both are always written to nvs_metrics.json
POSE_MODE="${POSE_MODE:-anchor}"
ANCHOR_CAMERA="${ANCHOR_CAMERA:-TC}"
# auto = the model's predicted pose when it has one, derived = GT TC mapped by the TL/TR fit
ANCHOR_SOURCE="${ANCHOR_SOURCE:-auto}"
# intrinsics of the global_fit renders: gt, or pred_scaled (model intrinsics, focal-rescaled for
# cameras the model does not have)
INTRINSICS="${INTRINSICS:-gt}"
SAVE_GLB="${SAVE_GLB:-1}"


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
Frame idx:             ${FRAME_IDX:-(first in split)}
Split:                 $SPLIT
Cameras:               $CAMERAS
Align cameras:         ${ALIGN_CAMERAS:-(model input cameras)}
Output root:           ${OUTPUT_ROOT:-(<model dir>/nvs)}
AMP / context batch:   $AMP / $CONTEXT_BATCH
Pose mode (row 2):     $POSE_MODE (anchor $ANCHOR_CAMERA, source $ANCHOR_SOURCE)
Global-fit intrinsics: $INTRINSICS
Save glb:              $SAVE_GLB
TORCH_CUDA_ARCH_LIST:  $TORCH_CUDA_ARCH_LIST
---------------------------------------
EOF

echo '=== GPU (PyTorch) ==='
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}  capability={cap[0]}.{cap[1]}")
PY

[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

cd "$REPO_ROOT"

for MODEL_DIR in $MODEL_DIRS; do
    echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] NVS for $MODEL_DIR"
    [ -d "$MODEL_DIR" ] || { echo "ERROR: Model dir not found: $MODEL_DIR"; continue; }

    if [ -n "$OUTPUT_ROOT" ]; then
        OUTPUT_DIR="$OUTPUT_ROOT/$(basename "$MODEL_DIR")"
    else
        OUTPUT_DIR="$MODEL_DIR/nvs"
    fi

    NVS_ARGS=(
        --model "$MODEL_DIR"
        --input "$DATASET_PATH"
        --output "$OUTPUT_DIR"
        --split "$SPLIT"
        --cameras $CAMERAS
        --amp "$AMP"
        --context-batch "$CONTEXT_BATCH"
        --intrinsics "$INTRINSICS"
        --pose-mode "$POSE_MODE"
        --anchor-camera "$ANCHOR_CAMERA"
        --anchor-source "$ANCHOR_SOURCE"
    )
    [ "$SAVE_GLB" != "1" ]   && NVS_ARGS+=(--no-save-glb)
    [ -n "$SESSIONS" ]       && NVS_ARGS+=(--sessions $SESSIONS)
    [ -n "$FRAME_IDX" ]      && NVS_ARGS+=(--frame-idx "$FRAME_IDX")
    [ -n "$ALIGN_CAMERAS" ]  && NVS_ARGS+=(--align-cameras $ALIGN_CAMERAS)

    python -m beast.sable_encoding_decoding.render.cheese3d_nvs "${NVS_ARGS[@]}" \
        || echo "ERROR: NVS failed for $MODEL_DIR"

    if [ "$SAVE_GLB" = "1" ]; then
        echo "glb scenes under $OUTPUT_DIR: $(find "$OUTPUT_DIR" -name '*.glb' 2>/dev/null | wc -l)"
        find "$OUTPUT_DIR" -name '*.glb' 2>/dev/null | sort | head -4
    fi
done

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Done."

conda deactivate
