#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 0-04:59:00
#SBATCH -J erz_infer_metrics
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/inference/infer_sable_ibl3d_metrics_%j.log
#SBATCH --export=ALL

# Exhaustive PSNR/SSIM evaluation on predicted views: runs inference over every frame of a
# split for the given DATASET_PATH/SESSION_NAMES, applies the segmentation mask to both the
# render and the target before scoring, saves every masked render-only PNG, and computes
# PSNR/SSIM across all of them. No PLY point clouds or GLB scenes are saved.

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
JOB_ID="${JOB_ID:-21047248}"

STAGE=eval
PRECACHED_VIDEO_ROOT="/work/hdd/bfsr/xdai3/IBL_data/synchronized"
DATASET_PATH="${DATASET_PATH:-$PRECACHED_VIDEO_ROOT/extracted_frames/$STAGE}"

# Model dir contains config.yaml saved during training; checkpoints live under tb_logs/
MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID}"

# Separate from infer_sable_ibl3d.sh's default output dir so exhaustive metrics/PNG runs never
# mix with or overwrite the general-purpose inference outputs.
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_DIR/inference_metrics}"

SPLITS="${SPLITS:-test}"
# 0 and 1 are boolean flags
SAVE_VISUALS="${SAVE_VISUALS:-0}"
SAVE_PLY="${SAVE_PLY:-0}"
SAVE_RENDER_VIEWS="${SAVE_RENDER_VIEWS:-1}"
COMPUTE_METRICS="${COMPUTE_METRICS:-1}"
USE_SEGMENTATION_MASK="${USE_SEGMENTATION_MASK:-1}"
SEGMENTATION_ROOT="${SEGMENTATION_ROOT:-$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/$STAGE}"
MAX_BATCHES="${MAX_BATCHES:-}"

# Space-separated override of the sessions to run inference on. Leave unset to use
# every session the model was trained on (training.session_names from config.yaml).
SESSION_NAMES="${SESSION_NAMES:-4b00df29-3769-43be-bb40-128b1cba6d35 72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"

# Unlimited by default (unlike infer_sable_ibl3d.sh's 60-per-session cap) so every frame in
# every session is processed and scored. Once a session hits a quota, later batches for that
# session are skipped entirely (no forward pass, so no metrics either) — set a number here only
# if you want a partial/quick run.
MAX_FILES_PER_SESSION="${MAX_FILES_PER_SESSION-}"

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
Model dir:             $MODEL_DIR
Dataset path:          $DATASET_PATH
Output dir:            $OUTPUT_DIR
Splits:                $SPLITS
Session names:         ${SESSION_NAMES:-(all sessions from saved training config)}
Max files per session: ${MAX_FILES_PER_SESSION:-(unlimited)}
Save visuals:           $SAVE_VISUALS
Save PLY point clouds:  $SAVE_PLY
Save render views:      $SAVE_RENDER_VIEWS
Compute metrics:        $COMPUTE_METRICS
Use segmentation mask:  $USE_SEGMENTATION_MASK
Segmentation root:      ${SEGMENTATION_ROOT:-(from saved training config)}
Max batches:            ${MAX_BATCHES:-(all)}
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

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting metrics inference..."

[ -d "$MODEL_DIR" ]    || { echo "ERROR: Model dir not found: $MODEL_DIR"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

cd "$REPO_ROOT"

PREDICT_ARGS=(
    --model "$MODEL_DIR"
    --input "$DATASET_PATH"
    --output "$OUTPUT_DIR"
    --splits $SPLITS
)
[ -n "$SESSION_NAMES" ]         && PREDICT_ARGS+=(--session-names $SESSION_NAMES)
[ -n "$MAX_FILES_PER_SESSION" ] && PREDICT_ARGS+=(--max-files-per-session "$MAX_FILES_PER_SESSION")
[ "$SAVE_VISUALS" = "1" ]       && PREDICT_ARGS+=(--save-visuals)
[ "$SAVE_PLY" != "1" ]          && PREDICT_ARGS+=(--no-save-pointclouds)
[ "$SAVE_RENDER_VIEWS" = "1" ]  && PREDICT_ARGS+=(--save-render-views)
[ "$COMPUTE_METRICS" = "1" ]    && PREDICT_ARGS+=(--compute-metrics)
if [ "$USE_SEGMENTATION_MASK" = "1" ]; then
    PREDICT_ARGS+=(--use-segmentation-mask)
    [ -n "$SEGMENTATION_ROOT" ] && PREDICT_ARGS+=(--segmentation-root "$SEGMENTATION_ROOT")
fi
[ -n "$MAX_BATCHES" ]           && PREDICT_ARGS+=(--max-batches "$MAX_BATCHES")

beast predict "${PREDICT_ARGS[@]}"

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Done."

conda deactivate
