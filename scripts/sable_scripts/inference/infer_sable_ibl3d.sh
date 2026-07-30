#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:10:00
#SBATCH -J erz_infer_multisession
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/inference/infer_sable_ibl3d_multisession_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
JOB_ID="${JOB_ID:-20434515}"

STAGE=finetune
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/$STAGE}"

# Model dir contains config.yaml saved during training; checkpoints live under tb_logs/
MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_multisession/$JOB_ID}"

OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_DIR/inference}"

SPLITS="${SPLITS:-val}"
SAVE_VISUALS="${SAVE_VISUALS:-1}"
SAVE_GLB="${SAVE_GLB:-1}"
MAX_BATCHES="${MAX_BATCHES:-}"

# Space-separated override of the sessions to run inference on. Leave unset to use
# every session the model was trained on (training.session_names from config.yaml).
SESSION_NAMES="${SESSION_NAMES:-}"

# Max PLY + GLB files saved per session; outputs are grouped under ply/<session>/ and
# glb/<session>/ so it's clear which session each file came from.
MAX_FILES_PER_SESSION="${MAX_FILES_PER_SESSION:-10}"

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
Max files per session: $MAX_FILES_PER_SESSION
Save visuals:           $SAVE_VISUALS
Save GLB scenes:        $SAVE_GLB
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

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting inference..."

[ -d "$MODEL_DIR" ]    || { echo "ERROR: Model dir not found: $MODEL_DIR"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

cd "$REPO_ROOT"

PREDICT_ARGS=(
    --model "$MODEL_DIR"
    --input "$DATASET_PATH"
    --output "$OUTPUT_DIR"
    --splits $SPLITS
    --max-files-per-session "$MAX_FILES_PER_SESSION"
)
[ -n "$SESSION_NAMES" ]  && PREDICT_ARGS+=(--session-names $SESSION_NAMES)
[ "$SAVE_VISUALS" = "1" ] && PREDICT_ARGS+=(--save-visuals)
[ "$SAVE_GLB" = "1" ]     && PREDICT_ARGS+=(--save-camera-pointcloud-scene)
[ -n "$MAX_BATCHES" ]     && PREDICT_ARGS+=(--max-batches "$MAX_BATCHES")

beast predict "${PREDICT_ARGS[@]}"

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Done."

conda deactivate
