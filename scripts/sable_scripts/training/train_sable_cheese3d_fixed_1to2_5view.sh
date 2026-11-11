#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-11:59:00
#SBATCH -J cheese3d_fixed_1to2_5view
#SBATCH -o /u/xdai3/project3d/SABLE_repo_2/beast/scripts/sable_scripts/training/train_sable_cheese3d_fixed_1to2_5view_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_2/beast"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/sable/sable_cheese3d_fixed_1to2_5view.yaml}"

# Data paths (override by exporting before sbatch, e.g.:
#   sbatch --export=ALL,DATASET_PATH=/path/to/cheese3d_cam scripts/train_sable_cheese3d_fixed_1to2_5view.sh)
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam}"

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/cheese3d}"
RESET_TRAINING_STATE="${RESET_TRAINING_STATE:-True}"
# how GT camera calibration is used: none | init (learnable init of the pose head) | gt (as input);
# ignored when resuming, which reuses the saved config
CAMERA_PRIOR="${CAMERA_PRIOR:-none}"
case "$CAMERA_PRIOR" in
    none|init|gt) ;;
    *) echo "ERROR: CAMERA_PRIOR must be none, init or gt (got: $CAMERA_PRIOR)"; exit 1 ;;
esac
RESUME_CKPT_JOB_ID="${RESUME_CKPT_JOB_ID:-}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/qitaoz--E-RayZer/checkpoints/erayzer_dl3dv.pt}"

if [ "$RESET_TRAINING_STATE" = "False" ]; then
    if [ -z "$RESUME_CKPT_JOB_ID" ]; then
        echo "ERROR: RESET_TRAINING_STATE=False requires RESUME_CKPT_JOB_ID set to the original run's directory name under $CHECKPOINT_BASE"
        exit 1
    fi
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RESUME_CKPT_JOB_ID}"
    [ -d "$CHECKPOINT_DIR" ] || { echo "ERROR: Resume directory not found: $CHECKPOINT_DIR"; exit 1; }
    CONFIG="$CHECKPOINT_DIR/config.yaml"
    if [ -z "${RESUME_CKPT:-}" ]; then
        # highest step=N.ckpt across tb_logs/version_*/checkpoints (excludes *-best.ckpt, which
        # is the best-val checkpoint rather than the latest state)
        RESUME_CKPT=$(find "$CHECKPOINT_DIR"/tb_logs -path '*/checkpoints/step=*.ckpt' 2>/dev/null \
            | sed -E 's/.*step=([0-9]+)\.ckpt$/\1 &/' | sort -n | tail -n 1 | cut -d' ' -f2-)
    fi
    [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ] || { echo "ERROR: No step=*.ckpt to resume from under $CHECKPOINT_DIR"; exit 1; }
else
    RESUME_CKPT="${RESUME_CKPT:-$PRETRAINED_CKPT}"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        CHECKPOINT_DIR="${CHECKPOINT_BASE}/${SLURM_JOB_ID}_fixed_1to2_5view${CAMERA_PRIOR:+_camera_prior_$CAMERA_PRIOR}"
    else
        CHECKPOINT_DIR="$CHECKPOINT_BASE"
    fi
fi
mkdir -p "$CHECKPOINT_DIR"

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
Job name: ${SLURM_JOB_NAME:-local}
Job ID: ${SLURM_JOB_ID:-local}
Running on node(s): ${SLURM_NODELIST:-$(hostname)}
Config: $CONFIG
Dataset path: $DATASET_PATH
Resume ckpt: ${RESUME_CKPT:-(none)}
Reset training state: $RESET_TRAINING_STATE
Camera prior: $CAMERA_PRIOR
Checkpoint dir (output): $CHECKPOINT_DIR
TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST
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

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting training..."

[ -f "$CONFIG" ] || { echo "ERROR: Config not found: $CONFIG"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

cd "$REPO_ROOT"

# Build overrides list; only include resume_ckpt when it is set.
OVERRIDES=(
    "training.dataset_path=$DATASET_PATH"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "training.reset_training_state=$RESET_TRAINING_STATE"
)
[ -n "$RESUME_CKPT" ] && OVERRIDES+=("training.resume_ckpt=$RESUME_CKPT")
[ "$RESET_TRAINING_STATE" != "False" ] && OVERRIDES+=("training.camera_prior=$CAMERA_PRIOR")

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
