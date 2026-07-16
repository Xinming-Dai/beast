#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-24:00:00
#SBATCH -J sable_cheese3d_train
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/training/train_sable_cheese3d_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
CONFIG="${CONFIG:-$REPO_ROOT/configs/sable/sable_cheese3d_center_camera_holdout.yaml}"

# Data paths (override by exporting before sbatch, e.g.:
#   sbatch --export=ALL,DATASET_PATH=/path/to/cheese3d_cam scripts/train_sable_cheese3d.sh)
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam}"
RESUME_CKPT="${RESUME_CKPT:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/qitaoz--E-RayZer/checkpoints/erayzer_dl3dv.pt}"

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/cheese3d}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${SLURM_JOB_ID}"
    mkdir -p "$CHECKPOINT_DIR"
else
    CHECKPOINT_DIR="$CHECKPOINT_BASE"
fi

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
    "training.reset_training_state=True"
)
[ -n "$RESUME_CKPT" ] && OVERRIDES+=("training.resume_ckpt=$RESUME_CKPT")

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
