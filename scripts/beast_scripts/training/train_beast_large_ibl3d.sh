#!/bin/bash
#SBATCH -A beez-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:10:00
#SBATCH -J beast_large_ibl
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/training/train_beast_large_ibl3d_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
CONFIG="${CONFIG:-$REPO_ROOT/configs/vit_large.yaml}"

# Data paths (override by exporting before sbatch, e.g.:
#   sbatch --export=ALL,DATASET_PATH=/path/to/frames scripts/beast_scripts/training/train_beast_large_ibl3d.sh)
STAGE=finetune
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/$STAGE}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_vit_large/$EID}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${SLURM_JOB_ID}"
    mkdir -p "$CHECKPOINT_DIR"
else
    CHECKPOINT_DIR="$CHECKPOINT_BASE"
fi

export PYTHONUNBUFFERED=1

cat <<EOF
---------------------------------------
Job name: ${SLURM_JOB_NAME:-local}
Job ID: ${SLURM_JOB_ID:-local}
Running on node(s): ${SLURM_NODELIST:-$(hostname)}
Config: $CONFIG
Dataset path: $DATASET_PATH
Session ID: $EID
Checkpoint dir (output): $CHECKPOINT_DIR
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

OVERRIDES=(
    "data.data_dir=$DATASET_PATH"
    "data.session_names=$EID"
)

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
