#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-11:59:00
#SBATCH -J pca_cheese3d
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/pca_scripts/training/train_pca_cheese3d_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/pca_cheese3d.yaml}"

# Data path (override by exporting before sbatch, e.g.:
#   sbatch --export=ALL,DATASET_PATH=/path/to/cheese3d_cam scripts/pca_scripts/training/train_pca_cheese3d.sh)
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam}"
# training sessions, matching configs/pca_cheese3d.yaml's data.session_names
SESSIONS=(
    20231031_B20_chew_bl_000
    20231031_B21_chew_bl_000
    20231031_B26_chew_bl_000
    20231031_B31_chew_bl_000
    20231031_B6_chew_temperature_000
    20231031_B20_chew_temperature_000
    20231031_B21_chew_temperature_000
    20231031_B26_chew_temperature_000
    20231031_B31_chew_temperature_000
)

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/cheese3d}"

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

[ -f "$CONFIG" ] || { echo "ERROR: Config not found: $CONFIG"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

PCA_INIT="$CHECKPOINT_DIR/pca_init.pkl"

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Fitting initial PCA subspace -> $PCA_INIT"

# fit an incremental PCA on the training images to initialize the autoencoder's mean/components,
# so gradient descent starts from a reasonable subspace instead of a random one
beast fit-pca \
    --data-dir "$DATASET_PATH" \
    --session-names "${SESSIONS[@]}" \
    --n-components 768 \
    --batch-size 768 \
    --output "$PCA_INIT"

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting training..."

OVERRIDES=(
    "data.data_dir=$DATASET_PATH"
    "model.model_params.pca_pickle_path=$PCA_INIT"
)

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
