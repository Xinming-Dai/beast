#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH -t 0-11:59:00
#SBATCH -J pca_ibl_restricted
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/pca_scripts/training/train_pca_ibl3d_restricted_sessions_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/pca_ibl3d_restricted_sessions.yaml}"

# Data paths (override by exporting before sbatch, e.g.:
#   sbatch --export=ALL,DATASET_PATH=/path/to/frames scripts/pca_scripts/training/train_pca_ibl3d_restricted_sessions.sh)
STAGE=finetune
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/$STAGE}"

# session set matches configs/sable/sable_ibl3d_pretrain_restricted_sessions.yaml
EIDS=(
    "9b528ad0-4599-4a55-9148-96cc1d93fb24"
    "51e53aff-1d5d-4182-a684-aba783d50ae5"
    "0802ced5-33a3-405e-8336-b65ebc5cb07c"
    "88224abb-5746-431f-9c17-17d7ef806e6a"
    "a4a74102-2af5-45dc-9e41-ef7f5aed88be"
    "ecb5520d-1358-434c-95ec-93687ecd1396"
    "f312aaec-3b6f-44b3-86b4-3a0c119c0438"
)

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/restricted_sessions}"

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
Session IDs: ${EIDS[*]}
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

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Fitting initial PCA subspace on ${#EIDS[@]} sessions -> $PCA_INIT"

# fit an incremental PCA on the training images (pooled across all restricted sessions) to
# initialize the autoencoder's mean/components, so gradient descent starts from a reasonable
# subspace instead of a random one
beast fit-pca \
    --data-dir "$DATASET_PATH" \
    --session-names "${EIDS[@]}" \
    --n-components 768 \
    --batch-size 768 \
    --output "$PCA_INIT"

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting training..."

# build a YAML flow-sequence string, e.g. [id1,id2,id3], since apply_config_overrides()
# parses each override value with yaml.safe_load and does not split on commas
SESSION_NAMES_OVERRIDE="[$(IFS=,; echo "${EIDS[*]}")]"

OVERRIDES=(
    "data.data_dir=$DATASET_PATH"
    "data.session_names=$SESSION_NAMES_OVERRIDE"
    "model.model_params.pca_pickle_path=$PCA_INIT"
)

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
