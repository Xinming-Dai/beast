#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-24:00:00
#SBATCH -J sable_ibl
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/training/train_sable_ibl3d_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
CONFIG="${CONFIG:-$REPO_ROOT/configs/sable/sable_ibl3d.yaml}"

# Data paths (override by exporting before sbatch, e.g.:
#   sbatch --export=ALL,EID=<session-id>,DATASET_PATH=/path/to/frames scripts/train_sable_ibl3d.sh)
STAGE=finetune
DATASET_ROOT="${DATASET_ROOT:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames_for_eyz/$STAGE}"
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/$STAGE}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
VDA_CACHE_ROOT="${VDA_CACHE_ROOT:-$DATASET_ROOT/depth_map}"
CORRESPONDENCE_CACHE_ROOT="${CORRESPONDENCE_CACHE_ROOT:-$DATASET_ROOT/litpose_correspondences/processed_correspondences}"

# RESUME_CKPT has two uses, selected by RESET_TRAINING_STATE:
#   - fresh init from a pretrained base checkpoint (default): leave RESUME_CKPT at its
#     default and RESET_TRAINING_STATE=True (default) — only model weights are loaded,
#     optimizer/scheduler/step-count/dataloader position all start from scratch.
#   - true resume of an interrupted job (e.g. after hitting the 24h wall): pass
#     RESUME_CKPT=<checkpoint_dir>/tb_logs/version_0/checkpoints/step=<N>.ckpt,
#     RESET_TRAINING_STATE=False, and RESUME_CKPT_JOB_ID=<original job's SLURM_JOB_ID>
#     (the directory name under CHECKPOINT_BASE holding that checkpoint) — this restores
#     model, optimizer, scheduler, step count, and the train dataloader's position, and
#     reuses the original job's CHECKPOINT_DIR so TensorBoard logs into the same
#     tb_logs/ directory instead of starting a fresh one, e.g.:
#       sbatch --export=ALL,RESUME_CKPT=/path/to/step=10000.ckpt,RESET_TRAINING_STATE=False,RESUME_CKPT_JOB_ID=1234567 \
#         scripts/sable_scripts/training/train_sable_ibl3d.sh
RESUME_CKPT="${RESUME_CKPT:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/qitaoz--E-RayZer/checkpoints/erayzer_dl3dv.pt}"
RESET_TRAINING_STATE="${RESET_TRAINING_STATE:-True}"
RESUME_CKPT_JOB_ID="${RESUME_CKPT_JOB_ID:-}"

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/${EID}}"

if [ "$RESET_TRAINING_STATE" = "False" ]; then
    if [ -z "$RESUME_CKPT_JOB_ID" ]; then
        echo "ERROR: RESET_TRAINING_STATE=False requires RESUME_CKPT_JOB_ID set to the original job's directory name under $CHECKPOINT_BASE"
        exit 1
    fi
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${RESUME_CKPT_JOB_ID}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/${SLURM_JOB_ID}"
else
    CHECKPOINT_DIR="$CHECKPOINT_BASE"
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
Session ID: $EID
VDA cache root: $VDA_CACHE_ROOT
Correspondence cache root: $CORRESPONDENCE_CACHE_ROOT
Resume ckpt: ${RESUME_CKPT:-(none)}
Reset training state: $RESET_TRAINING_STATE
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

OVERRIDES=(
    "training.dataset_path=$DATASET_PATH"
    "training.session_names=$EID"
    "model.vda.cache_root=$VDA_CACHE_ROOT"
    "model.merge_pcd.correspondence_cache_root=$CORRESPONDENCE_CACHE_ROOT"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "training.reset_training_state=$RESET_TRAINING_STATE"
)
[ -n "$RESUME_CKPT" ] && OVERRIDES+=("training.resume_ckpt=$RESUME_CKPT")

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
