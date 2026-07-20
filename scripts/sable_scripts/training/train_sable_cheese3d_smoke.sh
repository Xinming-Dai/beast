#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:30:00
#SBATCH -J sable_cheese3d_smoke
#SBATCH -o scripts/sable_scripts/training/train_sable_cheese3d_smoke_%j.log
#SBATCH --export=ALL

# 30-minute smoke test of Cheese3D training with the default 2-view settings
# (configs/sable/sable_cheese3d.yaml). Only deviation from the default config: VDA
# online mode is pointed at a locally-cloned Video-Depth-Anything repo + checkpoint,
# because the internal xdai3 VDA paths are permission-blocked for other accounts.

exec 2>&1
source ~/.bashrc
conda activate "${CONDA_ENV:-sable}"

# Resolve repo root without hardcoding a user account: explicit REPO_ROOT override wins,
# then the sbatch submit dir (run `sbatch` from the repo root), then this script's location.
if [ -z "${REPO_ROOT:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    fi
fi
CONFIG="${CONFIG:-$REPO_ROOT/configs/sable/sable_cheese3d.yaml}"

# Data paths (override by exporting before sbatch).
DATASET_PATH="${DATASET_PATH:-/work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam}"
RESUME_CKPT="${RESUME_CKPT:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/qitaoz--E-RayZer/checkpoints/erayzer_dl3dv.pt}"

# Locally-cloned public Video-Depth-Anything (default config points at blocked xdai3 paths).
VDA_REPO_ROOT="${VDA_REPO_ROOT:-/work/nvme/bfsr/${USER}/vda/Video-Depth-Anything}"
VDA_CKPT="${VDA_CKPT:-$VDA_REPO_ROOT/checkpoints/video_depth_anything_vitb.pth}"

CHECKPOINT_BASE="${CHECKPOINT_DIR:-/work/nvme/bfsr/${USER}/project3d/twoview3d_ckpts/cheese3d_smoke}"

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
VDA repo root: $VDA_REPO_ROOT
VDA checkpoint: $VDA_CKPT
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

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting smoke test..."

[ -f "$CONFIG" ] || { echo "ERROR: Config not found: $CONFIG"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }
[ -f "$VDA_REPO_ROOT/video_depth_anything/video_depth.py" ] || { echo "ERROR: VDA repo not found: $VDA_REPO_ROOT"; exit 1; }
[ -f "$VDA_CKPT" ] || { echo "ERROR: VDA checkpoint not found: $VDA_CKPT"; exit 1; }

cd "$REPO_ROOT"

# Restrict to sessions whose TL+TR segmentation masks are readable under this account
# (some of the default config's sessions have group-denied mask subdirs owned by xdai3).
# Override to skip via SESSION_NAMES="[...]" if access changes.
SESSION_NAMES="${SESSION_NAMES:-[20231031_B21_chew_bl_000,20231031_B6_chew_temperature_000,20231031_B21_chew_temperature_000,20231031_B26_chew_temperature_000,20231031_B31_chew_temperature_000]}"

# Build overrides list; only include resume_ckpt when it is set.
OVERRIDES=(
    "training.dataset_path=$DATASET_PATH"
    "training.checkpoint_dir=$CHECKPOINT_DIR"
    "training.reset_training_state=True"
    "training.session_names=$SESSION_NAMES"
    "model.vda.repo_root=$VDA_REPO_ROOT"
    "model.vda.checkpoint_path=$VDA_CKPT"
    # the 30-min SLURM wall clock bounds this smoke test; keep total steps above
    # optimizer.warmup (3000) so OneCycleLR pct_start = warmup/max_fwdbwd_passes stays < 1.
    # checkpoint often so we capture a saved checkpoint before the wall clock cuts it off.
    "training.max_fwdbwd_passes=${MAX_FWDBWD_PASSES:-20000}"
    "training.checkpoint_every=${CHECKPOINT_EVERY:-200}"
)

# opt-in: render the held-out front (TC) camera as novel-view synthesis at val time,
# saved under <checkpoint_dir>/visuals/front_nvs/. Enable with RENDER_FRONT_NVS=1.
[ "${RENDER_FRONT_NVS:-0}" = "1" ] && OVERRIDES+=("training.render_front_nvs.enabled=true")
[ -n "$RESUME_CKPT" ] && OVERRIDES+=("training.resume_ckpt=$RESUME_CKPT")

beast train \
    --config "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --overrides "${OVERRIDES[@]}"

conda deactivate
