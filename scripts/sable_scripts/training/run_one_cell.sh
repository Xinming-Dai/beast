#!/bin/bash
# Run one ablation cell. Sets up environment and launches `beast train` from
# the cell directory so that tb_logs/, ModelCheckpoint default dirpath, and
# the CWD-relative VGG lookup all land inside $CELL_DIR.
#
# Usage:
#   bash run_one_cell.sh <CELL_DIR> [extra --overrides KEY=VALUE ...]
#
# Examples:
#   bash run_one_cell.sh /data/jqh/Outputs/beast/rebuttal/loss_weighting/cell_default
#   bash run_one_cell.sh /data/jqh/Outputs/.../cell_no_percept \
#       training.l2_loss_weight=1.0 training.perceptual_loss_weight=0.0 training.gs_reg_loss_weight=1.0

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <CELL_DIR> [extra overrides KEY=VALUE ...]" >&2
    exit 1
fi

CELL_DIR="$1"; shift
EXTRA_OVERRIDES=("$@")

# Dry-run mode: resolve the environment, verify the VDA checkpoint hash, verify
# VGG symlink, and print the full training command without launching it.
DRY_RUN=${DRY_RUN:-0}
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] cell=${CELL_DIR}"
  # Still verify the VDA checkpoint hash so silent weight drift is caught.
  VDA_CKPT_DRY="/home/jqh/NeuralWorkshops/third_party/VDA/checkpoints/video_depth_anything_vitb.pth"
  VDA_EXPECTED_SHA_DRY="775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b"
  ACTUAL_SHA_DRY="$(sha256sum "${VDA_CKPT_DRY}" | awk '{print $1}')"
  if [[ "${ACTUAL_SHA_DRY}" != "${VDA_EXPECTED_SHA_DRY}" ]]; then
    echo "ERROR: VDA checkpoint SHA256 mismatch: expected ${VDA_EXPECTED_SHA_DRY}, got ${ACTUAL_SHA_DRY}" >&2
    exit 1
  fi
  echo "[DRY-RUN] VDA SHA256 ✓"
  VGG_DRY="/data/jqh/pretrained_checkpoints/beast/metric_checkpoint/imagenet-vgg-verydeep-19.mat"
  VGG_EXPECTED_SHA_DRY="abdb57167f82a2a1fbab1e1c16ad9373411883f262a1a37ee5db2e6fb0044695"
  ACTUAL_VGG_SHA_DRY="$(sha256sum "${VGG_DRY}" | awk '{print $1}')"
  if [[ "${ACTUAL_VGG_SHA_DRY}" != "${VGG_EXPECTED_SHA_DRY}" ]]; then
    echo "ERROR: VGG checkpoint SHA256 mismatch: expected ${VGG_EXPECTED_SHA_DRY}, got ${ACTUAL_VGG_SHA_DRY}" >&2
    exit 1
  fi
  echo "[DRY-RUN] VGG SHA256 ✓"
  echo "[DRY-RUN] would mkdir -p ${CELL_DIR}/metric_checkpoint"
  echo "[DRY-RUN] would symlink imagenet-vgg-verydeep-19.mat"
  echo "[DRY-RUN] would run: python -m beast.cli.main train --config sable_ibl3d.yaml --output ${CELL_DIR} --overrides ..."
  echo "[DRY-RUN] extra overrides: ${EXTRA_OVERRIDES[@]:-(none)}"
  exit 0
fi

# ---- Path-independent config (frozen contract) ----
REPO_ROOT="/home/jqh/NeuralWorkshops/beast"
BASE_CONFIG="${BASE_CONFIG:-${REPO_ROOT}/configs/sable/sable_ibl3d.yaml}"
EID="4b00df29-3769-43be-bb40-128b1cba6d35"
DATASET_PATH="/data/jqh/Datasets/beast3d-data/sable_ibl_4b00"
CORR_ROOT="${DATASET_PATH}/litpose_correspondences/processed_correspondences"
RESUME_CKPT="/data/jqh/pretrained_checkpoints/E-RayZer-private/checkpoints/erayzer_dl3dv.pt"
VDA_CKPT="/home/jqh/NeuralWorkshops/third_party/VDA/checkpoints/video_depth_anything_vitb.pth"
VDA_REPO_ROOT="/home/jqh/NeuralWorkshops/third_party/VDA"
VGG_SRC="/data/jqh/pretrained_checkpoints/beast/metric_checkpoint/imagenet-vgg-verydeep-19.mat"

# ---- Pinned VDA checkpoint SHA256 (online VDA contract) ----
# Hard-pinned SHA256. If the file drifts, abort.
VDA_EXPECTED_SHA="775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b"
VGG_EXPECTED_SHA="abdb57167f82a2a1fbab1e1c16ad9373411883f262a1a37ee5db2e6fb0044695"
RESUME_EXPECTED_SHA="56fd798d831e9c2a300932e5c41ead29a4ef3a084480c5bfc31d58ba3a834463"

# Isolate local beast from PyPI beast 2.1.0 (astronomy package collision).
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/home/jqh/miniconda3/envs/neuro/bin:${PATH}"
export HF_HOME="/home/jqh/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Per-cell working directory (so tb_logs/, checkpoint defaults, VGG symlink all land here).
mkdir -p "${CELL_DIR}"
cd "${CELL_DIR}"

mkdir -p metric_checkpoint
# Symlink (not copy) the 511 MB VGG matconvnet weights to avoid duplicating it per cell.
ln -sf "${VGG_SRC}" metric_checkpoint/imagenet-vgg-verydeep-19.mat

# Sanity-check the VDA checkpoint before launching.
ACTUAL_SHA="$(sha256sum "${VDA_CKPT}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA}" != "${VDA_EXPECTED_SHA}" ]]; then
    echo "ERROR: VDA checkpoint SHA256 mismatch: expected ${VDA_EXPECTED_SHA}, got ${ACTUAL_SHA}" >&2
    exit 1
fi
ACTUAL_VGG_SHA="$(sha256sum "${VGG_SRC}" | awk '{print $1}')"
if [[ "${ACTUAL_VGG_SHA}" != "${VGG_EXPECTED_SHA}" ]]; then
    echo "ERROR: VGG checkpoint SHA256 mismatch: expected ${VGG_EXPECTED_SHA}, got ${ACTUAL_VGG_SHA}" >&2
    exit 1
fi
ACTUAL_RESUME_SHA="$(sha256sum "${RESUME_CKPT}" | awk '{print $1}')"
if [[ "${ACTUAL_RESUME_SHA}" != "${RESUME_EXPECTED_SHA}" ]]; then
    echo "ERROR: resume checkpoint SHA256 mismatch: expected ${RESUME_EXPECTED_SHA}, got ${ACTUAL_RESUME_SHA}" >&2
    exit 1
fi

# Conda and CUDA setup (mirror train_sable_ibl3d.sh; harmless if not on cluster).
[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++
for _cuda in /usr/local/cuda \
             /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8 \
             /opt/cuda; do
    [ -d "$_cuda" ] && export CUDA_HOME="$_cuda" && break
done
export PATH="${CUDA_HOME:-}/bin:${PATH}"
export PYTHONUNBUFFERED=1
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]]; then
    unset TORCH_CUDA_ARCH_LIST
fi

# Sanity: confirm PYTHONPATH resolves to the local beast, not the PyPI astronomy package.
RESOLVED_BEAST="$(python3 -c 'import sys, beast; print(beast.__file__)' 2>/dev/null || true)"
if [[ -z "${RESOLVED_BEAST}" ]]; then
    echo "ERROR: cannot import beast. PYTHONPATH=${PYTHONPATH}" >&2
    exit 1
fi
if [[ "${RESOLVED_BEAST}" != "${REPO_ROOT}"* ]]; then
    echo "ERROR: beast resolves to ${RESOLVED_BEAST}, expected ${REPO_ROOT}/*" >&2
    exit 1
fi

echo "----------------------------------------"
echo "Cell: ${CELL_DIR}"
echo "Base config: ${BASE_CONFIG}"
echo "EID: ${EID}"
echo "Online VDA checkpoint: ${VDA_CKPT}"
echo "Online VDA checkpoint SHA256: ${VDA_EXPECTED_SHA}"
echo "VGG symlink: ${CELL_DIR}/metric_checkpoint/imagenet-vgg-verydeep-19.mat"
echo "VGG checkpoint SHA256: ${VGG_EXPECTED_SHA}"
echo "Resume ckpt: ${RESUME_CKPT}"
echo "Resume checkpoint SHA256: ${RESUME_EXPECTED_SHA}"
echo "Beast import: ${RESOLVED_BEAST}"
echo "Extra overrides: ${EXTRA_OVERRIDES[@]:-(none)}"
echo "----------------------------------------"

# Cell-level + frozen online-VDA overrides. Order matters for tie-breaking
# (later overrides win), so cell-level overrides come last via EXTRA_OVERRIDES.
BASE_OVERRIDES=(
    "training.dataset_path=${DATASET_PATH}"
    "training.session_names=${EID}"
    "training.checkpoint_dir=${CELL_DIR}"
    "training.resume_ckpt=${RESUME_CKPT}"
    "training.reset_training_state=True"
    "model.vda.mode=online"
    "model.vda.cache_root=null"
    "model.vda.encoder=vitb"
    "model.vda.metric=false"
    "model.vda.repo_root=${VDA_REPO_ROOT}"
    "model.vda.checkpoint_path=${VDA_CKPT}"
    "model.merge_pcd.correspondence_cache_root=${CORR_ROOT}"
)

# Keep the process rooted in the cell directory. PerceptualLoss, TensorBoard,
# and Lightning's default checkpoint path are all CWD-relative.
cd "${CELL_DIR}"
python -m beast.cli.main train \
    --config "${BASE_CONFIG}" \
    --output "${CELL_DIR}" \
    --overrides "${BASE_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}"
