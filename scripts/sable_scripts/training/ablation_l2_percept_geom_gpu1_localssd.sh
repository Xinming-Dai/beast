#!/bin/bash
# Loss-weighting ablation — GPU 1 half (3 cells: reconstruction + gs_reg sweeps).
# cell_low-recon (10000 steps) completed separately.
#
# Pinned CUDA_VISIBLE_DEVICES=1; run in parallel with ablation_l2_percept_geom_gpu0_localssd.sh.
# Dataset pinned to local NVMe (/localssd/jinqihang/...) — set DATASET_ROOT to override.

set -euo pipefail

ROOT=${ROOT:-/cephfs/jinqihang/SABLE/outputs/loss_weighting}
DATASET_ROOT=${DATASET_ROOT:-/localssd/jinqihang/datasets/beast3d-data/sable_ibl_4b00}
ABLATION_STEPS=${ABLATION_STEPS:-6000}
# Warmup scaled to match plan v4 §6.3 pct_start = 0.15 (Mia's reference ratio):
#   20000 steps -> 3000 warmup; 6000 steps -> 900 warmup.
ABLATION_WARMUP=${ABLATION_WARMUP:-900}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-24}     # effective batch 24
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
GPU_ID=${GPU_ID:-1}

# This half's 3 cells (cell_low-recon completed at 10000 steps): l2 + gs_reg sweeps.
CELLS=(
  "high-recon l2=2.0 p=0.3 g=1.0"
  "no-geom    l2=1.0 p=0.3 g=0.0"
  "high-geom  l2=1.0 p=0.3 g=2.0"
)

if (( ABLATION_WARMUP < 2 || ABLATION_WARMUP >= ABLATION_STEPS )); then
  echo "ERROR: OneCycleLR requires 2 <= ABLATION_WARMUP < ABLATION_STEPS; " \
       "got warmup=${ABLATION_WARMUP}, steps=${ABLATION_STEPS}." >&2
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: DATASET_ROOT not found: ${DATASET_ROOT}" >&2
  echo "  Run rsync first: rsync -a /cephfs/jinqihang/SABLE/datasets/beast3d-data/sable_ibl_4b00/ ${DATASET_ROOT}/" >&2
  exit 1
fi
if [[ ! -d "${DATASET_ROOT}/vda_cache" ]] || [[ ! -d "${DATASET_ROOT}/litpose_correspondences/processed_correspondences" ]]; then
  echo "ERROR: DATASET_ROOT is missing expected subdirs (vda_cache / litpose_correspondences/processed_correspondences)" >&2
  exit 1
fi

mkdir -p "${ROOT}"

launch_spec() {
  local spec="$1"
  local name rest kv key val l2 p g out
  name=$(echo "${spec}" | awk '{print $1}')
  rest="${spec#* }"
  for kv in ${rest}; do
    key="${kv%%=*}"
    val="${kv#*=}"
    case "${key}" in
      l2) l2="${val}" ;;
      p)  p="${val}" ;;
      g)  g="${val}" ;;
    esac
  done
  out="${ROOT}/cell_${name}"
  if [[ -d "${out}" ]] && [[ -n "$(find "${out}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      && [[ "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
    echo "ERROR: refusing to reuse non-empty cell directory: ${out}" >&2
    return 1
  fi
  mkdir -p "${out}"
  OVERRIDES=(
    "training.l2_loss_weight=${l2}"
    "training.perceptual_loss_weight=${p}"
    "training.gs_reg_loss_weight=${g}"
    "training.batch_size_per_gpu=${BATCH_SIZE_PER_GPU}"
    "training.grad_accum_steps=${GRAD_ACCUM_STEPS}"
    "training.max_fwdbwd_passes=${ABLATION_STEPS}"
    "training.checkpoint_every=${ABLATION_STEPS}"
    "training.save_val_best_checkpoint=false"
    "optimizer.warmup=${ABLATION_WARMUP}"
  )
  echo "============================================="
  echo "Launching cell '${name}' at ${out} on physical GPU ${GPU_ID}"
  echo "  l2=${l2}  perceptual=${p}  gs_reg=${g}"
  echo "  steps=${ABLATION_STEPS}  warmup=${ABLATION_WARMUP}"
  echo "  batch=${BATCH_SIZE_PER_GPU}  grad_accum=${GRAD_ACCUM_STEPS}"
  echo "  dataset: ${DATASET_ROOT}  (local NVMe)"
  echo "============================================="
  DATASET_PATH="${DATASET_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    bash /cephfs/jinqihang/SABLE/beast/scripts/sable_scripts/training/run_one_cell.sh \
      "${out}" "${OVERRIDES[@]}" >"${out}/launcher.log" 2>&1
}

for spec in "${CELLS[@]}"; do
  launch_spec "${spec}"
done

echo "GPU ${GPU_ID}: all ${#CELLS[@]} cells finished."
