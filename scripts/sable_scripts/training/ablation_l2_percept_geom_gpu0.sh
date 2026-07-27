#!/bin/bash
# Loss-weighting ablation — GPU 0 half (4 cells, perceptual weight sweep + baseline).
#
# Each cell varies only the three loss weights (l2, perceptual, gs_reg); all other
# hyperparameters (optimizer, schedule, data, VDA precomputed, batch, steps, warmup)
# are held constant via the frozen contract in run_one_cell.sh.
#
# This script pins CUDA_VISIBLE_DEVICES=0 and runs 4 cells SEQUENTIALLY on a single GPU.
# Run in parallel with ablation_l2_percept_geom_gpu1.sh on a second shell.
#
# Usage:
#   bash ablation_l2_percept_geom_gpu0.sh

set -euo pipefail

ROOT=${ROOT:-/cephfs/jinqihang/SABLE/outputs/loss_weighting}
ABLATION_STEPS=${ABLATION_STEPS:-10000}
# Warmup scaled to match plan v4 §6.3 pct_start = 0.15 (Mia's reference ratio):
#   20000 steps -> 3000 warmup; 10000 steps -> 1500 warmup.
ABLATION_WARMUP=${ABLATION_WARMUP:-1500}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-24}     # effective batch 24
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
GPU_ID=${GPU_ID:-0}

# This half's 4 cells: perceptual-weight sweep around default.
CELLS=(
  "default     l2=1.0 p=0.3 g=1.0"
  "no-percept  l2=1.0 p=0.0 g=1.0"
  "low-percept l2=1.0 p=0.1 g=1.0"
  "high-percept l2=1.0 p=1.0 g=1.0"
)

if (( ABLATION_WARMUP < 2 || ABLATION_WARMUP >= ABLATION_STEPS )); then
  echo "ERROR: OneCycleLR requires 2 <= ABLATION_WARMUP < ABLATION_STEPS; " \
       "got warmup=${ABLATION_WARMUP}, steps=${ABLATION_STEPS}." >&2
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
  echo "============================================="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    bash /cephfs/jinqihang/SABLE/beast/scripts/sable_scripts/training/run_one_cell.sh \
      "${out}" "${OVERRIDES[@]}" >"${out}/launcher.log" 2>&1
}

# Run 4 cells sequentially on GPU 0. A failure in one cell surfaces immediately
# rather than being swallowed by parallelism.
for spec in "${CELLS[@]}"; do
  launch_spec "${spec}"
done

echo "GPU ${GPU_ID}: all ${#CELLS[@]} cells finished."
