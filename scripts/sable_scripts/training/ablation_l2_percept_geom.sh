#!/bin/bash
# Stage all 8 cells of the loss-weighting ablation.
#
# Each cell varies only the three loss weights (l2, perceptual, gs_reg); all other
# hyperparameters (optimizer, schedule, data, online VDA, batch, steps, warmup) are
# held constant via the frozen contract in run_one_cell.sh.
#
# Steps and warmup are tunable via env so the same script can be used for the
# Phase 5 short smoke (200 steps / 20 warmup) and the Phase 6 full ablation
# (10000 steps / 1000 warmup).
#
# Usage:
#   ABLATION_STEPS=200 ABLATION_WARMUP=20 bash short_smoke_8cells.sh
#   ABLATION_STEPS=10000 ABLATION_WARMUP=1000 bash ablation_l2_percept_geom.sh

set -euo pipefail

ROOT=${ROOT:-/data/jqh/Outputs/beast/rebuttal/loss_weighting}
# Run cells under ROOT/cell_<name>/. If ROOT already contains a smoke subdir,
# callers can override ROOT to point at a different parent.
RUNS=(
  "default    l2=1.0 p=0.3 g=1.0"
  "no-percept l2=1.0 p=0.0 g=1.0"
  "low-percept l2=1.0 p=0.1 g=1.0"
  "high-percept l2=1.0 p=1.0 g=1.0"
  "low-recon  l2=0.5 p=0.3 g=1.0"
  "high-recon l2=2.0 p=0.3 g=1.0"
  "no-geom    l2=1.0 p=0.3 g=0.0"
  "high-geom  l2=1.0 p=0.3 g=2.0"
)

ABLATION_STEPS=${ABLATION_STEPS:-10000}
ABLATION_WARMUP=${ABLATION_WARMUP:-1000}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-12}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
GPU_IDS=${GPU_IDS:-0,1}

if (( ABLATION_WARMUP < 2 || ABLATION_WARMUP >= ABLATION_STEPS )); then
  echo "ERROR: OneCycleLR requires 2 <= ABLATION_WARMUP < ABLATION_STEPS; " \
       "got warmup=${ABLATION_WARMUP}, steps=${ABLATION_STEPS}." >&2
  exit 1
fi

FORMAL_ROOT=/data/jqh/Outputs/beast/rebuttal/loss_weighting
FORMAL_STEPS=10000
if (( ABLATION_STEPS < FORMAL_STEPS )) && [[ "${ROOT}" == "${FORMAL_ROOT}" ]] \
    && [[ "${ALLOW_SHORT_IN_FORMAL_ROOT:-0}" != "1" ]]; then
  echo "ERROR: refusing to write a short run (${ABLATION_STEPS} steps) into the formal root ${ROOT}." >&2
  echo "Use short_smoke_8cells.sh or set an explicit ROOT under _smoke/." >&2
  exit 1
fi

mkdir -p "${ROOT}"

launch_spec() {
  local spec="$1"
  local gpu_id="$2"
  local name rest kv key val l2 p g out
  name=$(echo "${spec}" | awk '{print $1}')
  # Parse "l2=X p=Y g=Z" into bash variables. Each spec has exactly 3 KV pairs,
  # so we strip the name and split on spaces.
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
    "model.vda.mode=online"
  )
  echo "============================================="
  echo "Launching cell '${name}' at ${out} on physical GPU ${gpu_id}"
  echo "  l2=${l2}  perceptual=${p}  gs_reg=${g}"
  echo "  steps=${ABLATION_STEPS}  warmup=${ABLATION_WARMUP}"
  echo "  batch=${BATCH_SIZE_PER_GPU}  grad_accum=${GRAD_ACCUM_STEPS}"
  echo "============================================="
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    bash /home/jqh/NeuralWorkshops/beast/scripts/sable_scripts/training/run_one_cell.sh \
      "${out}" "${OVERRIDES[@]}" >"${out}/launcher.log" 2>&1
}

IFS=',' read -r -a GPU_ARRAY <<<"${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_IDS must contain at least one GPU index" >&2
  exit 1
fi

# Launch one cell per GPU in waves. Wait for every cell in a wave so failures
# are surfaced before scheduling more full runs.
for ((start = 0; start < ${#RUNS[@]}; start += ${#GPU_ARRAY[@]})); do
  pids=()
  names=()
  for ((offset = 0; offset < ${#GPU_ARRAY[@]} && start + offset < ${#RUNS[@]}; offset++)); do
    spec="${RUNS[start + offset]}"
    gpu_id="${GPU_ARRAY[offset]}"
    name=$(echo "${spec}" | awk '{print $1}')
    launch_spec "${spec}" "${gpu_id}" &
    pids+=("$!")
    names+=("${name}")
  done

  wave_failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[i]}"; then
      echo "ERROR: cell '${names[i]}' failed; see ${ROOT}/cell_${names[i]}/launcher.log" >&2
      wave_failed=1
    fi
  done
  if [[ "${wave_failed}" == "1" ]]; then
    exit 1
  fi
done

echo "All ${#RUNS[@]} cells finished."
