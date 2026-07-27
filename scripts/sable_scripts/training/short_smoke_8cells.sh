#!/bin/bash
# Short smoke test for all 8 cells (200 steps each).
#
# This is Phase 5 of the loss-weighting ablation plan: each cell runs 200 steps
# with the same overridden loss weights as the full ablation, but with a small
# step budget so we can confirm all 8 cells initialize, train, and save
# checkpoints correctly before committing to 10k-step runs.
#
# Usage:
#   bash short_smoke_8cells.sh
#
# Override ROOT to redirect output (e.g. ROOT=/path/to/_smoke/8cells).

set -euo pipefail

ROOT=${ROOT:-/cephfs/jinqihang/SABLE/outputs/loss_weighting/_smoke/8cells}
ABLATION_STEPS=${ABLATION_STEPS:-200}
ABLATION_WARMUP=${ABLATION_WARMUP:-20}
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-24}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

mkdir -p "${ROOT}"

# Delegate to the shared launcher and only override the step budgets.
ROOT="${ROOT}" \
ABLATION_STEPS="${ABLATION_STEPS}" \
ABLATION_WARMUP="${ABLATION_WARMUP}" \
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
bash /cephfs/jinqihang/SABLE/beast/scripts/sable_scripts/training/ablation_l2_percept_geom.sh
