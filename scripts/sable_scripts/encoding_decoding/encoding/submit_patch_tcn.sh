#!/bin/bash
# Submit one patch-TCN job per (model, session) pair; each job runs independently.
#
# Usage (run on the login node; this only calls sbatch, no compute):
#   bash scripts/sable_scripts/encoding_decoding/encoding/submit_patch_tcn.sh --dry-run
#   bash scripts/sable_scripts/encoding_decoding/encoding/submit_patch_tcn.sh
# Env overrides: MODELS (space-separated, default "full nogeom dino"),
#   SESSIONS (default "78 72 4b"), FOLD, NUM_PCS, NUM_SAMPLES (forwarded to the job).

set -euo pipefail

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
JOB_SCRIPT="$REPO_ROOT/scripts/sable_scripts/encoding_decoding/encoding/step2_run_patch_tcn.sh"
MODELS="${MODELS:-full nogeom dino}"
SESSIONS="${SESSIONS:-78 72 4b}"

for model in $MODELS; do
    for session in $SESSIONS; do
        cmd=(sbatch -J "patch_tcn_${model}_${session}" "$JOB_SCRIPT" "$model" "$session")
        if [ "$DRY_RUN" = true ]; then
            echo "${cmd[*]}"
        else
            echo "[$model/$session] $("${cmd[@]}")"
        fi
    done
done
