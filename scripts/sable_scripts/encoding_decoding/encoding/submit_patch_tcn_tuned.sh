#!/bin/bash
# Submit the tuned patch-TCN protocol: one tune job per model on the development session, then
# one eval job per (model, session) that starts after that model's tune job succeeds.
#
# Usage (run on the login node; this only calls sbatch, no compute):
#   bash scripts/sable_scripts/encoding_decoding/encoding/submit_patch_tcn_tuned.sh --dry-run
#   bash scripts/sable_scripts/encoding_decoding/encoding/submit_patch_tcn_tuned.sh
#   bash .../submit_patch_tcn_tuned.sh --eval-only   # locked configs already exist
# Flags: --dry-run (print the sbatch commands), --tune-only, --eval-only.
# Env overrides: MODELS (default "full nogeom dino"), SESSIONS (eval sessions, default
#   "78 72 4b"); FOLD, NUM_SAMPLES, NUM_PCS_LIST are forwarded to the jobs.

set -euo pipefail

DRY_RUN=false
RUN_TUNE=true
RUN_EVAL=true
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --tune-only) RUN_EVAL=false ;;
        --eval-only) RUN_TUNE=false ;;
        *) echo "unknown flag: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="/u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding"
MODELS="${MODELS:-full nogeom dino}"
SESSIONS="${SESSIONS:-78 72 4b}"

submit() {
    # prints the job id (or a placeholder on a dry run)
    if [ "$DRY_RUN" = true ]; then
        echo "DRYRUN: sbatch $*" >&2
        echo "<jobid>"
    else
        sbatch --parsable "$@"
    fi
}

for model in $MODELS; do
    dependency=()
    if [ "$RUN_TUNE" = true ]; then
        tune_id=$(submit -J "patch_tcn_tune_${model}" "$SCRIPT_DIR/step2_tune_patch_tcn.sh" "$model")
        echo "[$model/tune] job $tune_id"
        dependency=(--dependency="afterok:${tune_id}")
    fi
    if [ "$RUN_EVAL" = true ]; then
        for session in $SESSIONS; do
            eval_id=$(submit "${dependency[@]}" -J "patch_tcn_eval_${model}_${session}" \
                "$SCRIPT_DIR/step2_eval_patch_tcn.sh" "$model" "$session")
            echo "[$model/$session/eval] job $eval_id"
        done
    fi
done
