#!/bin/bash
# Submit the full dependency chain for the mean-appearance-baseline rebuttal experiment, one chain
# per session (chains run in parallel), then one summary job that waits for all of them.
#
# Per session:
#   step1  PCA + camera sidecar (CPU)                    -> img_tokens_compressed/<eid>/
#   step2  neural decoding (GPU)            after step1  -> decoding_results_img_tokens_compressed.npy
#   step3  unproject val,test (CPU)         after step2
#   step4b finetune on val, score test (GPU) after step3  -> neural-decoded psnr_ssim_metrics.npz
#   step3b mean-token baseline (CPU)        after step1  (only needs the PCA bundle)
#   step4d score mean tokens (GPU)          after step3b and step4b (reuses the finetuned ckpt)
# Finally: summarize_mean_baseline.py (CPU) after every step4b and step4d job.
#
# step0 is skipped: img_tokens already exist under <JOB_ID>/latents/img_tokens for these sessions.
#
# Usage (run on the login node; this only calls sbatch, no compute):
#   bash scripts/sable_scripts/encoding_decoding/img_token/submit_mean_baseline_chain.sh --dry-run
#   bash scripts/sable_scripts/encoding_decoding/img_token/submit_mean_baseline_chain.sh
# Env overrides: JOB_ID, EIDS (space-separated), GPU_ACCOUNT, CPU_ACCOUNT.

set -euo pipefail

DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
SCRIPT_DIR="$REPO_ROOT/scripts/sable_scripts/encoding_decoding/img_token"
JOB_ID="${JOB_ID:-21047248}"
EIDS="${EIDS:-4b00df29-3769-43be-bb40-128b1cba6d35 72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
GPU_ACCOUNT="${GPU_ACCOUNT:-bezq-delta-gpu}"
CPU_ACCOUNT="${CPU_ACCOUNT:-bfsr-delta-cpu}"

DRY_COUNT=0

# submit <account> <dependency-or-empty> <export-vars> <sbatch args...>; sets JOB to the job id
# (a global rather than stdout, so the dry-run counter survives and no subshell is needed)
submit() {
    local account="$1" dep="$2" export_vars="$3"
    shift 3
    local args=(--parsable -A "$account" --export="ALL,$export_vars")
    [ -n "$dep" ] && args+=(--dependency="afterok:$dep")
    if [ "$DRY_RUN" = true ]; then
        DRY_COUNT=$((DRY_COUNT + 1))
        echo "sbatch ${args[*]} $*" >&2
        JOB="DRY$DRY_COUNT"
        return
    fi
    local out
    out="$(sbatch "${args[@]}" "$@")"
    JOB="${out%%;*}"  # --parsable prints "<id>[;<cluster>]"
}

join_deps() {
    local IFS=:
    echo "$*"
}

all_final_jobs=()
for eid in $EIDS; do
    env_vars="EID=$eid,JOB_ID=$JOB_ID"

    submit "$CPU_ACCOUNT" "" "$env_vars,SESSION_NAMES=$eid" "$SCRIPT_DIR/step1_run_pca_and_save.sh"; s1=$JOB
    submit "$GPU_ACCOUNT" "$s1" "$env_vars" "$SCRIPT_DIR/step2_run_decoding.sh"; s2=$JOB
    submit "$CPU_ACCOUNT" "$s2" "$env_vars" "$SCRIPT_DIR/step3_unproject.sh"; s3=$JOB
    submit "$GPU_ACCOUNT" "$s3" "$env_vars" "$SCRIPT_DIR/step4b_finetune_decoder.sh"; s4b=$JOB
    submit "$CPU_ACCOUNT" "$s1" "$env_vars" "$SCRIPT_DIR/step3b_unproject_mean_baseline.sh"; s3b=$JOB
    submit "$GPU_ACCOUNT" "$(join_deps "$s3b" "$s4b")" "$env_vars" "$SCRIPT_DIR/step4d_eval_mean_baseline.sh"; s4d=$JOB

    echo "[$eid] step1=$s1 -> step2=$s2 -> step3=$s3 -> step4b=$s4b"
    echo "[$eid] step1=$s1 -> step3b=$s3b -> step4d=$s4d (also after step4b=$s4b)"
    all_final_jobs+=("$s4b" "$s4d")
done

submit "$CPU_ACCOUNT" "$(join_deps "${all_final_jobs[@]}")" "JOB_ID=$JOB_ID" \
    -p cpu -c 1 --mem 4G -t 0-00:10:00 -J summarize_mean_baseline \
    -o "$SCRIPT_DIR/summarize_mean_baseline_%j.log" \
    --wrap "source ~/.bashrc; conda activate beast; python $SCRIPT_DIR/summarize_mean_baseline.py --job-id $JOB_ID --eids $EIDS"
summary=$JOB
echo "summary=$summary (after: ${all_final_jobs[*]})"
