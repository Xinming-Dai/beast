#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="pca_compress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 120G
#SBATCH -t 0-02:00:00
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step1_run_pca_and_save_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 2 runs in two sub-stages via --stage:
# 1 fits PCA on the train split only, 2 projects val/test using stage 1's fitted PCA
# (--stage all runs both in one pass). Requires step0's img_tokens_batch*.npz shards as INPUT_DIR,
# laid out as INPUT_DIR/<session_name>/<train|val|test>/...; each session is fit and applied
# independently (no pooling across sessions). By default every session subfolder under INPUT_DIR
# is auto-discovered and processed; set SESSION_NAMES to restrict to specific ones. Produces, per
# session, img_tokens_pca_joint.npz + img_tokens_compressed_trials.npz under
# MODEL_ROOT/img_tokens_compressed/<session_name>/, which step2_run_decoding.sh's
# LATENT_INPUT_DIR consumes.
#
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,INPUT_DIR=...,MODEL_ROOT=... \
#     scripts/sable_scripts/encoding_decoding/img_token/step1_run_pca_and_save.sh
MODEL_ROOT="${MODEL_ROOT:-<MODEL_ROOT>}"                    # output paths
INPUT_DIR="${INPUT_DIR:-<MODEL_ROOT>/img_tokens}"           # inference dir of img_tokens_batch*.npz shards
STAGE="${STAGE:-all}"                                       # 1 | 2 | all
N_FEAT_KEEP="${N_FEAT_KEEP:-6}"                             # PCA components to keep
SESSION_NAMES="${SESSION_NAMES:-}"                          # space-separated session/EID names; empty = auto-discover all

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running img-token PCA fit/apply, stage=$STAGE"

ARGS=(
    --input-dir "$INPUT_DIR"
    --model-root "$MODEL_ROOT"
    --stage "$STAGE"
    --n-feat-keep "$N_FEAT_KEEP"
)
[ -n "$SESSION_NAMES" ] && ARGS+=(--session-names $SESSION_NAMES)

python -m beast.sable_encoding_decoding.img_token.run_pca_and_save "${ARGS[@]}"

conda deactivate
