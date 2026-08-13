#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="beast_pca_compress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 120G
#SBATCH -t 0-00:59:00
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/encoding_decoding/img_token/step1_run_pca_and_save_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 1: PCA-compress beast img_tokens. Like Sable's own img_token step1
# (scripts/sable_scripts/encoding_decoding/img_token/step1_run_pca_and_save.sh),
# scripts/beast_scripts/encoding_decoding/encoding/step1_beast_latent.sh now shards output
# across many img_tokens_batch*.npz files under --input-dir
# (INPUT_DIR/<session_name>/<train|val|test>/...), so this step reads them the same way via
# --input-dir instead of --combined-trials-*-npz. run_pca_and_save.py itself is unchanged.
# Produces img_tokens_pca_joint.npz + img_tokens_compressed_trials.npz under MODEL_ROOT,
# consumed by step2_run_decoding.sh's LATENT_INPUT_DIR.
EID="${EID:-f312aaec-3b6f-44b3-86b4-3a0c119c0438}"
JOB_ID=20668699
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/beast_vit_large/$EID/$JOB_ID}"
INPUT_DIR="${INPUT_DIR:-$MODEL_DIR/latents/img_tokens}"     # root of img_tokens_batch*.npz shards
# no /$EID suffix here: --session-names below already selects INPUT_DIR/$EID as the per-session
# input, and run_pca_and_save.py's --input-dir loop appends session_name once under this anchor
# (mirrors scripts/sable_scripts/encoding_decoding/img_token/step1_run_pca_and_save.sh)
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents}"
STAGE="${STAGE:-all}"                                       # 1 | 2 | all
N_FEAT_KEEP=6                                               # PCA components to keep

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running beast img-token PCA fit/apply, stage=$STAGE"

python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \
    --input-dir "$INPUT_DIR" \
    --session-names "$EID" \
    --model-root "$MODEL_ROOT" \
    --stage "$STAGE" \
    --n-feat-keep "$N_FEAT_KEEP"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
