#!/bin/bash
#SBATCH -A beez-delta-cpu
#SBATCH --job-name="beast_pca_decompress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:20:00
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/encoding_decoding/img_token/step3_unproject_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 3: un-PCA and de-normalize decoded compressed img tokens back to full-dimensional beast
# img tokens. Requires step1's PCA bundle and step2's decoded output; produces per-trial
# img_tokens_estimated*.npz, consumed by step4_decode_tokens.sh as --estimated-dir.
EID="${EID:-f312aaec-3b6f-44b3-86b4-3a0c119c0438}"
JOB_ID=20668699
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/beast_vit_large/$EID/$JOB_ID}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
LATENT_ROOT="${LATENT_ROOT:-$MODEL_DIR/latents}"
DECODING_NPY="${DECODING_NPY:-$LATENT_ROOT/img_tokens_compressed/$EID/decoding_results_img_tokens_compressed.npy}"
PCA_NPZ="${PCA_NPZ:-$MODEL_ROOT/img_tokens_pca_joint.npz}"
COMPRESSED_TRIALS_NPZ="${COMPRESSED_TRIALS_NPZ:-$MODEL_ROOT/img_tokens_compressed_trials.npz}"
OUT_ROOT="${OUT_ROOT:-$MODEL_ROOT/img_tokens_compressed_estimated}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Unprojecting decoded beast img tokens for eid=$EID"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --decoding-npy "$DECODING_NPY" \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
