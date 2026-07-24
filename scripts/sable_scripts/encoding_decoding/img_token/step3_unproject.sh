#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-05:59:00
#SBATCH -J sable_unproject
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step3_unproject_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 4 un-PCA and de-normalize decoded compressed img tokens back to full-dimensional img tokens. 
# Requires step1's PCA bundle (PCA_NPZ) and step2's decoded output (DECODING_NPY); 
# produces per-trial img_tokens_estimated*.npz, consumed by step4_decode_and_render.sh as --z-source.
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,EID=...,MODEL_ROOT=... \
#     scripts/sable_scripts/encoding_decoding/img_token/step3_unproject.sh
EID="${EID:-<SESSION_ID>}"
MODEL_ROOT="${MODEL_ROOT:-<MODEL_ROOT>}"
LATENT_ROOT="${LATENT_ROOT:-<LATENT_ROOT>}"
DECODING_NPY="${DECODING_NPY:-$LATENT_ROOT/img_tokens_compressed/$EID/decoding_results_img_tokens_compressed.npy}"
PCA_NPZ="${PCA_NPZ:-$MODEL_ROOT/img_tokens_compressed/img_tokens_pca_joint.npz}"
COMPRESSED_TRIALS_NPZ="${COMPRESSED_TRIALS_NPZ:-$MODEL_ROOT/img_tokens_compressed/img_tokens_compressed_trials.npz}"
OUT_ROOT="${OUT_ROOT:-$MODEL_ROOT/img_tokens_compressed_estimated}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Unprojecting decoded img tokens for eid=$EID"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --decoding-npy "$DECODING_NPY" \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT"

conda deactivate
