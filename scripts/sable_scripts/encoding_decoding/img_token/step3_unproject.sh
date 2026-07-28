#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="pca_decompress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:20:00
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
#   sbatch --export=ALL,EID=...,LATENT_ROOT=... \
#     scripts/sable_scripts/encoding_decoding/img_token/step3_unproject.sh
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
MODEL_ROOT=/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/781b35fd-e1f0-4d14-b2bb-95b7263082bb/20014553/latents
LATENT_ROOT=$MODEL_ROOT/img_tokens_compressed/$EID
DECODING_NPY=$LATENT_ROOT/decoding_results_img_tokens_compressed.npy
PCA_NPZ=$LATENT_ROOT/img_tokens_pca_joint.npz
COMPRESSED_TRIALS_NPZ=$LATENT_ROOT/img_tokens_compressed_trials.npz
OUT_ROOT=$LATENT_ROOT/img_tokens_compressed_estimated

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Unprojecting decoded img tokens for eid=$EID"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --decoding-npy "$DECODING_NPY" \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done unprojecting decoded img tokens for eid=$EID"

conda deactivate
