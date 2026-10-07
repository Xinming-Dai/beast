#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="pca_decompress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:59:00
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
# EID="${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}"
# EID="${EID:-72cb5550-43b4-4ef0-add5-e4adfdfb5e02}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID="${JOB_ID:-21047248}"
MODEL_ROOT="/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID/latents"
LATENT_ROOT=$MODEL_ROOT/img_tokens_compressed/$EID
DECODING_NPY=$LATENT_ROOT/decoding_results_img_tokens_compressed.npy
PCA_NPZ=$LATENT_ROOT/img_tokens_pca_joint.npz
COMPRESSED_TRIALS_NPZ=$LATENT_ROOT/img_tokens_compressed_trials.npz
OUT_ROOT=$LATENT_ROOT/img_tokens_compressed_estimated
INCLUDE_SPLITS="${INCLUDE_SPLITS:-val,test}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Unprojecting decoded img tokens for eid=$EID splits=$INCLUDE_SPLITS"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --decoding-npy "$DECODING_NPY" \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT" \
    --include-splits "$INCLUDE_SPLITS"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done unprojecting decoded img tokens for eid=$EID"

conda deactivate
