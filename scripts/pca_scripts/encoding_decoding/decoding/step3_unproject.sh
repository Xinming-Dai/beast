#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="pca_pca_decompress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 5G
#SBATCH -t 0-00:20:00
#SBATCH -o /u/xdai3/project3d/SABLE_repo_3/beast/scripts/pca_scripts/encoding_decoding/decoding/step3_unproject_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 3: un-PCA and de-normalize decoded compressed PCA-autoencoder frame latents back to
# full-dimensional (n_components=768) latents. Requires step1's PCA bundle and step2's decoded
# output; produces per-trial img_tokens_estimated*.npz (z shape (1, T, 2, 768)), consumed by
# step4_decode_pca_latents.sh. This code path is generic over L/D, so it is identical to the
# resnet and beast step3_unproject.sh aside from paths.
# EID="${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}"
# EID="${EID:-72cb5550-43b4-4ef0-add5-e4adfdfb5e02}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=21810401
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/restricted_sessions/$JOB_ID}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
LATENT_ROOT="${LATENT_ROOT:-$MODEL_DIR/latents}"
DECODING_NPY="${DECODING_NPY:-$LATENT_ROOT/img_tokens_compressed/$EID/decoding_results_img_tokens_compressed.npy}"
PCA_NPZ="${PCA_NPZ:-$MODEL_ROOT/img_tokens_pca_joint.npz}"
COMPRESSED_TRIALS_NPZ="${COMPRESSED_TRIALS_NPZ:-$MODEL_ROOT/img_tokens_compressed_trials.npz}"
OUT_ROOT="${OUT_ROOT:-$MODEL_ROOT/img_tokens_compressed_estimated}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Unprojecting decoded PCA frame latents for eid=$EID"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --decoding-npy "$DECODING_NPY" \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
