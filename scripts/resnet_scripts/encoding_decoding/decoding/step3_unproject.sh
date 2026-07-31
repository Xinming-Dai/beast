#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="resnet_pca_decompress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:20:00
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/resnet_scripts/encoding_decoding/decoding/step3_unproject_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 3: un-PCA and de-normalize decoded compressed resnet frame latents back to
# full-dimensional (768) latents. Requires step1's PCA bundle and step2's decoded output;
# produces per-trial img_tokens_estimated*.npz (z shape (1, T, 2, 768)), consumed by
# step4_decode_resnet_latents.sh. This code path is generic over L/D, so it is identical to
# beast's own step3_unproject.sh aside from paths.
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=20505763
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/resnet_ae_152/$EID/$JOB_ID}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents/img_tokens_compressed/$EID}"
LATENT_ROOT="${LATENT_ROOT:-$MODEL_DIR/latents}"
DECODING_NPY="${DECODING_NPY:-$LATENT_ROOT/img_tokens_compressed/$EID/decoding_results_img_tokens_compressed.npy}"
PCA_NPZ="${PCA_NPZ:-$MODEL_ROOT/img_tokens_pca_joint.npz}"
COMPRESSED_TRIALS_NPZ="${COMPRESSED_TRIALS_NPZ:-$MODEL_ROOT/img_tokens_compressed_trials.npz}"
OUT_ROOT="${OUT_ROOT:-$MODEL_ROOT/img_tokens_compressed_estimated}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Unprojecting decoded resnet frame latents for eid=$EID"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --decoding-npy "$DECODING_NPY" \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
