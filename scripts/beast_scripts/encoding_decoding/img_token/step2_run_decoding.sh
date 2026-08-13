#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-00:30:00
#SBATCH -J beast_img_token_decoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/encoding_decoding/img_token/step2_run_decoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 2: neural decoding, predicts PCA-compressed beast img_tokens from neural activity.
# Requires step1's img_tokens_compressed_trials.npz as LATENT_INPUT_DIR; produces
# decoding_results_img_tokens_compressed.npy, consumed by step3_unproject.sh.
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data

EID="${EID:-f312aaec-3b6f-44b3-86b4-3a0c119c0438}"
JOB_ID=20668699
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/beast_vit_large/$EID/$JOB_ID}"
LATENT_INPUT_DIR="${LATENT_INPUT_DIR:-$MODEL_DIR/latents}"
LATENT_KIND="${LATENT_KIND:-img_tokens_compressed}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running beast neural decoding for eid=$EID"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task decoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
