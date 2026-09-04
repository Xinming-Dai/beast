#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-1:00:00
#SBATCH -J pca_cheese3d_encoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/pca_scripts/encoding_decoding/cheese3d/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Neural encoding: predicts neural activity from PCA autoencoder latents (RRR/CNN, via Ray
# Tune), for the Cheese3D ephys session (see scripts/sable_scripts/encoding_decoding/cheese3d/).
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/cheese3d_neural/neural_data   # root dir of neural (spike) data

EID="${EID:-20250523_B1_ephys-record_awake_000}"
JOB_ID="${JOB_ID:-21778748}"  # fill in after running scripts/pca_scripts/training/train_pca_cheese3d.sh
LATENT_KIND="${1:-${LATENT_KIND:-frame}}"                  # frame | dino | combined
LATENT_INPUT_DIR=/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/cheese3d/$JOB_ID/latents

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural encoding for eid=$EID with latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
