#!/bin/bash
#SBATCH -A beez-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-00:59:00
#SBATCH -J encoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/cheese3d/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Neural encoding: predicts neural activity from Sable latents (RRR/CNN, via Ray Tune).
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/cheese3d_neural/neural_data   # root dir of neural (spike) data

EID="20250523_B1_ephys-record_awake_000"
JOB_ID="20156493"                                         
LATENT_INPUT_DIR=/work/hdd/bfsr/xdai3/cheese3d_neural/behavior_data
LATENT_KIND="${1:-${LATENT_KIND:-behavior}}"                  # frame | dino | combined

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural encoding for eid=$EID with latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
