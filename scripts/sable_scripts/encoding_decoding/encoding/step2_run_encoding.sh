#!/bin/bash
#SBATCH -A beez-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-1:00:00
#SBATCH -J encoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Neural encoding: predicts neural activity from Sable latents (RRR/CNN, via Ray Tune).
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data   # root dir of neural (spike) data


JOB_ID="20014553"                                         
LATENT_INPUT_DIR=/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/781b35fd-e1f0-4d14-b2bb-95b7263082bb/$JOB_ID/latents
LATENT_KIND="${1:-${LATENT_KIND:-frame}}"                  # frame | dino | combined
EID="${2:-${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural encoding for eid=$EID with latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
