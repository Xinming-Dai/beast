#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-11:59:00
#SBATCH -J sable_run_encoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Neural encoding: predicts neural activity from Sable latents (RRR/CNN, via Ray Tune).
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,EID=<session-id>,... \
#     scripts/sable_scripts/encoding_decoding/encoding/step2_run_encoding.sh
EID="${EID:-<SESSION_ID>}"                                  # session / animal id
NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-<PATH_TO_NEURAL_ROOT>}"   # root dir of neural (spike) data
LATENT_INPUT_DIR="${LATENT_INPUT_DIR:-<PATH_TO_LATENT_ROOT>}"   # root dir of latent data
LATENT_KIND="${LATENT_KIND:-frame}"                         # frame | mu_s | psae | mu_u | depth | cat | img_tokens_compressed*

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural encoding for eid=$EID"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
