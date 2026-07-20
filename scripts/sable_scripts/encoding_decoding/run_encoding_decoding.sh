#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-11:59:00
#SBATCH -J sable_encoding_decoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/run_encoding_decoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,EID=<session-id>,... scripts/sable_scripts/encoding_decoding/run_encoding_decoding.sh
EID="${EID:-<SESSION_ID>}"                                  # session / animal id
NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-<PATH_TO_NEURAL_ROOT>}"   # root dir of neural (spike) data
LATENT_INPUT_DIR="${LATENT_INPUT_DIR:-<PATH_TO_LATENT_ROOT>}"   # root dir of latent data
EVAL_TASK="${EVAL_TASK:-encoding}"                          # encoding or decoding
LATENT_KIND="${LATENT_KIND:-frame}"                         # frame | mu_s | psae | mu_u | depth | cat | img_tokens_compressed*

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural $EVAL_TASK for eid=$EID"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task "$EVAL_TASK" \
    --latent_kind "$LATENT_KIND"

conda deactivate
