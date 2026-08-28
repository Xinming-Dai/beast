#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-1:00:00
#SBATCH -J dynamic_vars_encoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/dynamic_vars_scripts/encoding_decoding/encoding/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Neural encoding: predicts neural activity from the IBL DYNAMIC_VARS behavior traces (wheel
# speed, licks, whisker motion energy, nose speed, paw speed), via RRR/CNN, orchestrated by
# Ray Tune. Run step1_extract_dynamic_vars_trials.sh first.
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data   # root dir of neural (spike) data

LATENT_KIND="${1:-${LATENT_KIND:-dynamic_vars}}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
LATENT_INPUT_DIR=/projects/bfsr/xdai3/project3d/twoview3d_ckpts/dynamic_vars

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural encoding for eid=$EID with latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
