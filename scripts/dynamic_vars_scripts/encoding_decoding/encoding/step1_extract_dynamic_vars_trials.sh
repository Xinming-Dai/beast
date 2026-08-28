#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:59:00
#SBATCH -J dynamic_vars_extraction
#SBATCH -o /u/xdai3/project3d/SABLE_repo_3/beast/scripts/dynamic_vars_scripts/encoding_decoding/encoding/step1_extract_dynamic_vars_trials_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Repackages the IBL DYNAMIC_VARS (wheel speed, licks, whisker motion energy, nose speed, paw
# speed — keypoint-derived behavior traces) already stored inside each <eid>_aligned.npz into
# a dynamic_vars_z_trials.npz that beast.sable_encoding_decoding.neural.run_encoding_decoding
# can consume via --latent_kind dynamic_vars. Pure numpy — cheap CPU job.
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data
OUTPUT_DIR=/projects/bfsr/xdai3/project3d/twoview3d_ckpts/dynamic_vars

EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting DYNAMIC_VARS trials for eid=$EID"

python -m beast.preprocess.ibl.extract_ibl_dynamic_vars_trials \
    --eid "$EID" \
    --neural-input-dir "$NEURAL_INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> $OUTPUT_DIR/dynamic_vars_z/$EID/dynamic_vars_z_trials.npz"
