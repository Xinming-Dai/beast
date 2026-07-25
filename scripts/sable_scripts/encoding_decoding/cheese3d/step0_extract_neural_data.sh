#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-11:59:00
#SBATCH -J extract_cheese3d_neural_data
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/cheese3d/step0_extract_neural_data_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Bins Cheese3D ephys spikes into 1s trial windows, filters units by firing rate, splits
# train/val/test, and writes <eid>_aligned.npz + frame_manifest.json. Pure numpy — cheap CPU job.
# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,EID=...,FR_THRESH=... \
#     scripts/sable_scripts/encoding_decoding/cheese3d/step0_extract_neural_data.sh
EID="${EID:-20250523_B1_ephys-record_awake_000}"
ALIGNMENT_NPZ="${ALIGNMENT_NPZ:-/work/hdd/bfsr/xdai3/cheese3d/spike/${EID}_100fps.npz}"
NEURAL_OUTPUT_DIR="${NEURAL_OUTPUT_DIR:-/work/hdd/bfsr/xdai3/cheese3d_neural/neural_data}"
TRIAL_LEN_SEC="${TRIAL_LEN_SEC:-1.0}"
FR_THRESH="${FR_THRESH:-0.2}"
NUM_TRIALS="${NUM_TRIALS:-}"   # empty = keep all non-overlapping trial windows

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Cheese3D neural data for eid=$EID"

ARGS=(
    --alignment-npz "$ALIGNMENT_NPZ"
    --eid "$EID"
    --neural-output-dir "$NEURAL_OUTPUT_DIR"
    --trial-len-sec "$TRIAL_LEN_SEC"
    --fr-thresh "$FR_THRESH"
)
[ -n "$NUM_TRIALS" ] && ARGS+=(--num-trials "$NUM_TRIALS")

python -m beast.preprocess.cheese3d.extract_cheese3d_neural_data "${ARGS[@]}"

conda deactivate
