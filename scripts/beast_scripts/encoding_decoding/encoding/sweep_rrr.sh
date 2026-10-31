#!/bin/bash
#SBATCH -A bfsx-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-1:30:00
#SBATCH -J beast_rrr_sweep
#SBATCH -o /u/xdai3/project3d/SABLE_repo_3/beast/scripts/beast_scripts/encoding_decoding/encoding/sweep_rrr_%j.log
#SBATCH --export=ALL

# Single-job RRR hyperparameter sweep (lr x n_comp x l2) on beast latents, selected on VALIDATION
# BPS. For each session the best config is refit and scored on test, and only the 'rrr' entry of
# <latents>/<LATENT_KIND>_z/<eid>/encoding_results_<LATENT_KIND>.npy is replaced; the 'cnn' entry is
# left as is. The original file is kept as encoding_results_<LATENT_KIND>_pre_rrr_sweep.npy and the
# per-trial table is written to rrr_sweep_trials.csv in each session dir.
#
#   sbatch scripts/beast_scripts/encoding_decoding/encoding/sweep_rrr.sh
#   EIDS="4b00df29-3769-43be-bb40-128b1cba6d35" LRS="0.5" NCOMPS="3" L2S="100" sbatch <this script>
#
# If the winning value sits on the edge of a grid (e.g. lr=1.0 or n_comp=32), extend it and rerun.

exec 2>&1
source ~/.bashrc
conda activate beast
set -eo pipefail

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data   # root dir of neural (spike) data

JOB_ID="${JOB_ID:-21334260}"
LATENT_KIND="${LATENT_KIND:-frame}"                        # frame | dino | combined
LATENT_INPUT_DIR="${LATENT_INPUT_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/beast_vit/ibl_pretrain_restricted_sessions/$JOB_ID/latents}"
EIDS="${EIDS:-4b00df29-3769-43be-bb40-128b1cba6d35 72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"

LRS="${LRS:-0.1 0.5 1.0}"                                  # L-BFGS learning rate
NCOMPS="${NCOMPS:-3 9 32}"                                 # RRR rank
L2S="${L2S:-25 100 200}"                                   # L2 penalty on beta

echo "[$(date +'%Y-%m-%d %H:%M:%S')] RRR sweep latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"
echo "EIDS=$EIDS"
echo "LRS=$LRS NCOMPS=$NCOMPS L2S=$L2S"

for EID in $EIDS; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] RRR sweep for eid=$EID"
    python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
        --eid "$EID" \
        --neural_input_dir "$NEURAL_INPUT_DIR" \
        --latent_input_dir "$LATENT_INPUT_DIR" \
        --eval_task encoding \
        --latent_kind "$LATENT_KIND" \
        --rrr_sweep \
        --rrr_lrs $LRS \
        --rrr_ncomps $NCOMPS \
        --rrr_l2s $L2S
done

conda deactivate
