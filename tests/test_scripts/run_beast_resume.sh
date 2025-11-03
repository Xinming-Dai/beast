#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -J beast_resume
#SBATCH -o /work/nvme/bfsr/xdai3/runs/beast_train_%j.out
#SBATCH -e /work/nvme/bfsr/xdai3/runs/beast_train_%j.err


# --- Setup environment ---
source ~/.bashrc
module load ffmpeg
conda activate beast
cd /u/xdai3/artifact/beast

# --- Define paths ---
CONFIG="configs/vit.yaml"
DATA="/work/hdd/bfsr/xdai3/raw_data/beast/test_video1"
CHECKPOINT="/work/nvme/bfsr/xdai3/runs/beast/tb_logs/version_0/checkpoints/epoch=664-step=7315-best.ckpt"
BASE_DIR="/work/nvme/bfsr/xdai3/runs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
EXP_TAG="beast"  
OUTPUT="${BASE_DIR}/${EXP_TAG}_${TIMESTAMP}_${GIT_HASH}"
# Create directory
mkdir -p "$OUTPUT"


# --- Run BEAST ---
if [ -f "$CHECKPOINT" ]; then
    echo "Found checkpoint: $CHECKPOINT"
    echo "Resuming BEAST training from checkpoint..."
    beast train --config "$CONFIG" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$OUTPUT"
else
    echo "No checkpoint found. Starting new training run."
    beast train --config "$CONFIG" --data "$DATA" --output "$OUTPUT"
fi