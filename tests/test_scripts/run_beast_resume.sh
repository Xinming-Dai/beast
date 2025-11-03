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
CHECKPOINT="/work/hdd/bfsr/xdai3/raw_data/beast/test_video1/checkpoints/last.ckpt"
OUTPUT="/work/nvme/bfsr/xdai3/runs/beast"

# --- Run BEAST ---
if [ -f "$CHECKPOINT" ]; then
    echo "Found checkpoint: $CHECKPOINT"
    echo "Resuming BEAST training from checkpoint..."
    beast train --config "$CONFIG" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$OUTPUT"
else
    echo "No checkpoint found. Starting new training run."
    beast train --config "$CONFIG" --data "$DATA" --output "$OUTPUT"
fi