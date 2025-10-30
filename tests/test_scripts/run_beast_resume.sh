#!/bin/bash
#SBATCH -A bezq-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -J beast_resume
#SBATCH -o logs/beast_resume_%j.out
#SBATCH -e logs/beast_resume_%j.err

# --- Setup environment ---
source ~/.bashrc
module load ffmpeg
conda activate beast
cd /u/xdai3/artifact/beast

# --- Define paths ---
CONFIG="configs/vit.yaml"
DATA="/u/xdai3/artifact/beast/tests/test_video/extracted_frames"
CHECKPOINT="/u/xdai3/artifact/beast/runs/2025-10-23/07-58-57/checkpoints/last.ckpt"

# --- Run BEAST ---
if [ -f "$CHECKPOINT" ]; then
    echo "Found checkpoint: $CHECKPOINT"
    echo "Resuming BEAST training from checkpoint..."
    beast train --config "$CONFIG" --data "$DATA" --checkpoint "$CHECKPOINT"
else
    echo "No checkpoint found. Starting new training run."
    beast train --config "$CONFIG" --data "$DATA"
fi