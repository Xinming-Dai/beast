#!/bin/bash
#SBATCH -A bezq-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH -J interactive_gpu
#SBATCH -o /work/hdd/bfsr/xdai3/logs/inference_gpu%j.out
#SBATCH -e /work/hdd/bfsr/xdai3/logs/inference_gpu%j.err

# --- Commands below run on the compute node ---

# Load your environment
source ~/.bashrc
module load ffmpeg
conda activate beast
cd /u/xdai3/artifact/beast

# --- Define paths ---
model="/u/xdai3/artifact/beast/runs/2025-10-24/10-43-28"
DATA="/u/xdai3/artifact/beast/tests/test_video"

# --- Run BEAST Inference---
echo "No checkpoint found. Starting new training run."
beast predict --model "$model" --input "$DATA"

