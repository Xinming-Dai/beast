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
#SBATCH -o logs/interactive_gpu_%j.out
#SBATCH -e logs/interactive_gpu_%j.err

# --- Commands below run on the compute node ---

# Load your environment
source ~/.bashrc
conda activate beast

# Start an interactive shell (optional)
# You can comment this out and replace it with your actual job commands
/bin/bash
