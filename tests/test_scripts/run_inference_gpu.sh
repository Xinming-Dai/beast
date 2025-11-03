#!/bin/bash
#SBATCH -A bezq-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -J beast_infer
#SBATCH -o /work/hdd/bfsr/xdai3/logs/inference_gpu_%j.out
#SBATCH -e /work/hdd/bfsr/xdai3/logs/inference_gpu_%j.err

# --- Environment setup ---
source ~/.bashrc
module load ffmpeg
conda activate beast
cd /u/xdai3/artifact/beast

# --- Define paths ---
MODEL="/work/nvme/bfsr/xdai3/runs/beast_train_13512555/tb_logs/version_0/checkpoints"
DATA="/work/hdd/bfsr/xdai3/raw_data/beast/selected_images"

# Extract the training job ID automatically from the model path
TRAIN_ID=$(basename "$(dirname "$(dirname "$MODEL")")" | grep -o '[0-9]\+$')

# Define a unique output directory that includes both model and inference job IDs
OUTPUT="/work/hdd/bfsr/xdai3/beast/beast_inference_train${TRAIN_ID}_infer${SLURM_JOB_ID}"
mkdir -p "$OUTPUT"

echo "---------------------------------------"
echo "Starting BEAST inference"
echo "Training job ID: $TRAIN_ID"
echo "Inference job ID: $SLURM_JOB_ID"
echo "Model path: $MODEL"
echo "Data path:  $DATA"
echo "Output dir: $OUTPUT"
echo "---------------------------------------"

# --- Run BEAST Inference ---
beast predict \
    --model "$MODEL" \
    --input "$DATA" \
    --output "$OUTPUT" \
    --save_latents \
    --save_reconstructions

echo "Inference completed!"
echo "Results saved to: $OUTPUT"
