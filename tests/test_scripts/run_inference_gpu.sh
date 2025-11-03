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
#SBATCH -o /work/hdd/bfsr/xdai3/inference_logs/inference_gpu_%j.out
#SBATCH -e /work/hdd/bfsr/xdai3/inference_logs/inference_gpu_%j.err

# --- Environment setup ---
source ~/.bashrc
module load ffmpeg
conda activate beast
cd /u/xdai3/artifact/beast

# --- Define paths ---
MODEL="/work/nvme/bfsr/xdai3/runs/beast_train_13512555/tb_logs/version_0/checkpoints"
DATA="/work/hdd/bfsr/xdai3/raw_data/beast/selected_images"

MODEL_PARENT=$(basename "$(dirname "$(dirname "$(dirname "$MODEL")")")")
TRAIN_ID=$(echo "$MODEL_PARENT" | grep -Eo '[0-9]+$')

OUTPUT="/work/hdd/bfsr/xdai3/beast/beast_inference_train_${TRAIN_ID}_inference_${SLURM_JOB_ID}"
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
