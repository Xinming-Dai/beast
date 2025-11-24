#!/bin/bash

#SBATCH -A bfsr-delta-gpu

#SBATCH -p gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --gpus-per-task=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=100G

#SBATCH -t 02:00:00

#SBATCH -J beast_predict

#SBATCH -o /work/nvme/bfsr/xdai3/runs/beast_predict_%j.out

#SBATCH -e /work/nvme/bfsr/xdai3/runs/beast_predict_%j.err

# --- Setup environment ---

source ~/.bashrc

module load ffmpeg

conda activate beast

cd /u/xdai3/beast

# Set multiprocessing temp directory to a more stable location (avoid /tmp cleanup issues)

export TMPDIR="/work/nvme/bfsr/xdai3/tmp/${SLURM_JOB_ID:-$USER}"

mkdir -p "$TMPDIR"

echo "TMPDIR set to: $TMPDIR"

# --- Define paths ---

MODEL="/work/nvme/bfsr/xdai3/runs/beast_train_13778575"

INPUT="/work/nvme/bfsr/xdai3/raw_data/beast/selected_images"

# Define unique output directory per job (using Slurm job name + ID)

OUTPUT_DIR="/work/nvme/bfsr/xdai3/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"

echo "---------------------------------------"

echo "Job name: $SLURM_JOB_NAME"

echo "Job ID: $SLURM_JOB_ID"

echo "Running on node(s): $SLURM_NODELIST"

echo "Output directory: $OUTPUT_DIR"

echo "---------------------------------------"

# --- Run BEAST Inference ---

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting BEAST inference..."

if [ -d "$MODEL" ]; then

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Found model directory: $MODEL"

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Input directory: $INPUT"

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] About to call: beast predict --model \"$MODEL\" --input \"$INPUT\" --save_reconstructions"

    beast predict --model "$MODEL" --input "$INPUT" --save_reconstructions 2>&1 | tee "$OUTPUT_DIR/inference_output.log"

else

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: Model directory not found: $MODEL"

    exit 1

fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] BEAST inference completed."

conda deactivate

