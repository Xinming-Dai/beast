#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH -p cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH -t 02:00:00
#SBATCH -J beast_extract
#SBATCH -o /work/nvme/bfsr/runs/beast_extract_%j.out
#SBATCH -e /work/nvme/bfsr/runs/beast_extract_%j.err

# --- Setup environment ---
source ~/.bashrc
conda activate beast

# Limit threads to avoid OpenBLAS crash
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Move to BEAST code directory
cd /u/xdai3/code/beast

# --- Run BEAST frame extraction ---
INPUT="/work/hdd/bfsr/raw_data/beast/videos"
OUTPUT="/work/hdd/bfsr/raw_data/beast"

echo "Starting BEAST frame extraction..."
echo "Input directory:  $INPUT"
echo "Output directory: $OUTPUT"
echo "Using $OMP_NUM_THREADS threads."

beast extract --input "$INPUT" --output "$OUTPUT"

echo "Frame extraction completed!"
