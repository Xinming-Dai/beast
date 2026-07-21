#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 0-11:59:00
#SBATCH -J sable_pca_and_save
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step2_run_pca_and_save_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Fill these in (or export before sbatch, e.g.:
#   sbatch --export=ALL,INPUT_DIR=...,MODEL_ROOT=... \
#     scripts/sable_scripts/encoding_decoding/img_token/step2_run_pca_and_save.sh
INPUT_DIR="${INPUT_DIR:-<MODEL_ROOT>/img_tokens}"           # inference dir of img_tokens_batch*.npz shards
MODEL_ROOT="${MODEL_ROOT:-<MODEL_ROOT>}"                    # anchor dir for default output paths
STAGE="${STAGE:-all}"                                       # 1 | 2 | all
N_FEAT_KEEP="${N_FEAT_KEEP:-3}"                             # PCA components to keep

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running img-token PCA fit/apply, stage=$STAGE"

python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \
    --input-dir "$INPUT_DIR" \
    --model-root "$MODEL_ROOT" \
    --stage "$STAGE" \
    --n-feat-keep "$N_FEAT_KEEP"

conda deactivate
