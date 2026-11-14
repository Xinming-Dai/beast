#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 0-02:00:00
#SBATCH -J patch_tcn
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step2_run_patch_tcn_%j.log
#SBATCH --export=ALL

# Neural encoding with a TCN on PCA-compressed patch embeddings (one model, one session).
# Usage: sbatch step2_run_patch_tcn.sh <MODEL> <SESSION>
#   MODEL:   full | nogeom | dino
#   SESSION: 78 | 72 | 4b
# Env overrides: FOLD (default 0), NUM_PCS (default 6), NUM_SAMPLES (default 30).
# Use submit_patch_tcn.sh to submit every model/session pair.

exec 2>&1
export PYTHONUNBUFFERED=1  # flush print output to the slurm log as it happens
source ~/.bashrc
conda activate beast

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

MODEL="${1:?usage: step2_run_patch_tcn.sh <full|nogeom|dino> <78|72|4b>}"
SESSION="${2:?usage: step2_run_patch_tcn.sh <full|nogeom|dino> <78|72|4b>}"
FOLD="${FOLD:-0}"
NUM_PCS="${NUM_PCS:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-30}"

NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data
CKPT_ROOT=/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts

case "$SESSION" in
    78)
        EID="781b35fd-e1f0-4d14-b2bb-95b7263082bb"
        PCA_ROOT=/work/nvme/bezq/jzhang107/sable_patch_pca_cv5/session_78
        ;;
    72)
        EID="72cb5550-43b4-4ef0-add5-e4adfdfb5e02"
        PCA_ROOT=/work/hdd/bezq/jzhang107/sable/patch_pca_cv5_sessions/session_72
        ;;
    4b)
        EID="4b00df29-3769-43be-bb40-128b1cba6d35"
        PCA_ROOT=/work/hdd/bezq/jzhang107/sable/patch_pca_cv5_sessions/session_4b
        ;;
    *)
        echo "unknown SESSION=$SESSION (expected 78, 72 or 4b)"
        exit 1
        ;;
esac

case "$MODEL" in
    full)
        OUT_DIR=$CKPT_ROOT/beast_sable/ibl_pretrain_restricted_sessions/21047248/latents/patch_TCN
        ;;
    nogeom)
        OUT_DIR=$CKPT_ROOT/beast_sable/ibl_pretrain_restricted_sessions/22359987_disable_geom/latents/patch_TCN
        ;;
    dino)
        OUT_DIR=$CKPT_ROOT/dino_baseline/latents/patch_TCN
        ;;
    *)
        echo "unknown MODEL=$MODEL (expected full, nogeom or dino)"
        exit 1
        ;;
esac

echo "[$(date +'%Y-%m-%d %H:%M:%S')] patch TCN: model=$MODEL session=$SESSION eid=$EID fold=$FOLD num_pcs=$NUM_PCS"
echo "PCA_DIR=$PCA_ROOT/$MODEL OUT_DIR=$OUT_DIR"

python -m beast.sable_encoding_decoding.neural.patch_tcn \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --pca_dir "$PCA_ROOT/$MODEL" \
    --out_dir "$OUT_DIR" \
    --num_pcs "$NUM_PCS" \
    --fold "$FOLD" \
    --num_samples "$NUM_SAMPLES"

conda deactivate
