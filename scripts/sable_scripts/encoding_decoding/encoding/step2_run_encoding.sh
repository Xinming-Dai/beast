#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH -t 0-00:59:00
#SBATCH -J encoding
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Neural encoding: predicts neural activity from Sable latents (RRR/CNN, via Ray Tune).
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data   # root dir of neural (spike) data

JOB_ID="21047248"
EID="5dcee0eb-b34d-4652-acc3-d10afc6eae68"
# EID="3e6a97d3-3991-49e2-b346-6948cb4580fb"
LATENT_INPUT_DIR=/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID/latents
LATENT_KIND="${1:-${LATENT_KIND:-frame}}"                  # frame | dino | combined


echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running neural encoding for eid=$EID with latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND"

conda deactivate
