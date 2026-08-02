#!/bin/bash
#SBATCH -A bfsx-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-0:59:00
#SBATCH -J encoding_permuted
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/random_model/encoding/step2_run_encoding_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Frame permutation baseline, step 2: same as encoding/step2_run_encoding.sh, but loads the
# real Sable latents from LATENT_INPUT_DIR and shuffles them along the frame axis using the
# permutation table from step1_generate_permutation_tables.sh (--permutation_dir), instead of
# reading a duplicated, pre-shuffled latent tree.
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data   # root dir of neural (spike) data


JOB_ID="20503395"
MODEL_DIR="/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_multisession/$JOB_ID"
LATENT_INPUT_DIR="$MODEL_DIR/latents"
PERMUTATION_DIR="$MODEL_DIR/latents_permuted"
LATENT_KIND="${1:-${LATENT_KIND:-frame}}"                  # frame | dino | combined
EID="${2:-${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}}"
RESULT_NAME="encoding_results_${LATENT_KIND}_permuted"   
TUNE_STORAGE_PATH="$PERMUTATION_DIR/tune_${LATENT_KIND}"    

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running frame-permutation encoding baseline for eid=$EID with latent_kind=$LATENT_KIND"
echo "LATENT_INPUT_DIR=$LATENT_INPUT_DIR"
echo "PERMUTATION_DIR=$PERMUTATION_DIR"

python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --latent_input_dir "$LATENT_INPUT_DIR" \
    --permutation_dir "$PERMUTATION_DIR" \
    --eval_task encoding \
    --latent_kind "$LATENT_KIND" \
    --result_name "$RESULT_NAME" \
    --tune_storage_path "$TUNE_STORAGE_PATH"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done."
conda deactivate
