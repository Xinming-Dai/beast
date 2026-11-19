#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -t 0-04:00:00
#SBATCH -J patch_tcn_tune
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step2_tune_patch_tcn_%j.log
#SBATCH --export=ALL

# Tune the patch TCN (lr, weight decay, dropout) for one model on the development session.
# Searches on the train/val split of session 78, fold 0, with 6 PCs only; the test trials are not
# used. The best config is written to <patch_TCN dir>/tuned/locked_config.json and is applied
# unchanged by step2_eval_patch_tcn.sh.
# Usage: sbatch step2_tune_patch_tcn.sh <MODEL>
#   MODEL: full | nogeom | dino
# Env overrides: FOLD (default 0), NUM_SAMPLES (default 60).
# Use submit_patch_tcn_tuned.sh to submit the tune and eval jobs for every model.

source /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/patch_tcn_common.sh
setup_env

MODEL="${1:?usage: step2_tune_patch_tcn.sh <full|nogeom|dino>}"
DEV_SESSION=78
NUM_PCS=6
FOLD="${FOLD:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-60}"

resolve_session "$DEV_SESSION"
resolve_model "$MODEL"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] tune patch TCN: model=$MODEL session=$DEV_SESSION eid=$EID fold=$FOLD num_pcs=$NUM_PCS samples=$NUM_SAMPLES"
echo "PCA_DIR=$PCA_ROOT/$MODEL TUNED_DIR=$TUNED_DIR LOCKED_JSON=$LOCKED_JSON"

python -m beast.sable_encoding_decoding.neural.patch_tcn \
    --mode tune \
    --eid "$EID" \
    --neural_input_dir "$NEURAL_INPUT_DIR" \
    --pca_dir "$PCA_ROOT/$MODEL" \
    --out_dir "$TUNED_DIR" \
    --config_json "$LOCKED_JSON" \
    --num_pcs "$NUM_PCS" \
    --fold "$FOLD" \
    --num_samples "$NUM_SAMPLES"

conda deactivate
