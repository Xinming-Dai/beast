#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
#SBATCH -J patch_tcn_eval
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step2_eval_patch_tcn_%j.log
#SBATCH --export=ALL

# Evaluate the patch TCN with a fixed config on one session, for several PCA dimensions and folds.
# The config is applied unchanged: trained on the train trials, evaluated on the fold's test trials.
# By default it is the config shared by all models, configs/sable/encoding/patch_tcn_shared.yaml,
# and the results go to <patch_TCN dir>/tuned_shared. With USE_LOCKED=1 it is the model's own config
# from step2_tune_patch_tcn.sh (<patch_TCN dir>/tuned/locked_config.json) and the results go to
# <patch_TCN dir>/tuned.
# Usage: sbatch step2_eval_patch_tcn.sh <MODEL> <SESSION>
#   MODEL:   full | nogeom | dino
#   SESSION: 78 | 72 | 4b
# Env overrides: USE_LOCKED (default 0), FOLD (default 0), FOLD_LIST (default: FOLD; the folds to
#   run, e.g. "1 2 3 4"), NUM_PCS_LIST (default "6 12 24").
# Use submit_patch_tcn_tuned.sh to submit the jobs for every model/session.

source /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/patch_tcn_common.sh
setup_env

MODEL="${1:?usage: step2_eval_patch_tcn.sh <full|nogeom|dino> <78|72|4b>}"
SESSION="${2:?usage: step2_eval_patch_tcn.sh <full|nogeom|dino> <78|72|4b>}"
FOLD="${FOLD:-0}"
FOLD_LIST="${FOLD_LIST:-$FOLD}"
NUM_PCS_LIST="${NUM_PCS_LIST:-6 12 24}"
USE_LOCKED="${USE_LOCKED:-0}"

resolve_session "$SESSION"
resolve_model "$MODEL"

if [ "$USE_LOCKED" = 1 ]; then
    CONFIG_FILE="$LOCKED_JSON"
    RESULT_DIR="$TUNED_DIR"
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "locked config not found: $CONFIG_FILE (run step2_tune_patch_tcn.sh $MODEL first)"
        exit 1
    fi
else
    CONFIG_FILE="$SHARED_CONFIG"
    RESULT_DIR="$SHARED_DIR"
fi

for FOLD in $FOLD_LIST; do
    for NUM_PCS in $NUM_PCS_LIST; do
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] eval patch TCN: model=$MODEL session=$SESSION eid=$EID fold=$FOLD num_pcs=$NUM_PCS"
        echo "PCA_DIR=$PCA_ROOT/$MODEL RESULT_DIR=$RESULT_DIR CONFIG_FILE=$CONFIG_FILE"

        python -m beast.sable_encoding_decoding.neural.patch_tcn \
            --mode eval \
            --eid "$EID" \
            --neural_input_dir "$NEURAL_INPUT_DIR" \
            --pca_dir "$PCA_ROOT/$MODEL" \
            --out_dir "$RESULT_DIR" \
            --config "$CONFIG_FILE" \
            --num_pcs "$NUM_PCS" \
            --fold "$FOLD"
    done
done

conda deactivate
