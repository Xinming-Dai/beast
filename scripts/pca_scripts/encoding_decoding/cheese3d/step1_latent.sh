#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-03:59:00
#SBATCH -J pca_cheese3d_extraction
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/pca_scripts/encoding_decoding/cheese3d/step1_latent_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Extracts PCA autoencoder per-image latents for the Cheese3D ephys session's eval-layout frames
# (see scripts/sable_scripts/encoding_decoding/cheese3d/step1_extract_eval_frames.sh and
# docs/sable/neural_extraction.md), using the checkpoint trained from
# configs/pca_cheese3d.yaml (TL/TR cameras). Unlike the IBL encoding scripts, EID here is the
# real Cheese3D ephys session id, not a checkpoint label.
EID="${EID:-20250523_B1_ephys-record_awake_000}"
JOB_ID="${JOB_ID:-21778748}"  # fill in after running scripts/pca_scripts/training/train_pca_cheese3d.sh
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/cheese3d/$JOB_ID}"
EVAL_FRAMES_DIR="${EVAL_FRAMES_DIR:-/work/hdd/bfsr/xdai3/cheese3d_neural/eval_frames}"
# eval-layout per-camera directories (see docs/sable/neural_extraction.md); each holds
# <split>/interval<N>timebin<M>.png plus a <split>/frame_index_mapping.json sidecar
LEFT_INPUT="${LEFT_INPUT:-$EVAL_FRAMES_DIR/$EID/TL}"
RIGHT_INPUT="${RIGHT_INPUT:-$EVAL_FRAMES_DIR/$EID/TR}"
OUTPUT_DIR=$MODEL_DIR/latents/frame_z/$EID
RAW_DIR=$OUTPUT_DIR/raw
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting PCA per-image latents -> $OUTPUT_DIR"

# predict_images (called via `beast predict`) has no notion of trial/timebin structure: it
# globs every PNG under --input and saves one latent .npy per image, under
# <output>/latents/<split>/<frame_stem>.npy (split is the image's parent directory name).
# Run once per camera so the two views are never mixed together.
beast predict \
    --model "$MODEL_DIR" \
    --input "$LEFT_INPUT" \
    --output "$RAW_DIR/left" \
    --batch-size "$BATCH_SIZE" \
    --save_latents

beast predict \
    --model "$MODEL_DIR" \
    --input "$RIGHT_INPUT" \
    --output "$RAW_DIR/right" \
    --batch-size "$BATCH_SIZE" \
    --save_latents

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Pairing left/right latents -> $OUTPUT_DIR/frame_z_trials.npz"

# re-group per-frame latents into per-trial [T, 2, n_components] tensors, keyed by split, using
# each camera directory's frame_index_mapping.json for neural_trial_idx/neural_bin_idx alignment
beast combine-eval-layout-latents \
    --left-input-dir "$LEFT_INPUT" \
    --right-input-dir "$RIGHT_INPUT" \
    --left-latents-dir "$RAW_DIR/left/latents" \
    --right-latents-dir "$RAW_DIR/right/latents" \
    --output "$OUTPUT_DIR/frame_z_trials.npz"

# only clean up the per-frame latents once the combined output actually exists, so a failed
# combine leaves $RAW_DIR in place for debugging
if [ -f "$OUTPUT_DIR/frame_z_trials.npz" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Deleting per-frame latents -> $RAW_DIR"
    rm -rf "$RAW_DIR"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] frame_z_trials.npz missing; keeping $RAW_DIR for debugging" >&2
fi

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> $OUTPUT_DIR/frame_z_trials.npz"
