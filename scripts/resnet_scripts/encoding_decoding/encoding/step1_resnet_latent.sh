#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:59:00
#SBATCH -J resnet152_extraction
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/resnet_scripts/encoding_decoding/encoding/step1_resnet_latent_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"                                              # single session ID (eid)
JOB_ID=20505763
MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/resnet_ae_152/$EID/$JOB_ID}"
DATASET_BASE=/work/hdd/bfsr/xdai3/IBL_data/synchronized
# eval-layout per-camera directories (see docs/sable/neural_extraction.md); each holds
# <split>/interval<N>timebin<M>.png plus a <split>/frame_index_mapping.json sidecar
LEFT_INPUT="${LEFT_INPUT:-$DATASET_BASE/extracted_frames/eval/leftCamera.video/_iblrig_leftCamera.downsampled.$EID}"
RIGHT_INPUT="${RIGHT_INPUT:-$DATASET_BASE/extracted_frames/eval/rightCamera.video/_iblrig_rightCamera.downsampled.$EID}"
OUTPUT_DIR=$MODEL_DIR/latents
RAW_DIR=$OUTPUT_DIR/raw
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting ResNet-152 per-image latents -> $OUTPUT_DIR"

# predict_images (called via `beast predict`) has no notion of trial/timebin structure: it
# globs every PNG under --input and saves one 768-dim latent .npy per image, under
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

# re-group per-frame latents into per-trial [T, 2, 768] tensors, keyed by split, using each
# camera directory's frame_index_mapping.json for neural_trial_idx/neural_bin_idx alignment
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
