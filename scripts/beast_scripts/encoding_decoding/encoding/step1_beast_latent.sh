#!/bin/bash
#SBATCH -A beez-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-00:59:00
#SBATCH -J beast_large_extraction
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/encoding_decoding/encoding/step1_beast_latent_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"                                              # single session ID (eid)
JOB_ID=20505751
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/beast_vit_large/$EID/$JOB_ID}"
DATASET_BASE=/work/hdd/bfsr/xdai3/IBL_data/synchronized
# eval-layout per-camera directories (see docs/sable/neural_extraction.md); each holds
# <split>/interval<N>timebin<M>.png plus a <split>/frame_index_mapping.json sidecar
LEFT_INPUT="${LEFT_INPUT:-$DATASET_BASE/extracted_frames/eval/leftCamera.video/_iblrig_leftCamera.downsampled.$EID}"
RIGHT_INPUT="${RIGHT_INPUT:-$DATASET_BASE/extracted_frames/eval/rightCamera.video/_iblrig_rightCamera.downsampled.$EID}"
LATENT_DIR=$MODEL_DIR/latents
FRAME_Z_OUTPUT_DIR=$LATENT_DIR/frame_z/$EID
RAW_DIR=$LATENT_DIR/raw/$EID
BATCH_SIZE="${BATCH_SIZE:-60}"                                       # also controls img_tokens shard size below

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting ViT-large per-image latents + img_tokens -> $RAW_DIR"

# predict_images (called via `beast predict`) has no notion of trial/timebin structure: it
# globs every PNG under --input and saves one 768-dim latent .npy per image, under
# <output>/latents/<split>/<frame_stem>.npy (split is the image's parent directory name), plus
# --return-img-tokens saves the per-patch token grid (+ ids_restore) needed by the img_token/
# pipeline (see scripts/beast_scripts/encoding_decoding/img_token/step1_run_pca_and_save.sh).
# --return-img-tokens does not change the saved latents (same CLS token either way; see
# beast.models.vits.VisionTransformer.predict_step), so a single `beast predict` call per camera
# produces both outputs and avoids running the forward pass twice.
# Run once per camera so the two views are never mixed together.
beast predict \
    --model "$MODEL_DIR" \
    --input "$LEFT_INPUT" \
    --output "$RAW_DIR/left" \
    --batch-size "$BATCH_SIZE" \
    --save_latents \
    --return-img-tokens

beast predict \
    --model "$MODEL_DIR" \
    --input "$RIGHT_INPUT" \
    --output "$RAW_DIR/right" \
    --batch-size "$BATCH_SIZE" \
    --save_latents \
    --return-img-tokens

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Pairing left/right latents -> $FRAME_Z_OUTPUT_DIR/frame_z_trials.npz"

# re-group per-frame latents into per-trial [T, 2, 768] tensors, keyed by split, using each
# camera directory's frame_index_mapping.json for neural_trial_idx/neural_bin_idx alignment
beast combine-eval-layout-latents \
    --left-input-dir "$LEFT_INPUT" \
    --right-input-dir "$RIGHT_INPUT" \
    --left-latents-dir "$RAW_DIR/left/latents" \
    --right-latents-dir "$RAW_DIR/right/latents" \
    --output "$FRAME_Z_OUTPUT_DIR/frame_z_trials.npz"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Sharding left/right img_tokens -> $LATENT_DIR/img_tokens/$EID"

# re-group per-frame img_tokens + ids_restore into img_tokens_batch*.npz shards (left+right
# merged into one token axis per row), keeping memory under budget: for ViT-large (L=197
# tokens, D=1024) a single combined file would be tens of GiB, held entirely in RAM
beast combine-eval-layout-img-tokens \
    --left-input-dir "$LEFT_INPUT" \
    --right-input-dir "$RIGHT_INPUT" \
    --left-img-tokens-dir "$RAW_DIR/left/img_tokens" \
    --right-img-tokens-dir "$RAW_DIR/right/img_tokens" \
    --left-ids-restore-dir "$RAW_DIR/left/ids_restore" \
    --right-ids-restore-dir "$RAW_DIR/right/ids_restore" \
    --output-dir "$LATENT_DIR" \
    --session-id "$EID" \
    --batch-size "$BATCH_SIZE"

# only clean up the per-frame latents/img_tokens once both combined outputs actually exist, so a
# failed run leaves $RAW_DIR in place for debugging
if [ -f "$FRAME_Z_OUTPUT_DIR/frame_z_trials.npz" ] \
    && find "$LATENT_DIR/img_tokens/$EID" -name 'img_tokens_batch*.npz' -print -quit | grep -q .; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Deleting per-frame latents/img_tokens -> $RAW_DIR"
    rm -rf "$RAW_DIR"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] frame_z_trials.npz or img_tokens_batch*.npz shards missing; keeping $RAW_DIR for debugging" >&2
fi

conda deactivate

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done -> $FRAME_Z_OUTPUT_DIR/frame_z_trials.npz and $LATENT_DIR/img_tokens/$EID"
