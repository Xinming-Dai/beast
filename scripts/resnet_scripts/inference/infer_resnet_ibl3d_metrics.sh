#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 0-04:59:00
#SBATCH -J resnet_infer_metrics
#SBATCH -o /u/xdai3/project3d/SABLE_repo_3/beast/scripts/resnet_scripts/inference/infer_resnet_ibl3d_metrics_%j.log
#SBATCH --export=ALL

# PSNR/SSIM evaluation of a resnet autoencoder's reconstructions against their inputs, on the
# same DATASET_PATH/SESSION_NAMES as
# scripts/sable_scripts/inference/infer_sable_ibl3d_metrics.sh, for apples-to-apples numbers.
# predict_images has no scene/session concept, so this loops `beast predict` once per
# session x camera over the eval-layout leaf directories; masks are resolved via each split's
# frame_index_mapping.json (see beast.data.datasets.BaseDataset), matching Sable's SAM3 mask
# convention exactly. The two per-camera flat metrics npz files are then merged into one
# OUTPUT_DIR/{EID}/psnr_ssim_metrics.npz per EID via combine_view_metrics.py. See
# scripts/beast_scripts/inference/infer_beast_ibl3d_metrics.sh for the equivalent beast ViT
# script this one mirrors.
#
# When NEURAL_INPUT_DIR is set, EIDs with neurally-estimated resnet latents under it get their
# metrics from decode_resnet_latents.py's K/T/V neural-trial pipeline instead (one call per EID,
# same as step4_decode_resnet_latents.sh), written to OUTPUT_DIR/{EID}/psnr_ssim_metrics.npz; the
# flat beast-predict metrics (and the merge above) are skipped for those EIDs to avoid a
# redundant/inconsistent second file.

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Model dir contains config.yaml saved during training; checkpoints live under tb_logs/
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/resnet_ae_18/ibl_pretrain_restricted_sessions/21334264}"

STAGE=eval
SPLIT="${SPLIT:-test}"
PRECACHED_VIDEO_ROOT="/work/hdd/bfsr/xdai3/IBL_data/synchronized"
DATASET_PATH="${DATASET_PATH:-$PRECACHED_VIDEO_ROOT/extracted_frames/$STAGE}"

OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_DIR/inference_metrics}"

BATCH_SIZE="${BATCH_SIZE:-32}"
# 0 and 1 are boolean flags
SAVE_RECONSTRUCTIONS="${SAVE_RECONSTRUCTIONS:-0}"
SAVE_RENDER_VIEWS="${SAVE_RENDER_VIEWS:-1}"
COMPUTE_METRICS="${COMPUTE_METRICS:-1}"
USE_SEGMENTATION_MASK="${USE_SEGMENTATION_MASK:-1}"
SEGMENTATION_ROOT="${SEGMENTATION_ROOT:-$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/$STAGE}"

# Same session set as infer_sable_ibl3d_metrics.sh's SESSION_NAMES default
SESSION_NAMES="${SESSION_NAMES:-4b00df29-3769-43be-bb40-128b1cba6d35 72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"

# Root directory of per-EID neurally-estimated resnet latents, laid out like
# step4_decode_resnet_latents.sh's MODEL_ROOT (i.e. contains
# latents/img_tokens_compressed/{EID}/img_tokens_compressed_estimated/{EID}/{SPLIT}). Leave unset
# to skip the neural K/T/V pathway entirely and only run the flat beast-predict metrics below.
NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-}"

cat <<EOF
---------------------------------------
Job name:               ${SLURM_JOB_NAME:-local}
Job ID:                 ${SLURM_JOB_ID:-local}
Running on node(s):     ${SLURM_NODELIST:-$(hostname)}
Model dir:              $MODEL_DIR
Dataset path:           $DATASET_PATH
Split:                  $SPLIT
Session names:          $SESSION_NAMES
Output dir:             $OUTPUT_DIR
Batch size:             $BATCH_SIZE
Save reconstructions:   $SAVE_RECONSTRUCTIONS
Save render views:      $SAVE_RENDER_VIEWS
Compute metrics:        $COMPUTE_METRICS
Use segmentation mask:  $USE_SEGMENTATION_MASK
Segmentation root:      ${SEGMENTATION_ROOT:-(none)}
Neural input dir:       ${NEURAL_INPUT_DIR:-(none, flat metrics only)}
---------------------------------------
EOF

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting metrics inference..."

[ -d "$MODEL_DIR" ]    || { echo "ERROR: Model dir not found: $MODEL_DIR"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

for EID in $SESSION_NAMES; do
    if [ -n "$NEURAL_INPUT_DIR" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $EID has NEURAL_INPUT_DIR set, skipping flat" \
             "beast-predict metrics (handled by the K/T/V pathway below)"
        continue
    fi

    VIEW_NPZ_ARGS=()
    for CAMERA_ROLE in left right; do
        CAMERA_CAP="$(tr '[:lower:]' '[:upper:]' <<< "${CAMERA_ROLE:0:1}")${CAMERA_ROLE:1}"
        INPUT_DIR="$DATASET_PATH/${CAMERA_ROLE}Camera.video/_iblrig_${CAMERA_ROLE}Camera.downsampled.$EID/$SPLIT"
        if [ ! -d "$INPUT_DIR" ]; then
            echo "WARNING: skipping $EID/$CAMERA_ROLE, no such dir: $INPUT_DIR"
            continue
        fi

        VIEW_OUTPUT_DIR="$OUTPUT_DIR/$EID/$CAMERA_ROLE"
        PREDICT_ARGS=(
            --model "$MODEL_DIR"
            --input "$INPUT_DIR"
            --output "$VIEW_OUTPUT_DIR"
            --batch-size "$BATCH_SIZE"
        )
        [ "$SAVE_RECONSTRUCTIONS" = "1" ] && PREDICT_ARGS+=(--save_reconstructions)
        [ "$SAVE_RENDER_VIEWS" = "1" ]    && PREDICT_ARGS+=(--save-render-views)
        [ "$COMPUTE_METRICS" = "1" ]      && PREDICT_ARGS+=(--compute-metrics)
        if [ "$USE_SEGMENTATION_MASK" = "1" ]; then
            PREDICT_ARGS+=(
                --use-segmentation-mask
                --segmentation-root "$SEGMENTATION_ROOT"
                --mask-session-id "$EID"
                --mask-camera-role "$CAMERA_ROLE"
            )
        fi

        echo "[$(date +'%Y-%m-%d %H:%M:%S')] $EID / $CAMERA_CAP camera -> $VIEW_OUTPUT_DIR"
        beast predict "${PREDICT_ARGS[@]}"

        if [ "$COMPUTE_METRICS" = "1" ]; then
            VIEW_NPZ_ARGS+=(--$CAMERA_ROLE-npz "$VIEW_OUTPUT_DIR/psnr_ssim_metrics.npz")
        fi
    done

    if [ "$COMPUTE_METRICS" = "1" ] && [ "${#VIEW_NPZ_ARGS[@]}" -gt 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Merging left/right metrics for $EID"
        python -m beast.sable_encoding_decoding.render.combine_view_metrics \
            "${VIEW_NPZ_ARGS[@]}" \
            --out-npz "$OUTPUT_DIR/$EID/psnr_ssim_metrics.npz"
    fi
done

if [ -n "$NEURAL_INPUT_DIR" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting neural K/T/V metrics..."
    for EID in $SESSION_NAMES; do
        MODEL_ROOT="$NEURAL_INPUT_DIR/latents/img_tokens_compressed/$EID"
        ESTIMATED_DIR="$MODEL_ROOT/img_tokens_compressed_estimated/$EID/$SPLIT"
        TARGET_LEFT="$DATASET_PATH/leftCamera.video/_iblrig_leftCamera.downsampled.$EID"
        TARGET_RIGHT="$DATASET_PATH/rightCamera.video/_iblrig_rightCamera.downsampled.$EID"

        if [ ! -d "$ESTIMATED_DIR" ]; then
            echo "WARNING: skipping $EID, no neurally-estimated resnet latents dir: $ESTIMATED_DIR"
            continue
        fi

        NEURAL_ARGS=(
            --model-dir "$MODEL_DIR"
            --estimated-dir "$ESTIMATED_DIR"
            --target-frame-mapping-left "$TARGET_LEFT"
            --target-frame-mapping-right "$TARGET_RIGHT"
            --out-dir "$OUTPUT_DIR/$EID"
            --metrics-only
        )
        if [ "$USE_SEGMENTATION_MASK" = "1" ]; then
            NEURAL_ARGS+=(--use-segmentation-mask --segmentation-root "$SEGMENTATION_ROOT" --eid "$EID")
        fi

        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Neural K/T/V metrics for $EID from $ESTIMATED_DIR"
        python -m beast.sable_encoding_decoding.resnet.decode_resnet_latents "${NEURAL_ARGS[@]}"
    done
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done with neural K/T/V metrics."
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done."

conda deactivate
