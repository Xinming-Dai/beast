#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 0-04:59:00
#SBATCH -J erz_infer_metrics
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/inference/infer_sable_ibl3d_metrics_%j.log
#SBATCH --export=ALL

# Exhaustive PSNR/SSIM evaluation on predicted views: runs inference over every frame of a
# split for the given DATASET_PATH/SESSION_NAMES, applies the segmentation mask to both the
# render and the target before scoring, saves every masked render-only PNG, and computes
# PSNR/SSIM across all of them. No PLY point clouds or GLB scenes are saved.
#
# When NEURAL_INPUT_DIR is set, EIDs with neural token latents under it get their metrics from
# decode_and_render.py's K/T/V neural-trial pipeline instead (one call per EID, same as
# step4_decode_and_render.sh), written to OUTPUT_DIR/{EID}/psnr_ssim_metrics.npz; the flat
# beast-predict metrics are skipped in that case to avoid a redundant/inconsistent second file.

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
JOB_ID="${JOB_ID:-21047248}"

STAGE=eval
PRECACHED_VIDEO_ROOT="/work/hdd/bfsr/xdai3/IBL_data/synchronized"
DATASET_PATH="${DATASET_PATH:-$PRECACHED_VIDEO_ROOT/extracted_frames/$STAGE}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODEL_DIR/inference_metrics}"

SPLITS="${SPLITS:-test}"
# 0 and 1 are boolean flags
SAVE_VISUALS="${SAVE_VISUALS:-0}"
SAVE_PLY="${SAVE_PLY:-0}"
SAVE_RENDER_VIEWS="${SAVE_RENDER_VIEWS:-1}"
COMPUTE_METRICS="${COMPUTE_METRICS:-1}"
USE_SEGMENTATION_MASK="${USE_SEGMENTATION_MASK:-1}"
MAX_BATCHES="${MAX_BATCHES:-}"
MAX_FILES_PER_SESSION="${MAX_FILES_PER_SESSION:-}"
SEGMENTATION_ROOT="${SEGMENTATION_ROOT:-$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/$STAGE}"

VDA_CACHE_ROOT="${VDA_CACHE_ROOT:-$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/$STAGE/depth_map}"
CORRESPONDENCE_CACHE_ROOT="${CORRESPONDENCE_CACHE_ROOT:-$PRECACHED_VIDEO_ROOT/extracted_frames_for_eyz/$STAGE/litpose_correspondences/processed_correspondences}"
NEURAL_BATCH_SIZE="${NEURAL_BATCH_SIZE:-60}"
SESSION_NAMES="${SESSION_NAMES:-4b00df29-3769-43be-bb40-128b1cba6d35 72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
# Model dir contains config.yaml saved during training; checkpoints live under tb_logs/
MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID}"

# Root directory of per-EID neural token latents, laid out like step4_decode_and_render.sh's
# LATENT_ROOT (i.e. contains latents/img_tokens_compressed/{EID}/...). Leave unset to skip the
# neural K/T/V pathway entirely and only run the flat beast-predict metrics below.
NEURAL_INPUT_DIR="${NEURAL_INPUT_DIR:-/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data}"

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

for _cuda in /usr/local/cuda \
             /opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8 \
             /opt/cuda; do
    [ -d "$_cuda" ] && export CUDA_HOME="$_cuda" && break
done
export PATH="${CUDA_HOME:-}/bin:${PATH}"
export PYTHONUNBUFFERED=1

cat <<EOF
---------------------------------------
Job name:              ${SLURM_JOB_NAME:-local}
Job ID:                ${SLURM_JOB_ID:-local}
Running on node(s):    ${SLURM_NODELIST:-$(hostname)}
Model dir:             $MODEL_DIR
Dataset path:          $DATASET_PATH
Output dir:            $OUTPUT_DIR
Splits:                $SPLITS
Session names:         ${SESSION_NAMES:-(all sessions from saved training config)}
Max files per session: ${MAX_FILES_PER_SESSION:-(unlimited)}
Save visuals:           $SAVE_VISUALS
Save PLY point clouds:  $SAVE_PLY
Save render views:      $SAVE_RENDER_VIEWS
Compute metrics:        $COMPUTE_METRICS
Use segmentation mask:  $USE_SEGMENTATION_MASK
Segmentation root:      ${SEGMENTATION_ROOT:-(from saved training config)}
Max batches:            ${MAX_BATCHES:-(all)}
Neural input dir:       ${NEURAL_INPUT_DIR:-(none, flat metrics only)}
TORCH_CUDA_ARCH_LIST:  $TORCH_CUDA_ARCH_LIST
---------------------------------------
EOF

echo '=== GPU (PyTorch) ==='
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"  cuda:{i} {torch.cuda.get_device_name(i)}  capability={cap[0]}.{cap[1]}")
PY

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting metrics inference..."

[ -d "$MODEL_DIR" ]    || { echo "ERROR: Model dir not found: $MODEL_DIR"; exit 1; }
[ -d "$DATASET_PATH" ] || { echo "ERROR: Dataset path not found: $DATASET_PATH"; exit 1; }

cd "$REPO_ROOT"

PREDICT_ARGS=(
    --model "$MODEL_DIR"
    --input "$DATASET_PATH"
    --output "$OUTPUT_DIR"
    --splits $SPLITS
)
[ -n "$SESSION_NAMES" ]         && PREDICT_ARGS+=(--session-names $SESSION_NAMES)
[ -n "$MAX_FILES_PER_SESSION" ] && PREDICT_ARGS+=(--max-files-per-session "$MAX_FILES_PER_SESSION")
[ "$SAVE_VISUALS" = "1" ]       && PREDICT_ARGS+=(--save-visuals)
[ "$SAVE_PLY" != "1" ]          && PREDICT_ARGS+=(--no-save-pointclouds)
[ "$SAVE_RENDER_VIEWS" = "1" ]  && PREDICT_ARGS+=(--save-render-views)
[ -n "$NEURAL_INPUT_DIR" ]      && PREDICT_ARGS+=(--neural-input-dir "$NEURAL_INPUT_DIR")
# Skip the flat metrics npz when NEURAL_INPUT_DIR is set: EIDs with neural data get their metrics
# from decode_and_render.py's K/T/V pipeline below instead.
[ "$COMPUTE_METRICS" = "1" ] && [ -z "$NEURAL_INPUT_DIR" ] && PREDICT_ARGS+=(--compute-metrics)
if [ "$USE_SEGMENTATION_MASK" = "1" ]; then
    PREDICT_ARGS+=(--use-segmentation-mask)
    [ -n "$SEGMENTATION_ROOT" ] && PREDICT_ARGS+=(--segmentation-root "$SEGMENTATION_ROOT")
fi
[ -n "$MAX_BATCHES" ]           && PREDICT_ARGS+=(--max-batches "$MAX_BATCHES")

beast predict "${PREDICT_ARGS[@]}"

if [ -n "$NEURAL_INPUT_DIR" ]; then
    if [ -z "$SESSION_NAMES" ]; then
        echo "ERROR: NEURAL_INPUT_DIR is set but SESSION_NAMES is empty; the neural pathway needs" \
             "an explicit --eid per decode_and_render.py call, so it can't fall back to" \
             "training.session_names from the saved config. Set SESSION_NAMES explicitly."
        exit 1
    fi
    echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Starting neural K/T/V metrics..."
    for EID in $SESSION_NAMES; do
        SUBDIR="latents/img_tokens_compressed/$EID"
        Z_SOURCE="$NEURAL_INPUT_DIR/$SUBDIR/img_tokens_compressed_estimated/$EID/$SPLITS"
        CAMERA_NPZ="$NEURAL_INPUT_DIR/$SUBDIR/img_tokens_camera_parameters.npz"

        if [ ! -d "$Z_SOURCE" ]; then
            echo "WARNING: no neural token dir for $EID at $Z_SOURCE, skipping neural metrics for this EID"
            continue
        fi

        echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Neural metrics for $EID from $Z_SOURCE"

        NEURAL_ARGS=(
            --z-source "$Z_SOURCE"
            --camera-npz "$CAMERA_NPZ"
            --out-dir "$OUTPUT_DIR/$EID"
            --model-dir "$MODEL_DIR"
            --dataset-path "$DATASET_PATH"
            --vda-cache-root "$VDA_CACHE_ROOT"
            --correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT"
            --batch-size "$NEURAL_BATCH_SIZE"
            --include-splits "$SPLITS"
            --metrics-only
            --eid "$EID"
        )
        if [ "$USE_SEGMENTATION_MASK" = "1" ]; then
            NEURAL_ARGS+=(--use-segmentation-mask)
            [ -n "$SEGMENTATION_ROOT" ] && NEURAL_ARGS+=(--segmentation-root "$SEGMENTATION_ROOT")
        fi

        python -m beast.sable_encoding_decoding.render.decode_and_render "${NEURAL_ARGS[@]}"
    done
    echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Done with neural K/T/V metrics."
fi

echo "[$(TZ=America/New_York date +'%Y-%m-%d %H:%M:%S')] Done."

conda deactivate
