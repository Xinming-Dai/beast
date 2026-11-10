#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-05:59:00
#SBATCH -J extraction_disable_geom
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/encoding/step1_sable_latent_disable_geom_%j.log
#SBATCH --export=ALL

# Latents (frame_z CLS + img_tokens patches) from the SABLE model trained without geometric loss
# (model.gaussians.init_gs=false), a 2D-vs-3D baseline with the same architecture. Uses the only
# checkpoint of that run (step 972). Outputs go to the hdd, not next to the checkpoint.

exec 2>&1
source ~/.bashrc
conda activate beast

# Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
fi

[ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

RUN_NAME="22359987_disable_geom"
EIDS="${EIDS:-4b00df29-3769-43be-bb40-128b1cba6d35 72cb5550-43b4-4ef0-add5-e4adfdfb5e02 781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
MODEL_DIR="${MODEL_DIR:-/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$RUN_NAME}"
DATASET_BASE=/work/hdd/bfsr/xdai3/IBL_data/synchronized
DATASET_PATH="${DATASET_PATH:-$DATASET_BASE/extracted_frames/eval}"
VDA_CACHE_ROOT=$DATASET_BASE/extracted_frames_for_eyz/eval/depth_map
CORRESPONDENCE_CACHE_ROOT=$DATASET_BASE/extracted_frames_for_eyz/eval/litpose_correspondences/processed_correspondences
OUTPUT_DIR="${OUTPUT_DIR:-/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$RUN_NAME/latents}"
LATENT_FLAGS="${LATENT_FLAGS:---return-frame-z --return-img-tokens}"
RESUME="${RESUME:-true}"                                             # set to "false" to force recompute
BATCH_SIZE="${BATCH_SIZE:-64}"                                       # override training.batch_size_per_gpu

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Extracting Sable latents ($RUN_NAME) -> $OUTPUT_DIR"

ARGS=(
    --model "$MODEL_DIR"
    --input "$DATASET_PATH"
    --output "$OUTPUT_DIR"
    --extract-latents
    --session-names $EIDS
    --vda-cache-root "$VDA_CACHE_ROOT"
    --correspondence-cache-root "$CORRESPONDENCE_CACHE_ROOT"
    --latent-batch-size "$BATCH_SIZE"
)
[ "$RESUME" = "false" ] && ARGS+=(--no-resume)

beast predict "${ARGS[@]}" $LATENT_FLAGS

conda deactivate
