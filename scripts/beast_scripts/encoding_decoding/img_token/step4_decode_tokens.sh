#!/bin/bash
#SBATCH -A beez-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0-01:00:00
#SBATCH -J beast_decode_tokens
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/beast_scripts/encoding_decoding/img_token/step4_decode_tokens_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 4: decode (real or PCA-round-tripped) beast img_tokens back into reconstructed frames,
# via beast.sable_encoding_decoding.img_token.decode_beast_tokens (the beast analog of Sable's
# camera/Gaussian-splat decode_and_render.py — beast has no camera geometry, so this just runs
# the model's own ViT-MAE decoder + unpatchify). Produces rendered frames under OUT_DIR,
# consumed by step5_generate_video.sh.
#
# Decodes step0's img_tokens_batch*.npz shards directly (a sanity-check round trip: encode then
# immediately decode the same tokens) via --input-dir/--session-id. To decode step3's
# neurally-estimated tokens instead, switch to combined-npz mode (--img-tokens-npz pointed at an
# img_tokens_estimated*.npz + --ids-restore-npz) and note that IDS_RESTORE_NPZ still needs to
# resolve to the *original* per-trial ids_restore rows for those trials (trial-indexed lookup
# into step0's shards) — this indexed lookup is not yet implemented in decode_beast_tokens.py
# (see its module docstring), so estimated-token decoding is a known follow-up.
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=20505751
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/beast_vit_large/$EID/$JOB_ID}"
INPUT_DIR="${INPUT_DIR:-$MODEL_DIR/latents/img_tokens}"     # root of img_tokens_batch*.npz shards
OUT_DIR="${OUT_DIR:-$MODEL_DIR/decode_out}"

mkdir -p "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Decoding beast img_tokens from $INPUT_DIR/$EID"

python -m beast.sable_encoding_decoding.img_token.decode_beast_tokens \
    --model-dir "$MODEL_DIR" \
    --input-dir "$INPUT_DIR" \
    --session-id "$EID" \
    --out-dir "$OUT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done
conda deactivate
