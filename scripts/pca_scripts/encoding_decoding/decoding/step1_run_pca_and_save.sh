#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="pca_pca_compress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 5G
#SBATCH -t 0-00:59:00
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/SABLE_repo_3/beast/scripts/pca_scripts/encoding_decoding/decoding/step1_run_pca_and_save_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 1: PCA-compress the PCA-autoencoder frame latents (component scores) so the neural
# decoder predicts a small feature set. The encoding side's step1_latent.sh already writes one
# combined frame_z_trials.npz with all three splits, in exactly resnet's layout, so this mirrors
# scripts/resnet_scripts/encoding_decoding/decoding/step1_run_pca_and_save.sh: split the
# combined file into per-split files (run_pca_and_save.py's --combined-trials-*-npz flags each
# trust every row in the file as belonging to that split), then fit/apply the compression PCA.
# The splitter is reused from the resnet directory unchanged since it is generic over the npz.
# Produces img_tokens_pca_joint.npz + img_tokens_compressed_trials.npz under
# $MODEL_ROOT/img_tokens_compressed/$EID/.
EID="${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}"
# EID="${EID:-72cb5550-43b4-4ef0-add5-e4adfdfb5e02}"
# EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=21810401
MODEL_DIR="${MODEL_DIR:-/projects/bfsr/xdai3/project3d/twoview3d_ckpts/pca_ae/restricted_sessions/$JOB_ID}"
TRIALS_NPZ="${TRIALS_NPZ:-$MODEL_DIR/latents/frame_z/$EID/frame_z_trials.npz}"
SPLIT_DIR="${SPLIT_DIR:-$MODEL_DIR/latents/frame_z/$EID/per_split}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents}"
N_FEAT_KEEP=6                                               # PCA components to keep

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Splitting $TRIALS_NPZ into per-split files -> $SPLIT_DIR"

python "$REPO_ROOT/scripts/resnet_scripts/encoding_decoding/decoding/split_trials_by_split.py" \
    --trials-npz "$TRIALS_NPZ" \
    --out-dir "$SPLIT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running PCA frame-latent compression fit/apply"

# --input-dir is passed (unused for reading, since every split is covered by
# --combined-trials-*-npz) purely so main()'s per-session loop nests the PCA bundle under $EID.
# --output-pca-npz / --output-trials-npz are given explicitly because the trials npz is never
# auto-nested in --combined-trials-*-npz mode; the $EID/ nesting must match the
# img_tokens_compressed/$EID/ layout step2_run_decoding.sh and step3_unproject.sh expect.
python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \
    --input-dir "$MODEL_ROOT/frame_z" \
    --combined-trials-train-npz "$SPLIT_DIR/frame_z_trials_train.npz" \
    --combined-trials-val-npz "$SPLIT_DIR/frame_z_trials_val.npz" \
    --combined-trials-test-npz "$SPLIT_DIR/frame_z_trials_test.npz" \
    --session-names "$EID" \
    --output-pca-npz "$MODEL_ROOT/img_tokens_compressed/img_tokens_pca_joint.npz" \
    --output-trials-npz "$MODEL_ROOT/img_tokens_compressed/$EID/img_tokens_compressed_trials.npz" \
    --n-feat-keep "$N_FEAT_KEEP"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Job done"
conda deactivate
