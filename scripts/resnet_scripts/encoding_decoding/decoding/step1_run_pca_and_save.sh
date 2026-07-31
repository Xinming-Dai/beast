#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="resnet_pca_compress"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 120G
#SBATCH -t 0-00:05:00
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/resnet_scripts/encoding_decoding/decoding/step1_run_pca_and_save_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SABLE_repo_3/beast"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Stage 1: PCA-compress resnet frame latents. Unlike beast's own step1 (which reads sharded
# img_tokens_batch*.npz under --input-dir), resnet's step1_resnet_latent.sh already writes one
# combined frame_z_trials.npz with all three splits. run_pca_and_save.py's --combined-trials-*-npz
# flags each expect a file containing ONLY that split's rows (it never filters by a trial_split
# label — whatever rows are in the file are trusted as belonging to the role the flag was passed
# under), so split_trials_by_split.py first splits frame_z_trials.npz into three single-split
# files under $SPLIT_DIR. Requires frame_z_trials.npz to carry a trial_split key — older files
# written before beast.inference.combine_eval_layout_latents saved one need patch_trial_split.py
# run once first. --input-dir is passed (unused for reading, since every split is covered by
# --combined-trials-*-npz) purely so main()'s per-session loop nests the PCA bundle under $EID.
# Produces img_tokens_pca_joint.npz + img_tokens_compressed_trials.npz under MODEL_ROOT.
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID=20505763
MODEL_DIR="${MODEL_DIR:-/work/hdd/bfsr/xdai3/project3d_ckpt/resnet_ae_152/$EID/$JOB_ID}"
TRIALS_NPZ="${TRIALS_NPZ:-$MODEL_DIR/latents/frame_z/$EID/frame_z_trials.npz}"
SPLIT_DIR="${SPLIT_DIR:-$MODEL_DIR/latents/frame_z/$EID/per_split}"
MODEL_ROOT="${MODEL_ROOT:-$MODEL_DIR/latents}"
N_FEAT_KEEP=6                                               # PCA components to keep

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Splitting $TRIALS_NPZ into per-split files -> $SPLIT_DIR"

python "$REPO_ROOT/scripts/resnet_scripts/encoding_decoding/decoding/split_trials_by_split.py" \
    --trials-npz "$TRIALS_NPZ" \
    --out-dir "$SPLIT_DIR"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running resnet frame-latent PCA fit/apply"

# --output-pca-npz / --output-trials-npz are given explicitly (rather than relying on
# --model-root defaults) because run_pca_and_save.py only auto-nests the PCA bundle under
# $EID/ for multi-session --input-dir runs (main()'s per-session loop appends session_name to
# the PCA path only); the trials npz is never auto-nested in --combined-trials-*-npz mode
# (no trial_session_ids to partition on), so its $EID/ nesting must be requested explicitly to
# match the img_tokens_compressed/$EID/ layout step2_run_decoding.sh and step3_unproject.sh
# expect (mirroring beast's own step1_run_pca_and_save.sh layout).
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
