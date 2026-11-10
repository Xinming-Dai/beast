#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="mean_baseline_tokens"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-00:59:00
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/img_token/step3b_unproject_mean_baseline_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

# Stage 3b mean-appearance baseline for the rebuttal: instead of the neural-decoded tokens, write
# the train-split session mean token (x_session_avg from step1's PCA bundle) for every test frame.
# Same per-trial file layout as step3_unproject.sh, so step4d_eval_mean_baseline.sh can score it
# against the real frames. Needs only step1's output (not step2), so it can run beside step2.
# EID="${EID:-4b00df29-3769-43be-bb40-128b1cba6d35}"
# EID="${EID:-72cb5550-43b4-4ef0-add5-e4adfdfb5e02}"
EID="${EID:-781b35fd-e1f0-4d14-b2bb-95b7263082bb}"
JOB_ID="${JOB_ID:-21047248}"
MODEL_ROOT="/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/ibl_pretrain_restricted_sessions/$JOB_ID/latents"
LATENT_ROOT=$MODEL_ROOT/img_tokens_compressed/$EID
PCA_NPZ=$LATENT_ROOT/img_tokens_pca_joint.npz
COMPRESSED_TRIALS_NPZ=$LATENT_ROOT/img_tokens_compressed_trials.npz
OUT_ROOT=$LATENT_ROOT/img_tokens_compressed_estimated_meanbaseline
INCLUDE_SPLITS="${INCLUDE_SPLITS:-test}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Writing mean-baseline img tokens for eid=$EID splits=$INCLUDE_SPLITS"

python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid "$EID" \
    --mean-baseline \
    --pca-npz "$PCA_NPZ" \
    --compressed-trials-npz "$COMPRESSED_TRIALS_NPZ" \
    --out-root "$OUT_ROOT" \
    --include-splits "$INCLUDE_SPLITS"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done writing mean-baseline img tokens for eid=$EID"

conda deactivate
