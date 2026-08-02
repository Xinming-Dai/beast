#!/bin/bash
#SBATCH -A bfsr-delta-cpu
#SBATCH --job-name="permute_tables"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 32G
#SBATCH -t 0-00:05:00
#SBATCH --export=ALL
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/encoding_decoding/random_model/encoding/step1_generate_permutation_tables_%j.log

exec 2>&1
source ~/.bashrc
conda activate beast

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
cd "$REPO_ROOT"

JOB_ID="20503395"
MODEL_DIR="/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts/beast_sable/781b35fd-e1f0-4d14-b2bb-95b7263082bb/20014553"
LATENT_ROOT="$MODEL_DIR/latents"
OUTPUT_DIR="$MODEL_DIR/latents_permuted"
LATENT_KIND="${LATENT_KIND:-frame}"                         # frame | dino | combined 
EIDS="${EIDS:-}"                                             # space-separated session ids; empty = auto-discover all

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Generating permutation tables -> $OUTPUT_DIR"

ARGS=(
    --latent_root "$LATENT_ROOT"
    --output_dir "$OUTPUT_DIR"
    --latent_kind "$LATENT_KIND"
)
[ -n "$EIDS" ] && ARGS+=(--eids $EIDS)

python -m beast.sable_encoding_decoding.neural.generate_permutation_tables "${ARGS[@]}"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done."
conda deactivate
