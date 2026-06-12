#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# K-ablation visual audit: 5 rigid-head keypoint variants.
# REPO_ROOT = /home/jqh/NeuralWorkshops/beast  (script auto-appends this)
# Output: ${OUTPUT_ROOT}/<variant>_1k/
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT="scripts/sable_scripts/run_cheese3d_phase3_smoke.py"
CONFIG="configs/sable_cheese3d_nvs.yaml"
SESSION="20231031_B20_chew_bl_000"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/jqh/pretrained_checkpoints/E-RayZer-private/checkpoints}"
DINO_HOME="${DINO_HOME:-${CHECKPOINT_ROOT}/dinov3-vitb16-pretrain-lvd1689m}"
RESUME="${RESUME:-${CHECKPOINT_ROOT}/erayzer_dl3dv.pt}"
CACHE_ROOT="${CACHE_ROOT:-/data/jqh/Outputs/beast/outputs/cheese3d_stage1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/jqh/Outputs/beast/outputs/cheese3d_visual_audit}"
PYTHON_BIN="${PYTHON_BIN:-/home/jqh/miniconda3/envs/neuro/bin/python}"
MAX_STEPS="${MAX_STEPS:-1000}"
MAX_BATCHES="${MAX_BATCHES:-4}"
VIS_SAMPLES="${VIS_SAMPLES:-4}"
GS_REG="${GS_REG:-1.0}"
GPU="${GPU:-1}"

declare -A CACHE_MAP=(
    [rigidEarNose11]="${CACHE_ROOT}/cache_rigidEarNose11"
    [rigidFace13]="${CACHE_ROOT}/cache_rigidFace13"
    [rigidFace15]="${CACHE_ROOT}/cache_rigidFace15"
    [rigidNoEyeBottom17]="${CACHE_ROOT}/cache_rigidNoEyeBottom17"
    [rigidHead]="${CACHE_ROOT}/cache_rigidHead"
)

# ── Step 1: fine-tune 1000 steps ─────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "STEP 1: Fine-tuning (1000 steps × 5 variants)  GPU=$GPU"
echo "═══════════════════════════════════════════════════════════"

for variant in rigidEarNose11 rigidFace13 rigidFace15 rigidNoEyeBottom17 rigidHead; do
    cache="${CACHE_MAP[$variant]}"
    out_dir="${OUTPUT_ROOT}/${variant}_1k"

    if [ -f "${out_dir}/checkpoint_step_01000.pt" ]; then
        echo ""
        echo "[${variant}] checkpoint exists — skipping fine-tune"
        echo ""
        continue
    fi

    echo ""
    echo "─── Fine-tuning ${variant} (cache: ${cache}) ───"
    echo ""

    CUDA_VISIBLE_DEVICES=${GPU} \
    HF_HOME="${DINO_HOME}" \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${SCRIPT}" \
        --config "${CONFIG}" \
        --session "${SESSION}" \
        --correspondence_cache "${cache}" \
        --finetune \
        --finetune_output_dir "${out_dir}" \
        --resume_ckpt "${RESUME}" \
        --reset_training_state \
        --max_steps ${MAX_STEPS} \
        --val_every 250 \
        --vis_samples ${VIS_SAMPLES} \
        --gs_reg_loss_weight ${GS_REG} \
        --sample_limit 32

    echo ""
    echo "[${variant}] fine-tune done → ${out_dir}"
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "STEP 1 complete"
echo "═══════════════════════════════════════════════════════════"

# ── Step 2: fixed eval on final checkpoint ────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "STEP 2: Fixed eval on final checkpoint per variant"
echo "═══════════════════════════════════════════════════════════"

echo ""
printf "%-22s %-10s %-10s %-10s %-10s %-10s\n" "Variant" "K(actual)" "PSNR" "L2" "gs_reg" "perceptual"
echo "──────────────────────────────────────────────────────────"

for variant in rigidEarNose11 rigidFace13 rigidFace15 rigidNoEyeBottom17 rigidHead; do
    cache="${CACHE_MAP[$variant]}"
    out_dir="${OUTPUT_ROOT}/${variant}_1k"
    ckpt="${out_dir}/checkpoint_step_01000.pt"
    eval_dir="${out_dir}/fixed_eval_step1000"

    echo ""
    echo "─── Eval ${variant} ───"

    CUDA_VISIBLE_DEVICES=${GPU} \
    HF_HOME="${DINO_HOME}" \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    "${PYTHON_BIN}" "${SCRIPT}" \
        --config "${CONFIG}" \
        --session "${SESSION}" \
        --correspondence_cache "${cache}" \
        --eval \
        --eval_output_dir "${eval_dir}" \
        --resume_ckpt "${ckpt}" \
        --reset_training_state \
        --max_batches ${MAX_BATCHES} \
        --vis_samples ${VIS_SAMPLES} \
        --save_pointclouds \
        --gs_reg_loss_weight ${GS_REG} \
        --sample_limit 32

    # Print table row
    if [ -f "${eval_dir}/metrics.json" ]; then
        python3 - "${eval_dir}/metrics.json" "${variant}" <<'PY'
import json
import sys

metrics_path = sys.argv[1]
variant = sys.argv[2]
with open(metrics_path) as f:
    m = json.load(f)
k_map = {
    'rigidEarNose11': '11',
    'rigidFace13': '13',
    'rigidFace15': '15',
    'rigidNoEyeBottom17': '17',
    'rigidHead': '19',
}
k_str = k_map.get(variant, '?')
print(
    f"{variant:<22} {k_str:<10} "
    f"{m.get('val_psnr', 0):<10.4f} "
    f"{m.get('val_l2', 0):<10.6f} "
    f"{m.get('val_gs_reg', 0):<10.6f} "
    f"{m.get('val_perceptual', 0):<10.6f}"
)
PY
    fi
    echo ""
done

echo "═══════════════════════════════════════════════════════════"
echo "All done. Next: inspect outputs/cheese3d_visual_audit/*_1k/ visuals"
echo "═══════════════════════════════════════════════════════════"
