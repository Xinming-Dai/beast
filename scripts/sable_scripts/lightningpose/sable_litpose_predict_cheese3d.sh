#!/bin/bash
#SBATCH -A beez-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH -t 0-24:00:00
#SBATCH -J litpose
#SBATCH -o /u/xdai3/project3d/SBALE_repo/beast/scripts/sable_scripts/lightningpose/litpose_predict_sable_%j.log
#SBATCH --export=ALL

exec 2>&1
source ~/.bashrc
conda activate lp

# lp env has no ffmpeg binary on PATH (only the pip imageio-ffmpeg package); append (not
# prepend) the base anaconda env's bin dir so `ffmpeg` resolves without shadowing lp's python
export PATH="${PATH}:/sw/external/python/anaconda3/bin"

BEAST_REPO=/u/xdai3/project3d/SBALE_repo/beast
LIGHTNING_POSE_REPO=/u/xdai3/project3d/lightning-pose
ROOT=/work/hdd/bfsr/xdai3/cheese3d/videos
LIGHTNING_POSE_MODEL_DIR=/work/nvme/bfsr/xdai3/project3d/twoview3d_ckpts/lightning_pose/outputs_cheese-3d_test_450_LP3D/mvt_3d_loss_450_0
SCRIPT=/u/xdai3/project3d/SBALE_repo/beast/beast/preprocess/sable/run_litpose_predict_cheese3d.py

SESSION_IDS=(
    # "20231031_B20_chew_bl_000",
    # "20231031_B20_chew_temperature_000",
    # "20231031_B21_chew_bl_000",
    # "20231031_B21_chew_temperature_000",
    # "20231031_B26_chew_bl_000",
    # "20231031_B26_chew_temperature_000",
    # "20231031_B31_chew_bl_000",
    # "20231031_B31_chew_temperature_000",
    # "20231031_B6_chew_bl_000",
    # "20231031_B6_chew_temperature_000",
    "20250523_B1_ephys-record_awake_000"
)

# to skip labeled overlay videos, append: --skip-viz

PYTHONPATH="${BEAST_REPO}:${PYTHONPATH}" python -u "${SCRIPT}" \
  --root "${ROOT}" \
  --model-dir "${LIGHTNING_POSE_MODEL_DIR}" \
  --litpose-repo "${LIGHTNING_POSE_REPO}" \
  --session-ids "${SESSION_IDS[@]}" \
  --skip-existing

conda deactivate
