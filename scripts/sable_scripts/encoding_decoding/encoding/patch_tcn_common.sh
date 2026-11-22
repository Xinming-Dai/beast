#!/bin/bash
# Shared setup for the tuned patch-TCN scripts; sourced (not run) by
# step2_tune_patch_tcn.sh and step2_eval_patch_tcn.sh.
#
# Provides: setup_env, resolve_session <78|72|4b> (sets EID, PCA_ROOT),
#           resolve_model <full|nogeom|dino> (sets OUT_DIR, TUNED_DIR, LOCKED_JSON, SHARED_DIR,
#           SHARED_CONFIG).

REPO_ROOT="/u/xdai3/project3d/SBALE_repo/beast"
NEURAL_INPUT_DIR=/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data
CKPT_ROOT=/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts

setup_env() {
    exec 2>&1
    export PYTHONUNBUFFERED=1  # flush print output to the slurm log as it happens
    source ~/.bashrc
    conda activate beast

    # Blackwell 10.0 unsupported by gsplat; use a safe default if missing or 10.0.
    if [[ "${TORCH_CUDA_ARCH_LIST:-}" == *"10.0"* ]] || [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        export TORCH_CUDA_ARCH_LIST="8.0;8.6"
    fi

    [ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc CXX=/usr/bin/g++
    cd "$REPO_ROOT"
}

resolve_session() {
    case "$1" in
        78)
            EID="781b35fd-e1f0-4d14-b2bb-95b7263082bb"
            PCA_ROOT=/work/nvme/bezq/jzhang107/sable_patch_pca_cv5/session_78
            ;;
        72)
            EID="72cb5550-43b4-4ef0-add5-e4adfdfb5e02"
            PCA_ROOT=/work/hdd/bezq/jzhang107/sable/patch_pca_cv5_sessions/session_72
            ;;
        4b)
            EID="4b00df29-3769-43be-bb40-128b1cba6d35"
            PCA_ROOT=/work/hdd/bezq/jzhang107/sable/patch_pca_cv5_sessions/session_4b
            ;;
        *)
            echo "unknown SESSION=$1 (expected 78, 72 or 4b)"
            exit 1
            ;;
    esac
}

# the tuned runs go to <patch_TCN dir>/tuned so the untuned results are never overwritten
resolve_model() {
    case "$1" in
        full)
            OUT_DIR=$CKPT_ROOT/beast_sable/ibl_pretrain_restricted_sessions/21047248/latents/patch_TCN
            ;;
        nogeom)
            OUT_DIR=$CKPT_ROOT/beast_sable/ibl_pretrain_restricted_sessions/22359987_disable_geom/latents/patch_TCN
            ;;
        dino)
            OUT_DIR=$CKPT_ROOT/dino_baseline/latents/patch_TCN
            ;;
        *)
            echo "unknown MODEL=$1 (expected full, nogeom or dino)"
            exit 1
            ;;
    esac
    TUNED_DIR=$OUT_DIR/tuned
    LOCKED_JSON=$TUNED_DIR/locked_config.json
    SHARED_DIR=$OUT_DIR/tuned_shared
    SHARED_CONFIG=$REPO_ROOT/configs/sable/encoding/patch_tcn_shared.yaml
}
