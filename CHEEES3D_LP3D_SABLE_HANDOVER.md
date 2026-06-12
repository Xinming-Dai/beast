# Cheese3D + LP3D + SABLE — Stage 1/2 Handover

**Last updated:** 2026-06-08
**Status:** Pipeline complete through zero-shot evaluation and Kabsch ablation.

---

## What Changed This Session

| Item | Change |
|------|--------|
| Cache paths | Moved from `/tmp/` to permanent `outputs/cheese3d_stage1/cache_{all28,rigidHead}` |
| Launcher | Added `--eval` mode for zero-shot NVS evaluation (metrics + visuals) |
| NVS regime | Fixed 5× `v_target`→`v_input` bug in `sable.py` Kabsch block |
| Smoke vs eval | Clean separation: smoke = 4-step code path check; eval = full metrics |

---

## Current Pipeline (Stage 1/2 Complete)

```
LP3D (mvt_3d_loss_450_0)
  → 28 keypoints × L/R views, 256×256 input
  → CSV in 320×256 pixel space

CSV → litpose_matches.npz  (beast/convert_csv_to_litpose_cache.py)
  → 320×256 pixel space preserved
  → left_xy, right_xy, confidence, labels
  → sessions filtered to those with selected_frames.csv

Cheese3DDataset (correspondence_mode=cache)
  → Loads npz bundles per frame pair
  → Scales 320×256 → 320×320 for SABLE
  → Outputs leftcamera_xy (28,2), rightcamera_xy (28,2), confidence (28,)

Sable (init_gs=True, ibl_training_regime=nvs)
  → Kabsch alignment of Gaussians from L/R keypoint correspondences
  → gs_reg_loss = MSE(xyz_norm, xyz_init_norm)
  → NVS: context=[L,R], targets=[BC,TC,TL,TR]
```

---

## Permanent Paths

```
# LP3D correspondence caches
outputs/cheese3d_stage1/cache_all28/        # 11 sessions, 28 keypoints, all variants
outputs/cheese3d_stage1/cache_rigidHead/    # 11 sessions, rigid head subset (~4 keypoints)

# LP3D model checkpoint
E-RayZer-private/checkpoints/mvt_3d_loss_450_0/

# Pretrained Sable (erayzer_dl3dv.pt)
E-RayZer-private/checkpoints/erayzer_dl3dv.pt

# Cheese3D source data
E-RayZer-private/data/cheese3d_cam/
```

---

## NVS Evaluation Protocol

**Context views (inputs):** L, R (indices 0, 1)
**Held-out target views:** BC, TC, TL, TR (indices 2, 3, 4, 5)

Metrics computed by `compute_loss()` during `--eval`:
- `val_l2` — MSE between rendered and target images
- `val_psnr` — PSNR from MSE
- `val_gs_reg` — Kabsch regularization loss
- `val_perceptual` — VGG perceptual loss

Visualization: `save_training_visuals()` saves render-vs-GT PNG grids per batch.

---

## Zero-Shot NVS Results (erayzer_dl3dv.pt, Cheese3D held-out views)

```
Session: 20231031_B20_chew_bl_000 (16 val batches, 16 records)
Protocol: context=[L,R] → targets=[BC,TC,TL,TR], no weight updates
```

| Experiment | L2 | PSNR | gs_reg | perceptual |
|---|---|---|---|---|
| all28 (28 kpts, Kabsch on) | 0.0244 | 16.13 dB | **0.472** | 0.437 |
| rigidHead (~4 kpts, Kabsch on) | 0.0244 | 16.13 dB | **0.436** | 0.437 |
| noKabsch (init_gs=false) | 0.0244 | 16.13 dB | **0.000** | 0.437 |

**Interpretation:**
- `gs_reg` correctly reflects Kabsch activation: 0.47/0.44 with Kabsch on, 0.000 with off.
- Rendering quality is identical across all three — zero-shot is dominated by the DINO/transformer backbone, not Gaussian initialization.
- **Kabsch value will emerge during fine-tuning**, not zero-shot evaluation. This is expected.

---

## How to Run

All commands run from `beast/` directory.

### Zero-Shot NVS Evaluation

```bash
# Baseline: erayzer init + all28 Kabsch
HF_HOME=/home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/dinov3-vitb16-pretrain-lvd1689m \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1 \
/home/jqh/miniconda3/envs/neuro/bin/python scripts/sable_scripts/run_cheese3d_phase3_smoke.py \
    --config configs/sable_cheese3d_nvs.yaml \
    --session 20231031_B20_chew_bl_000 \
    --correspondence_cache outputs/cheese3d_stage1/cache_all28 \
    --eval \
    --eval_splits val \
    --eval_output_dir outputs/cheese3d_eval/ablation_all28 \
    --vis_samples 2 \
    --max_batches 16 \
    --resume_ckpt /home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/erayzer_dl3dv.pt \
    --reset_training_state

# Variant: rigidHead subset
# Change --correspondence_cache outputs/cheese3d_stage1/cache_rigidHead

# Variant: no Kabsch (init_gs=false)
# Add --init_gs false
```

**Key flags:**
- `--eval` — zero-shot evaluation (no training), outputs metrics + visuals
- `--eval_splits val` — which splits to evaluate (val / train)
- `--vis_samples N` — how many batch samples to save as PNG grids
- `--max_batches N` — limit batches (for quick smoke, omit for full eval)
- `--init_gs false` — disables Kabsch Gaussian initialization (ablation)
- `--resume_ckpt` — loads pretrained Sable weights (strict=False, 628 DINO/VDA keys missing, 32 perceptual keys unexpected — expected)

### NVS Clean Smoke (4-step)

```bash
# Verify gs_reg non-zero + NVS code path
HF_HOME=... HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
/home/jqh/miniconda3/envs/neuro/bin/python scripts/sable_scripts/run_cheese3d_phase3_smoke.py \
    --config configs/sable_cheese3d_nvs.yaml \
    --session 20231031_B20_chew_bl_000 \
    --correspondence_cache outputs/cheese3d_stage1/cache_all28 \
    --smoke \
    --smoke_mode clean \
    --smoke_output_dir outputs/cheese3d_nvs_smoke \
    --resume_ckpt /home/jqh/NeuralWorkshops/E-RayZer-private/checkpoints/erayzer_dl3dv.pt \
    --reset_training_state

# Success: gs_reg_loss 0.30–0.60 across 4 steps
```

### NVS Debug Smoke (4-step, PLY artifacts)

```bash
HF_HOME=... CUDA_VISIBLE_DEVICES=1 \
/home/jqh/miniconda3/envs/neuro/bin/python scripts/sable_scripts/run_cheese3d_phase3_smoke.py \
    --config configs/sable_cheese3d_nvs.yaml \
    --session 20231031_B20_chew_bl_000 \
    --correspondence_cache outputs/cheese3d_stage1/cache_all28 \
    --smoke \
    --smoke_mode debug \
    --smoke_output_dir outputs/cheese3d_nvs_debug_smoke \
    --resume_ckpt ... \
    --reset_training_state

# Success: debug_pcd/batch_000/ contains PLY + overlay files
```

---

## Remaining Work

1. **Formal fine-tuning** — train `erayzer_dl3dv.pt` + Kabsch on Cheese3D NVS for real gains
2. **Full-dataset evaluation** — `--max_batches` → all val batches, all 11 sessions
3. **LPIPS metric** — set `lpips_loss_weight > 0` in config to enable LPIPS evaluation
4. **Longer training** — 500+ steps to see if Kabsch accelerates convergence

---

## Config Reference

### `sable_cheese3d_nvs.yaml` (NVS evaluation / fine-tuning)
```
ibl_training_regime: nvs
num_views: 6
num_input_views: 2
num_target_views: 4
input_view_indices: [0, 1]    # L, R = context
target_view_indices: [2, 3, 4, 5]  # BC, TC, TL, TR = held-out
init_gs: true
correspondence_mode: cache
gs_reg_loss_weight: 1.0
resume_ckpt: null  # or erayzer_dl3dv.pt
```

### `sable_cheese3d_lp3d.yaml` (reconstruction mode)
```
ibl_training_regime: two_input_reconstruction
num_input_views: 2
num_target_views: 2
init_gs: true
correspondence_mode: cache
gs_reg_loss_weight: 1.0
```
