# Cheese3D dataset for SABLE training

`Cheese3DDataset` (in
[beast/data/sable_dataset.py](../../beast/data/sable_dataset.py)) trains SABLE on
multi-view frames from the Cheese3D multi-camera frame dumps, without requiring
precomputed VDA depth or real correspondence files.  The default setup uses two views
(TL + TR); a third center view (TC) can be enabled via config.

## Expected directory layout

```
{dataset_path}/
  {session_id}/
    TL/
      img00000000.png
      img00000000.npy   # camera intrinsics/extrinsics; loaded when use_camera_params: true
      ...
    TR/
      img00000000.png
      ...
    TC/                 # optional; required when cheese3d_center_camera is set
      img00000000.png
      ...
```

For each session in `training.cheese3d_session_names`, the dataset pairs frames from
the camera directories by matching frame index (intersection of indices present in all
configured cameras). The per-frame `.npy` files are static camera calibration dicts.
When `training.use_camera_params: true`, they are loaded and returned as `c2w` and
`fxfycxcy` tensors so the SABLE model uses the calibrated cameras instead of learning
them via its pose predictor.

## Config

See [configs/sable_cheese3d.yaml](../../configs/sable_cheese3d.yaml) for a full
example. Key fields:

```yaml
training:
  dataset_name: beast.data.sable_dataset.Cheese3DDataset
  dataset_path: /work/hdd/bfsr/xdai3/cheese3d_cam/cheese3d_cam
  cheese3d_session_names:
    - 20231031_B20_chew_bl_000
    - ...
  cheese3d_left_camera: TL    # default
  cheese3d_right_camera: TR   # default
  cheese3d_center_camera: TC  # optional; omit or set to null for two-view training
  use_camera_params: true     # load .npy calibration files; false to learn camera params

model:
  vda:
    mode: online   # required -- see "Depth" below
```

When `cheese3d_center_camera` is set, also update `num_views`, `num_input_views`,
`num_target_views`, and `vis_max_views` to `3` in the config.

`cheese3d_session_names`, `cheese3d_left_camera`, `cheese3d_right_camera`, and
`cheese3d_center_camera` are read directly from the config dict (`extra='allow'` on
`SableTrainingConfig`); they are not declared as typed fields in `beast/config.py`.

## Behavior notes

- **Depth**: `__getitem__` always returns zero `depth_vda` tensors. Set
  `model.vda.mode: online` so the SABLE model computes depth from `data['image']`
  itself ([beast/models/sable.py](../../beast/models/sable.py)) — the dataset's zero
  placeholder is unused in this mode.
- **Correspondences**: every scene/camera pair uses the same fixed 3-point set
  (`_CHEESE3D_FIXED_XY`, `_CHEESE3D_FIXED_CONFIDENCE` in `sable_dataset.py`), rescaled
  from native (320x256) pixel space to `model.image_tokenizer.image_size x image_size`.
- **Resizing**: images are stretch-resized (non-aspect-preserving) to
  `image_size x image_size`, matching `training.image_preprocess: stretch`.
- **Train/val split**: `training.val_split_ratio` + `model.seed` give a deterministic
  split via `SABLEDataset._split_records`, shared with `SABLEDataset` and
  `IBLTwoViewDataset`.

## Segmentation masking

SAM3 segmentation masks can optionally be loaded alongside the raw frames:

```yaml
training:
  use_segmentation:
    enabled: true
    cache_root: /work/hdd/bfsr/xdai3/cheese3d_cam/segmentation_masks

model:
  mask_geom_loss: true   # restrict gs_reg_loss to foreground (mouse) points only
  mask_l2_loss: true      # restrict the L2 photometric loss to foreground (mouse) pixels only
```

Expected layout:

```
{segmentation_root}/
  {session_id}_{camera}_{HH-MM-SS}/
    masks/
      mask00000000.png   # binary {0, 255}, single channel
      ...
```

There must be exactly one `{session_id}_{camera}_*` directory per session/camera. When
enabled:

- only frame indices that have a mask for **all configured cameras** (left, right, and
  optionally center, in addition to having the corresponding PNG frames) are included
  in the dataset;
- each mask is resized (nearest-neighbor) to `image_size x image_size` and returned
  under the `'mask'` key as a `[V, 1, H, W]` float32 tensor (1 = foreground,
  0 = background);
- `data['image']` always contains the **raw** frames — masks are **not** pre-applied —
  so VDA, the image tokeniser, and DINO all receive full scene context.

The SABLE model (`beast/models/sable.py`) applies the masks in three places:

1. **L2 photometric loss** (requires `model.mask_l2_loss: true`): the segmentation
   mask is combined into `pixel_mask` (the same per-pixel weight tensor used for
   MAE-style token masking) so that `masked_mse_loss` only accumulates error on
   foreground (mouse) pixels; background pixels contribute zero loss regardless of
   how well they're reconstructed. Set `model.mask_l2_loss: false` to train the L2
   loss over the full frame instead — `target_mask` and `target_gaussian_mask` are
   still computed either way, so `mask_geom_loss`, VDA depth masking, and PLY
   export are unaffected by this flag.
2. **Depth map** (requires `model.vda.mask_depth: true`): after VDA generates depth
   and `pseudo_pointcloud_normalized` normalises it to [−0.5, 0.5], background pixels
   have their Z coordinate forced to −0.5 (the far end).  This ensures background
   Gaussians initialise far from the camera rather than at an arbitrary depth.  Raw
   images are always passed to VDA unmasked; masking is applied only after normalisation.
3. **Geometry loss / gs_reg_loss** (requires `model.mask_geom_loss: true`): the
   SAM3 mask is reshaped into a per-Gaussian weight tensor (same patch-major
   `(hh ww ph pw)` layout as the pixel-aligned point cloud) and combined into the
   `gaussian_mask` argument of `masked_gs_reg_loss`.  This restricts the point-cloud
   regularization loss to foreground (mouse) Gaussians only, ignoring background
   points.  When a token-keep `gaussian_mask` is also active (MAE-style training
   masking), the two masks are combined elementwise so both conditions must hold.

## Running training

```bash
sbatch scripts/sable_scripts/train_sable_cheese3d.sh
```

`train_sable.py` resolves `training.dataset_name` dynamically (via
`_resolve_dataset_class`), so other dataset classes can be swapped in the same way by
changing this one config field.
