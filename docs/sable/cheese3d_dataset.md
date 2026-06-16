# Cheese3D dataset for SABLE training

`Cheese3DDataset` (in
[beast/data/sable_dataset.py](../../beast/data/sable_dataset.py)) trains SABLE on
two-view stereo pairs from the Cheese3D multi-camera frame dumps, without requiring
precomputed VDA depth or real correspondence files.

## Expected directory layout

```
{dataset_path}/
  {session_id}/
    TL/
      img00000000.png
      img00000000.npy   # camera intrinsics/extrinsics, ignored
      ...
    TR/
      img00000000.png
      ...
```

For each session in `training.cheese3d_session_names`, the dataset pairs frames from
the left/right camera directories by matching frame index (intersection of indices
present in both). The per-frame `.npy` files are static camera calibration dicts, not
segmentation masks, and are not loaded.

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
  cheese3d_left_camera: TL   # default
  cheese3d_right_camera: TR  # default

model:
  vda:
    mode: online   # required -- see "Depth" below
```

`cheese3d_session_names`, `cheese3d_left_camera`, and `cheese3d_right_camera` are read
directly from the config dict (`extra='allow'` on `SableTrainingConfig`); they are not
declared as typed fields in `beast/config.py`.

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
  cheese3d_use_segmentation: true
  cheese3d_segmentation_root: /work/hdd/bfsr/xdai3/cheese3d_cam/segmentation_masks
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

- only frame indices that have a mask for **both** the left and right cameras (in
  addition to having both PNG frames) are included in the dataset;
- each mask is resized (nearest-neighbor) to `image_size x image_size` and returned
  under the `'mask'` key as a `[V, 1, H, W]` float32 tensor (1 = foreground,
  0 = background);
- `data['image']` always contains the **raw** frames — masks are **not** pre-applied —
  so VDA, the image tokeniser, and DINO all receive full scene context.

The SABLE model (`beast/models/sable.py`) applies the masks in three places:

1. **Gaussian opacity**: background Gaussians (mask = 0) have their logit-opacity
   forced to −10 (sigmoid ≈ 0, fully transparent).  The renderer fills those pixels
   with its default white background `(1, 1, 1)`.
2. **Target image**: the ground-truth frame used in the loss is
   `raw * mask + (1 − mask)`, giving a white background in masked-out regions to
   match the rendered output.
3. **Depth map** (requires `model.vda.mask_depth: true`): after VDA generates depth
   and `pseudo_pointcloud_normalized` normalises it to [−0.5, 0.5], background pixels
   have their Z coordinate forced to 0.5 (the far end).  This ensures background
   Gaussians initialise far from the camera rather than at an arbitrary depth.  Raw
   images are always passed to VDA unmasked; masking is applied only after normalisation.

## Running training

```bash
sbatch scripts/sable_scripts/train_sable_cheese3d.sh
```

`train_sable.py` resolves `training.dataset_name` dynamically (via
`_resolve_dataset_class`), so other dataset classes can be swapped in the same way by
changing this one config field.
