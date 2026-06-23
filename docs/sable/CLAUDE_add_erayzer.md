# Adding E-RayZer Model to Beast

Source repo: `/u/xdai3/project3d/erayzer_cls/E-RayZer-private`
Target repo: `/u/xdai3/project3d/SBALE_repo/beast`

## Files to Move

All imports in `erayzer.py` originate from `erayzer_core.*`. The mapping below flattens that
namespace into `beast.models` and `beast.models.model_utils`.

### Main model

| Source (relative to `src/`) | Destination (relative to `beast/`) |
|---|---|
| `erayzer_core/model/erayzer.py` | `beast/models/erayzer.py` |

### Model utilities → `beast/models/model_utils/`

| Source (relative to `src/erayzer_core/model/`) | Destination filename |
|---|---|
| `gaussians_renderer.py` | `gaussians_renderer.py` |
| `utils_dino.py` | `utils_dino.py` |
| `utils_gaussian.py` | `utils_gaussian.py` |
| `utils_icp.py` | `utils_icp.py` |
| `utils_latent.py` | `utils_latent.py` |
| `utils_pe.py` | `utils_pe.py` |
| `utils_psae.py` | `utils_psae.py` |
| `utils_rot.py` | `utils_rot.py` |
| `utils_transformer.py` | `utils_transformer.py` |
| `utils_vda.py` | `utils_vda.py` |
| `utils_vis.py` | `utils_vis.py` |
| `utils_vits.py` | `utils_vits.py` |

### Training utilities → `beast/models/model_utils/`

| Source (relative to `src/erayzer_core/training/`) | Destination filename |
|---|---|
| `data_utils.py` | `data_utils.py` |
| `latent_partition_cfg.py` | `latent_partition_cfg.py` |
| `losses.py` | `losses.py` |
| `patch_masking.py` | `patch_masking.py` (later deleted — see Post-Migration Removals) |

### Core utilities → `beast/models/model_utils/`

| Source (relative to `src/`) | Destination filename |
|---|---|
| `erayzer_core/camera_overrides.py` | `camera_overrides.py` |
| `erayzer_core/utils/camera_utils.py` | `camera_utils.py` |

## `__init__.py` Files to Create

- `beast/models/model_utils/__init__.py` — create empty (currently missing; pycache exists for some files)

## Import Rewriting Required

Every `erayzer_core.*` import inside the moved files must be rewritten to `beast.*`:

| Old prefix | New prefix |
|---|---|
| `erayzer_core.model.` | `beast.models.model_utils.` |
| `erayzer_core.training.` | `beast.models.model_utils.` |
| `erayzer_core.utils.` | `beast.models.model_utils.` |
| `erayzer_core.camera_overrides` | `beast.models.model_utils.camera_overrides` |
| `.gaussians_renderer` (relative import) | `beast.models.model_utils.gaussians_renderer` (absolute) |

## CLAUDE.md Conventions to Apply During Migration

The source files must be updated to comply with `CLAUDE.md` as they are copied. Do NOT do a
blind copy — each file needs the following fixes applied:

### 1. Replace `easydict` with `types.SimpleNamespace`

`easydict` is not used in beast. All `edict(...)` calls create dot-accessible dicts. The
drop-in replacement is `types.SimpleNamespace` from the standard library.

Files affected: `erayzer.py`, `utils_gaussian.py`, `data_utils.py`, `losses.py`,
`patch_masking.py`.

Replacement pattern:
```python
# Remove:
from easydict import EasyDict as edict
# Add (stdlib, no new dep):
from types import SimpleNamespace

# Replace every occurrence of:
edict(key=val, ...)      →  SimpleNamespace(key=val, ...)
edict({...})             →  SimpleNamespace(**{...})
config: edict            →  config: SimpleNamespace  (or just: dict, since configs are dicts)
```

Note: `patch_masking.py` uses `edict` only in type hints for `config` params — since configs
passed from beast are plain `dict`, replace those annotations with `dict`. This file was
subsequently deleted (see Post-Migration Removals).

### 2. Replace `os` path operations with `pathlib.Path`

Files affected: `gaussians_renderer.py`, `utils_vda.py`, `losses.py`.

Replace `os.path.join(...)`, `os.makedirs(...)`, `os.path.exists(...)`, etc. with `Path`
equivalents. Keep `import os` only if `os` is used for non-path operations (e.g. `os.environ`).

### 3. Modernize `typing` imports

Per CLAUDE.md: use `X | Y`, `list[X]`, `dict[K, V]` syntax; avoid `typing` imports for these.

Files affected: `utils_icp.py`, `utils_vits.py`, `utils_vda.py`, `latent_partition_cfg.py`,
`data_utils.py`, `camera_overrides.py`, `camera_utils.py`, `gaussians_renderer.py`.

| Old | New |
|---|---|
| `from typing import Dict` → remove | use `dict[...]` inline |
| `from typing import List` → remove | use `list[...]` inline |
| `from typing import Optional` → remove | use `X \| None` inline |
| `from typing import Tuple` → remove | use `tuple[...]` inline |
| `from typing import Union` → remove | use `X \| Y` inline |
| `from typing import Any, Mapping, Literal` | keep (no builtin replacement) |

### 4. Replace `.format()` with f-strings

Files affected: `utils_rot.py` (3 occurrences).

```python
# Replace:
"Got {}".format(x)
# With:
f'Got {x}'
```

### 5. Absolute imports (no relative imports)

Files affected: `utils_gaussian.py` uses `from .gaussians_renderer import ...`.

Replace with: `from beast.models.model_utils.gaussians_renderer import ...`

### 6. Use single quotes for strings

Scan all moved files and replace double-quoted strings with single quotes where applicable
(string content permitting). Focus on import lines and short literals — don't alter multi-line
docstrings or strings containing apostrophes.

### 7. Module-level docstrings

Every moved `.py` file must have a module-level docstring describing its purpose.
Add a one-line docstring at the top of any file that is missing one.

### 8. Trailing whitespace and blank-line whitespace

After all edits, ensure no trailing whitespace and no whitespace-only blank lines.
Each `.py` file must end with exactly one newline.

## Post-Migration Removals

The following files and features were removed after the initial migration, and
are **not** present in the beast repo even though they appear in the source repo.

### Files deleted

| File | Reason |
|---|---|
| `beast/models/model_utils/utils_psae.py` | PS-VAE latent partition not needed |
| `beast/models/model_utils/latent_partition_cfg.py` | latent partition config resolver not needed |
| `beast/models/model_utils/camera_overrides.py` | inference-time camera override machinery removed |
| `beast/models/model_utils/patch_masking.py` | masking inlined directly in `ERayZer.forward` (see below) |
| `beast/models/erayzer_lightning.py` | lightning interface merged into `ERayZer` directly |

### Features removed from `erayzer.py`

- **`LatentPartition` / `PoseMapping`** (from `utils_psae`): latent-space
  partitioning into supervised/unsupervised subspaces and supervised pose
  prediction from the latent — all removed. `latent_partition_enabled` flag
  and all `mu_s`, `psae_z`, `pose_pred`, `target_pose` result fields are gone.
- **`DepthEncoder` / `DepthDecoder` classes**: unused depth-encoder classes that
  referenced a non-existent `utils_resnets` module — removed.
- **`depth_loss`**: was always `None` (commented-out code); removed from forward
  result.
- **`camera_control_spec` / `resolve_camera_tensors`**: inference-time camera
  override (GT pose injection) removed. `pred_c2w` and `pred_fxfycxcy` are now
  used directly as `camera_c2w_all` / `camera_fxfycxcy_all`.

### What was added at the same time

- `erayzer.py` forward now returns `xyz_norm` and `xyz_init_norm` (both `None`
  when `model.init_gs: false`) so `LossComputer` receives them directly.
- `beast/models/model_utils/losses.py`: config key renamed from `gs_loss_weight`
  (source repo) to `gs_reg_loss_weight` (beast convention).
- `beast/inference.py`: new `predict_erayzer` function supports multi-view
  inference (pass V image paths per scene, builds `[B, V, 3, H, W]` batch).
- `configs/erayzer_ibl3d.yaml`: example training config adapted from the source
  repo config, with beast class names and `gs_reg_loss_weight` key.

### Lightning merge (`erayzer_lightning.py` → `erayzer.py`)

`ERayZer` now inherits from `BaseLightningModel` directly, consistent with
`VisionTransformer` and `ResnetAutoencoder`. The separate `erayzer_lightning.py`
wrapper has been deleted.

Changes made to `erayzer.py`:
- Base class changed from `nn.Module` to `BaseLightningModel`; `super().__init__(config)`
  replaces `super().__init__()` (no separate `self.config = config` needed).
- `self.loss_computer = LossComputer(config)` added to `__init__`.
- `get_model_outputs`, `compute_loss`, `predict_step` methods moved in from the old wrapper.
- `configure_optimizers` overrides the base implementation to use a step-based
  `OneCycleLR` schedule. It reads from `config['optimizer']` (same section as other
  beast models) using keys `lr`, `wd`, `beta1`, `beta2`, `warmup`, `total_steps`,
  `div_factor`, and `final_div_factor`.

Changes made to `configs/erayzer_ibl3d.yaml`:
- Added `model.model_class: erayzer` (registry lookup key for `beast.api.model.Model`).
- Added `model.seed: 0` (required by `BaseLightningModel.__init__`).
- Added top-level `optimizer:` section with keys `type`, `lr`, `wd`, `beta1`, `beta2`,
  `warmup`, `total_steps`, `div_factor`, `final_div_factor`, `scheduler`. The
  corresponding keys (`beta1`, `beta2`, `lr`, `weight_decay`, `warmup`,
  `scheduler_type`) were removed from `training:`; `max_fwdbwd_passes` remains in
  `training:` as it is also used for checkpointing.

Changes made to `beast/api/model.py`:
- Import updated from `ERayZerLightning` → `ERayZer`; registry entry updated to match.

Changes made to `beast/inference.py`:
- `predict_erayzer` now calls `model.get_model_outputs(batch_dict)` instead of
  `vars(model.model(batch_dict))`.

### Patch masking simplification

ERayZer's patch masking was simplified to match ViT's approach (static `mask_ratio`,
no curriculum, no learnable mask token, no expectation preservation).

Changes made to `erayzer.py`:
- `__init__`: replaced the 18-line `patch_masking` config block (curriculum fields,
  learnable `patch_mask_token`, `preserve_expectation`) with a single line:
  `self.mask_ratio = float(self.config['model'].get('mask_ratio', 0.0))`.
- `forward`: replaced the `apply_patch_token_mask(...)` call with two inline lines:
  ```python
  keep = (torch.rand(img_tokens_input.shape[:-1], device=...) >= self.mask_ratio)
  masked_img_tokens_input = img_tokens_input * keep.unsqueeze(-1).to(img_tokens_input.dtype)
  ```
- `patch_masking.py` import removed; `patch_mask` and `patch_mask_ratio` removed
  from the forward output dict.

Changes made to `configs/erayzer_ibl3d.yaml`:
- Replaced the 7-line `patch_masking:` block with `mask_ratio: 0.75`.

`beast/models/model_utils/patch_masking.py` deleted (no remaining callers).

## New 3rd-Party Dependencies to Add to `pyproject.toml`

Only add what is genuinely needed by the moved files:

| Package | Used by | Notes |
|---|---|---|
| `easydict` | — | **Do NOT add** — replaced by `types.SimpleNamespace` |
| `einops` | `erayzer.py`, `utils_transformer.py`, `utils_vda.py`, `gaussians_renderer.py` | add |
| `gsplat` | `gaussians_renderer.py` | add |
| `open3d` | `erayzer.py`, `utils_icp.py` | add |
| `plyfile` | `gaussians_renderer.py` | add |
| `scipy` | `erayzer.py`, `utils_psae.py` | add |
| `videoio` | `gaussians_renderer.py` | add |

Already in `pyproject.toml`: `jaxtyping`, `torchvision`, `transformers`, `typeguard`,
`opencv-python-headless`.

## Training Pipeline Wiring

The model migration above left training unwired — `beast/train.py` is designed for
vit/resnet (hardcoded `BaseDataset`, epoch-based, `data.data_dir`) and is incompatible
with the erayzer config structure. The following files complete the training integration.

### New files

#### `beast/data/ibl_dataset.py`

`IBLDataset(Dataset)` — loads two-view IBL pairs for ERayZer training.

Supports two `training.dataset_path` layouts:

- **Precache directory** (path is a dir): walks session subdirs, reads
  `pair_metadata.json` per session, filters pairs by `split` field.  The
  constructor accepts `include_splits: list[str] | None` so callers can request
  `['train']` or `['val']` subsets independently.
- **Scene JSON list** (path is a `.txt` file): each line is a path to a scene JSON
  with a `frames` list; no split filtering.

`__getitem__` returns:

| Key | Shape | Source |
|---|---|---|
| `image` | `[V, 3, H, W]` float32 0–1 | images resized to `model.image_tokenizer.image_size` |
| `context_indices` | `[n_ctx]` long | from `ibl_training_regime` |
| `target_indices` | `[n_tgt]` long | from `ibl_training_regime` |
| `depth_vda` | `[V, 1, H, W]` float32 | `{model.vda.cache_root}/{session_id}/{camera}/{frame:06d}.npy` |
| `leftcamera_xy` | `[512, 2]` float32 | `{model.merge_pcd.correspondence_cache_root}/litpose_correspondences/processed_correspondences/{session_id}/correspondences{pair_idx:08d}.npz`; coordinates rescaled to `image_size` space |
| `rightcamera_xy` | `[512, 2]` float32 | same bundle; coordinates rescaled to `image_size` space |
| `confidence` | `[512]` float32 | same bundle; padding slots are `0.0` |
| `scene_name` | str | |

`valid_mask` is **not** produced. Use `confidence > 0` to distinguish real matches from
padding. `ERayZer.forward` does this at line ~741: `valid = data['confidence'][b_i] > 0`.

Camera parameters (c2w, fxfycxcy) are **not** provided — ERayZer predicts them.

Missing correspondence bundles return all-zero tensors; the Kabsch step falls back to
ICP without correspondence hints.

**Coordinate rescaling**: raw `.npz` coordinates are in the original camera pixel space
(left: 256 W × 320 H raw; right: 320 W × 256 H). `_load_image` now returns
`(tensor, orig_w, orig_h)` and `_load_correspondences` scales each axis by
`image_size / orig_dim` before returning, so coordinates are in `[0, image_size]` space
as expected by `pixel_xy_to_pointcloud_flat_indices`.

Config keys read: `training.dataset_path`, `model.merge_pcd.correspondence_cache_root`,
`model.vda.cache_root`, `model.image_tokenizer.image_size`,
`training.ibl_training_regime`.

#### `beast/train_erayzer.py`

`train_erayzer(config, model, output_dir)` — Lightning training loop for ERayZer.

Reuses `reset_seeds`, `pretty_print_config`, and `get_callbacks` from `beast/train.py`.
Appends a step-based `ModelCheckpoint` (every `training.checkpoint_every` steps) on top
of the standard val-best checkpoint from `get_callbacks`.

Key config keys (all under `training:`):

| Key | Purpose |
|---|---|
| `batch_size_per_gpu` | DataLoader batch size |
| `max_fwdbwd_passes` | `Trainer(max_steps=...)` |
| `grad_accum_steps` | `Trainer(accumulate_grad_batches=...)` |
| `use_amp` + `amp_dtype` | `Trainer(precision='bf16-mixed')` when bf16 |
| `val_every` | `Trainer(val_check_interval=...)` |
| `grad_clip_norm` | `Trainer(gradient_clip_val=...)` |
| `checkpoint_every` | step-based periodic checkpoint |
| `resume_ckpt` | `trainer.fit(ckpt_path=...)` |

### Modified files

**`beast/api/model.py`** — `Model.train()` now dispatches by model type:
```python
if isinstance(self.model, ERayZer):
    from beast.train_erayzer import train_erayzer
    self.model = train_erayzer(self.config, self.model, output_dir=self.model_dir)
else:
    self.model = train(self.config, self.model, output_dir=self.model_dir)
```
This means `beast train --config configs/erayzer_ibl3d.yaml` works without any
additional flags.

**`configs/erayzer_ibl3d.yaml`** — three changes:
- `training.dataset_name` updated to `beast.data.ibl_dataset.IBLDataset`
- Added `model.merge_pcd.correspondence_cache_root: null`
- Removed `training.val_dataset_path` and `training.val_split_ratio` — val split
  is determined by the `split` field in `pair_metadata.json`, not a random ratio

### New script

**`scripts/train_erayzer_ibl3d.sh`** — SLURM script equivalent to the old
`E-RayZer-private/scripts/mia/erz_dino/train/erz_train_vda_cache.sh`.

Usage (equivalent mapping):

| Old flag | New mechanism |
|---|---|
| `--config` | `--config configs/erayzer_ibl3d.yaml` |
| `--dataset-path` | `--overrides training.dataset_path=...` |
| `--vda-cache-root` | `--overrides model.vda.cache_root=...` |
| `--correspondence-cache-root` | `--overrides training.correspondence_cache_root=...` |
| `--checkpoint-dir` | `--output ...` |
| `--resume` | `--overrides training.resume_ckpt=...` |
| `--vda-checkpoint-path` | **not needed** — `model.vda.mode: precomputed` skips VDA model load |
| `--device cuda:0` | controlled by `training.num_gpus` in config |

### Correspondence precompute script

**`beast/preprocess/sable/precompute_litpose_correspondences.py`** — self-contained
(no `erayzer_core` imports) replacement for the erayzer-owned
`src/erayzer_core/data/ibl_datasets/precompute_litpose_roma2_bundles.py`.

Reads LitPose DLC-style CSVs from `litpose.video_preds_dir` (set in the extraction config)
and writes one bundle per frame pair to:
`{output_dir}/litpose_correspondences/processed_correspondences/{session_id}/correspondences{pair_idx:08d}.npz`

Session and frame discovery is driven by scanning `input_dir` from the extraction config
(no pair-JSON list needed). Frame indices are read directly from `img*.png` filenames in
`{input_dir}/{cam}Camera.video/_iblrig_{cam}Camera.downsampled.{session_id}/`. Since the
cameras are synchronized, `pair_idx == source_frame_index` for both left and right.

Key differences from the erayzer source script:

- **No `--dataset-list`**: sessions are discovered from `input_dir` in the config.
- Default is `--no-left-frames-stretched` (`left_frames_stretched=False`): left-camera
  coordinates are saved in **raw 256×320 pixel space**. `IBLDataset` rescales them to
  `image_size` at load time (see coordinate rescaling note above).
- Output path uses `correspondences{pair_idx:08d}.npz` (matches `n_digits` from
  `frame.n_digits` in the config, default 8).

Typical invocation — pass the extraction config and everything else is read from it:

```bash
python beast/preprocess/sable/precompute_litpose_correspondences.py \
  --config configs/multiview/extraction_pipeline_sable.yaml
```

A SLURM wrapper is at `scripts/sable_scripts/precompute_litpose_correspondences.sh`.

CLI flags that override config values:

| Flag | Overrides config field |
|---|---|
| `--litpose-root PATH` | `litpose.video_preds_dir` (appends `/video_preds/`) |
| `--output-root PATH` | `output_dir/dataset` |
| `--keypoints A,B,C` | `litpose.keypoints` |
| `--min-likelihood F` | `litpose.min_likelihood` |
| `--shift-nose-leftCamera X,Y` | `litpose.keypoint_shifts.nose.left` |
| `--shift-nose-rightCamera X,Y` | `litpose.keypoint_shifts.nose.right` |
| `--n-digits N` | `frame.n_digits` |
| `--max-workers N` | `max_workers` |
| `--eids EID ...` | `sessionids` |

Set `model.merge_pcd.correspondence_cache_root` to `{output_dir}` so
`IBLDataset` resolves correspondence bundles at the correct path.

### `valid_mask` removed from correspondence data

`valid_mask` (`[N]` bool, first N slots True) has been **removed** from the
correspondence pipeline. It appeared in the erayzer source repo's dataset output,
`data_utils.pad_correspondence_fields_to_batch_max`, and `ERayZer.forward`.

**For any future preprocessing scripts** that produce `.npz` correspondence bundles:
do **not** include a `valid_mask` array. Produce `left_xy`, `right_xy`, and
`confidence` only. Invalid / low-confidence matches should either be excluded before
saving or given `confidence = 0`.

## Inference: PLY Point Cloud Export

Two new functions in `beast/inference.py` add the `--save-pointclouds` capability
from `E-RayZer-private/src/inference.py`:

### `save_gaussian_pointclouds`

```python
from beast.inference import save_gaussian_pointclouds

paths = save_gaussian_pointclouds(
    result,        # dict from model.get_model_outputs(batch)
    output_dir,    # root dir; PLY files go under output_dir/ply/
    batch_idx,     # used in filename
    max_samples=None,  # cap on batch items; None = all
)
```

`result` already contains `gaussians`, `pixelalign_xyz`, and `image` from the
ERayZer forward pass — nothing extra needs to be passed.

**Color rule:**

| Condition | Color source |
|-----------|-------------|
| `pixelalign_xyz` and `image` both present **and** total point counts match (`v_input × H × W == v_all × H × W`) | Per-pixel RGB from input images; y/z axes flipped to match viewer conventions |
| Otherwise | Opacity-as-grayscale |

**Output filename:** `{output_dir}/ply/pointcloud_batch{batch_idx:04d}_sample{sample_idx:02d}.ply`

**open3d dependency:** Uses `o3d.io.write_point_cloud` (binary PLY) when `open3d` is
installed; falls back to ASCII PLY otherwise.

### `infer_erayzer`

Dataset-level inference loop — equivalent to running `src/inference.py
--save-pointclouds` in the original repo.

```python
from beast.inference import infer_erayzer

summary = infer_erayzer(
    config,                        # full beast config dict
    model,                         # trained ERayZer model
    output_dir,                    # root output directory
    save_pointclouds=True,         # write PLY files
    save_visuals=False,            # write render-vs-target PNG grids
    max_batches=None,              # stop early if set
    include_splits=['train', 'val'],  # IBL dataset splits to load
)
# summary keys: 'output_dir', 'num_batches', 'ply_files', 'vis_files'
```

Uses `IBLDataset` + `collate_with_correspondence_padding` (same as `train_erayzer`).
Runs `model.eval()` under `torch.no_grad()`. Moves each batch to the model's device.
