# Extracting Data for SABLE

This guide walks through preparing an IBL stereo dataset for SABLE model training.
The pipeline takes raw synchronized multi-camera videos from an IBL-2view directory
and produces a self-contained dataset of extracted frame pairs with optional
precomputed VDA depth maps and LitPose keypoint correspondences.

---

## Overview

| Step | Config flag | What it does |
|------|-------------|--------------|
| Stats | always runs | Scans input videos; writes `video_stats.csv` and `video_stats.json` |
| Trim | `trim.enabled` | Clips every video to a fixed frame/time range |
| Downsample | `downsample.enabled` | Re-encodes videos at a lower frame rate |
| Extract | always runs | Selects frames via k-means on anchor view, exports images |
| VDA depth | `vda.enabled` | Precomputes per-frame depth maps alongside images |
| LitPose CSV→npy | `litpose.enabled` | Converts LitPose CSVs to per-frame correspondence `.npy` files |

LitPose predict (the step that generates the CSVs) is run **separately** in the `lp`
conda environment using `beast/preprocess/run_litpose_predict_sable.py`.

Run all steps (except LitPose predict) with one command:

```bash
beast extract_sable --config configs/multiview/extraction_pipeline_sable.yaml
```

Pass `--skip-stats` to skip the stats scan if you have already run it:

```bash
beast extract_sable --config configs/multiview/extraction_pipeline_sable.yaml --skip-stats
```

Override session IDs from the command line (takes precedence over the config):

```bash
beast extract_sable --config configs/multiview/extraction_pipeline_sable.yaml \
  --sessionids 03d9a098-07bf-4765-88b7-85f8d8f620cc 1735d2be-b388-411a-896a-60b01eaa1cfe
```

A complete, annotated config is at
[configs/multiview/extraction_pipeline_sable.yaml](../configs/multiview/extraction_pipeline_sable.yaml).

---

## Input Data

### Directory layout

```
IBL-2view/
├── leftCamera.video/
│   └── _iblrig_leftCamera.downsampled.<session_id>.mp4
├── rightCamera.video/
│   └── _iblrig_rightCamera.downsampled.<session_id>.mp4
└── timestamps/                          # optional; stored as metadata if present
    ├── _ibl_leftCamera.times.<session_id>.npy
    └── _ibl_rightCamera.times.<session_id>.npy
```

Set `input_dir` in the config to the root of this layout.  The file naming convention
is controlled by template constants at the top of
`beast/preprocess/extraction_sable.py` — edit those to adapt to different naming.

---

## Step 1 (optional): LitPose predict

If you want precomputed keypoint correspondences, run LitPose predict first to generate
per-session CSVs. The predict script lives at
`beast/preprocess/sable/run_litpose_predict_sable.py`.

### 1.1 Environment

Create a `lp` conda environment using the
[Lightning Pose installation guide](https://lightning-pose.readthedocs.io/en/latest/source/installation_guide.html#step-2-create-conda-environment).

Alternatively, pass `--litpose-repo` (see §1.3) to run directly from the Lightning Pose
source repo without activating a separate environment.

### 1.2 Model directory layout

The `--model-dir` argument should point to a trained Lightning Pose model directory:

```text
<model_dir>/
├── config.yaml
├── tb_logs/         ← model weights
└── video_preds/     ← prediction CSVs are written here after litpose predict
```

### 1.3 Run prediction

**Option A — via `lp` conda environment:**

```bash
conda activate lp
python beast/preprocess/sable/run_litpose_predict_sable.py \
  --root /work/hdd/bfsr/xdai3/IBL-2view \
  --model-dir /path/to/lightning_pose_model \
  --config configs/multiview/extraction_pipeline_sable.yaml \
  [--session-ids <eid1> <eid2>] \
  [--skip-existing] \
  [-- --skip_viz]
```

**Option B — via Lightning Pose source repo (no env activation needed):**

```bash
python beast/preprocess/sable/run_litpose_predict_sable.py \
  --root /work/hdd/bfsr/xdai3/IBL-2view \
  --model-dir /path/to/lightning_pose_model \
  --config configs/multiview/extraction_pipeline_sable.yaml \
  --litpose-repo /u/xdai3/project3d/lightning-pose \
  [-- --skip_viz]
```

With `--litpose-repo`, the script calls
`python -m lightning_pose.cli.main predict` with `PYTHONPATH` set to the repo — no
`litpose` binary or `lp` environment activation required.

`--config` reads `cameras` and `sessionids` from the extraction config so you don't have
to repeat them. `--session-ids` on the command line takes precedence over the config value.

### 1.4 Outputs

Lightning Pose writes under the model directory:

- `<model_dir>/video_preds/<mp4_stem>.csv` — per-camera keypoint predictions
- `<model_dir>/video_preds/labeled_videos/` — overlay videos (skip with `--skip_viz`)

### 1.5 Extra `litpose predict` options

Anything after `--` is forwarded to `litpose predict`:

```bash
python beast/preprocess/sable/run_litpose_predict_sable.py \
  --root /work/hdd/bfsr/xdai3/IBL-2view \
  --model-dir /path/to/lightning_pose_model \
  --config configs/multiview/extraction_pipeline_sable.yaml \
  -- --skip_viz
```

`--skip_viz` skips labeled overlay MP4s and keeps only the CSV predictions.

### 1.6 Argument reference

| Argument | Description |
|---|---|
| `--root` | IBL-2view root with per-camera video subdirectories |
| `--model-dir` | Lightning Pose model directory |
| `--config` | Path to `extraction_pipeline_sable.yaml`; cameras and sessionids are read from it |
| `--session-ids` / `--only-eids` | Optional session ID subset; overrides config `sessionids` |
| `--output-dir` | Optional: copy per-session CSVs here after each run |
| `--litpose-repo` | Lightning Pose source repo directory; runs via `python -m lightning_pose.cli.main` (mutually exclusive with `--litpose-bin`) |
| `--litpose-bin` | Explicit `litpose` executable path (default: `litpose`; mutually exclusive with `--litpose-repo`) |
| `--skip-existing` | Skip sessions whose prediction CSVs already exist |
| `--dry-run` | Print commands without running |
| After `--` | Passed through to `litpose predict` |

### After prediction

```yaml
litpose:
  enabled: true
  video_preds_dir: /path/to/lightning_pose_model/video_preds
```

Then run `beast extract_sable` with `litpose.enabled: true` to convert the CSVs to
per-frame `.npz` bundles.

---

## Step 2: Run the pipeline

```bash
beast extract_sable --config configs/multiview/extraction_pipeline_sable.yaml
```

The pipeline runs in sequence:

1. **Stats** — scans all session videos; writes `video_stats.csv` and `video_stats.json`
2. **Trim** (if `trim.enabled`) — clips videos; output to `videos_trim/`
3. **Downsample** (if `downsample.enabled`) — reduces frame rate; output to `videos/`
4. **Extract** — runs `select_frame_idxs_kmeans` on the anchor view to select
   `frame.frames_per_video` diverse frames via PCA k-means; applies the same frame
   indices to all cameras; writes `pair_metadata.json` per session
5. **VDA depth** (if `vda.enabled`) — runs Video Depth Anything inference on extracted
   frames; saves depth maps co-located with images
6. **LitPose CSV→npy** (if `litpose.enabled`) — reads LitPose CSVs from
   `litpose.video_preds_dir`, applies optional keypoint shifts, saves per-frame
   correspondence arrays co-located with images

---

## Output Structure

```
output_dir/
├── video_stats.csv              # per-video stats (fps, resolution, frame count)
├── video_stats.json             # aggregate summary stats
├── videos_trim/                 # (trim.enabled) trimmed videos
│   ├── leftCamera/
│   └── rightCamera/
├── videos/                      # (downsample.enabled) downsampled videos
│   ├── leftCamera/
│   └── rightCamera/
└── dataset/                     # point training.dataset_path here
    ├── info.json                # dataset-level metadata
    ├── {session_id}/
    │   ├── pair_metadata.json   # list of frame pairs with splits and frame indices
    │   ├── left/
    │   │   └── img00000042.png  # extracted frame (native resolution)
    │   └── right/
    │       └── img00000042.png  # same frame index as left
    ├── depth_map/               # (vda.enabled)
    │   └── {session_id}/
    │       ├── left/
    │       │   └── depth00000042.npy   # float32 depth map
    │       └── right/
    │           └── depth00000042.npy
    └── litpose_correspondences/ # (litpose.enabled)
        └── processed_correspondences/
            └── {session_id}/
                └── correspondences00000042.npz  # float32 .npz: left_xy [K,2], right_xy [K,2], confidence [K]
```

`pair_metadata.json` format:

```json
{
  "session_id": "03d9a098-07bf-4765-88b7-85f8d8f620cc",
  "pairs": [
    {
      "pair_idx": 0,
      "split": "train",
      "left_path": "left/img00000042.png",
      "right_path": "right/img00000042.png",
      "left_source_frame_index": 42,
      "right_source_frame_index": 42,
      "left_timestamp_sec": 1.234,     (optional, from timestamps/*.npy)
      "right_timestamp_sec": 1.234
    }
  ]
}
```

---

## Training Dataset Class

Point `training.dataset_path` to `output_dir/dataset/` in your training config.
`depth_map/` and `litpose_correspondences/` are resolved relative to this path.

Use `IBLTwoViewDataset` for the `extract_sable` pipeline output — it reads VDA depth
from `depth_map/` and correspondences from `litpose_correspondences/`:

```yaml
training:
  dataset_name: beast.data.sable_dataset.IBLTwoViewDataset
  dataset_path: /path/to/output_dir/dataset
  training_regime: all_views_reconstruction
  val_split_ratio: 0.1

model:
  image_tokenizer:
    image_size: 320
```

Alternatively, `SABLEDataset` also works with `pair_metadata.json` but uses a
separate `model.vda.cache_root`.

---

## Configuration Reference

All settings live in `configs/multiview/extraction_pipeline_sable.yaml`.

### Top-level fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | `''` | Short dataset name written into `dataset/info.json` |
| `input_dir` | `''` | IBL-2view root directory |
| `output_dir` | `''` | All pipeline outputs are written here |
| `anchor_view` | `left` | Camera whose frames drive k-means selection; must be in `cameras` |
| `cameras` | `[left, right]` | Cameras to extract; same frame index is applied to all |
| `max_workers` | `4` | Parallel workers for trim and downsample steps |
| `author` | `anonymous` | Written into `dataset/info.json` |
| `seed` | `42` | Random seed |
| `sessionids` | `null` | Session ID filter; `null` = process all sessions |

### `frame`

| Field | Default | Description |
|-------|---------|-------------|
| `frames_per_video` | `2000` | Frames to select per session (k-means on anchor view) |
| `n_digits` | `8` | Zero-padding width for exported filenames, e.g. `img00000042.png` |
| `extension` | `png` | Image format for exported frames |
| `kmeans_resize` | `32` | Frame is resized to this square size before k-means pixel clustering |
| `split_names` | `[train, val]` | Split names |
| `split_ratios` | `[0.9, 0.1]` | Fraction of pairs per split |

### `trim`

Optional. Disabled by default.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Set to `true` to trim all videos |
| `start_frame` | `null` | First frame to keep (frame bounds take priority over second bounds) |
| `end_frame` | `null` | Last frame to keep |
| `start_sec` | `null` | Start time in seconds |
| `end_sec` | `null` | End time in seconds |
| `ffmpeg_threads` | `null` | Threads per ffmpeg process; `null` = `cpu_count / max_workers` |

### `downsample`

Optional. Disabled by default.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Set to `true` to downsample all videos |
| `target_fps` | `null` | Target frame rate; `null` keeps original FPS |
| `ffmpeg_threads` | `null` | Threads per ffmpeg process |
| `phase_offset_frames` | `0` | Skip N frames at start before downsampling begins |

### `video_naming`

Controls IBL-2view directory layout and file naming. Defaults assume IBL naming conventions.

| Field | Default | Description |
|-------|---------|-------------|
| `camera_video_subdir` | `{cam}Camera.video` | Subdirectory pattern for each camera's videos |
| `video_filename` | `_iblrig_{cam}Camera.downsampled.{session_id}.mp4` | Video file pattern |
| `timestamp_filename` | `_ibl_{cam}Camera.times.{session_id}.npy` | Timestamp file pattern |
| `timestamp_subdir` | `timestamps` | Subdirectory containing timestamp files |

**Example: custom layout**

```yaml
video_naming:
  camera_video_subdir: 'videos/{cam}'
  video_filename: '{session_id}_{cam}.mp4'
  timestamp_filename: '{session_id}_{cam}_times.npy'
  timestamp_subdir: 'timestamps'
```

### `vda`

VDA depth precomputation, runs after the extract step.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `true` | Set to `false` to skip VDA precomputation |
| `encoder` | `vitb` | VDA encoder variant: `vits`, `vitb`, or `vitl` |
| `checkpoint_path` | `null` | Path to VDA `.pth` checkpoint. If `null`, searches `third_party/VDA/checkpoints/video_depth_anything_{encoder}.pth` |
| `device` | `auto` | Inference device: `cuda`, `cpu`, or `auto` |
| `input_size` | `518` | VDA inference spatial resolution |
| `fp32` | `false` | Disable autocast (use full FP32) |

Output: `dataset/depth_map/{session_id}/{cam}/depth{n_digits}.npy` (float32 depth map).

### `litpose`

LitPose CSV→npy conversion. LitPose predict must be run separately first.

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Set to `true` to convert LitPose CSVs |
| `video_preds_dir` | `null` | Directory containing LitPose prediction CSVs |
| `keypoints` | `[pawL, pawR, nose]` | Keypoints to extract; defines K |
| `min_likelihood` | `0.0` | Keypoints below this threshold have `confidence = 0` |
| `keypoint_shifts` | `{}` | Per-keypoint per-camera pixel shift |

`keypoint_shifts` example (works with any camera names):

```yaml
keypoint_shifts:
  nose:
    left: [5, 0]     # shift nose x by +5, y by 0 for left camera
    right: [-12, 0]
  pawL:
    front: [0, 2]    # works with arbitrary camera names like 'front', 'side', etc.
    side: [1, 0]
```

Output: `dataset/litpose_correspondences/processed_correspondences/{session_id}/correspondences{n_digits}.npz`
(float32 `.npz` bundle: `left_xy [K, 2]`, `right_xy [K, 2]`, `confidence [K]`).

---

## Tips

**VDA checkpoint not found**
If the pipeline fails with "checkpoint not found", either:
1. Specify `vda.checkpoint_path` in your config pointing to the `.pth` file
2. Place the checkpoint in `third_party/VDA/checkpoints/video_depth_anything_{encoder}.pth`
   (e.g., `third_party/VDA/checkpoints/video_depth_anything_vitb.pth` for encoder `vitb`)

**Filtering sessions**
Set `sessionids` in the config to avoid passing `--sessionids` on every call:

```yaml
sessionids: [03d9a098-07bf-4765-88b7-85f8d8f620cc]
```

**Re-running after interruption**
The extract step skips sessions whose `pair_metadata.json` already exists.
Pass `--overwrite` to force re-extraction.  VDA and LitPose steps similarly
skip existing `.npy` files unless `--overwrite` is set.

**Adapting to different naming conventions**
Edit the template constants at the top of `beast/preprocess/extraction_sable.py`:
- `_CAMERA_VIDEO_SUBDIR_TMPL` — subdirectory for each camera's videos
- `_VIDEO_FILENAME_TMPL` — video filename pattern
- `_TS_FILENAME_TMPL` — timestamp filename pattern
