# Neural data extraction (per dataset)

This guide documents, per source dataset, how to turn raw neural recordings into the two
artifacts `beast.sable_encoding_decoding` needs (see
[`neural_encoding_decoding.md`](neural_encoding_decoding.md)):

1. `<eid>_aligned.npz` — spikes and trial intervals, split train/val/test.
2. SABLE latents extracted from images sitting in the "eval layout" (`{camera}/{split}/
   interval{N}timebin{M}.png` + `frame_index_mapping.json`), so spike trials and latent trials
   line up row-for-row via `neural_trial_idx`.

Named generically since more than one dataset may extract into this same contract; every
section below is prefixed with the dataset it covers.

---

## Cheese3D: Overview

Cheese3D's ephys session (`20250523_B1_ephys-record_awake_000`, six cameras: `BC`, `L`, `R`,
`TC`, `TL`, `TR`) already has spikes aligned to raw video frames via a hardware-clock trigger
CSV — see `/work/hdd/bfsr/xdai3/cheese3d/README.md` and its `align_ephys.py`. Two beast scripts
turn that into the contract above:

| Step | Script | What it does |
|---|---|---|
| 1 | `beast.preprocess.cheese3d.extract_cheese3d_neural_data` | Bins spikes into 1s trial windows, filters units by firing rate, splits train/val/test, writes `<eid>_aligned.npz` + `frame_manifest.json` |
| 2 | `beast.preprocess.cheese3d.extract_cheese3d_eval_frames` | Extracts the exact corresponding frames from raw session videos into the eval layout, for all six cameras |

SLURM wrappers: `scripts/sable_scripts/encoding_decoding/cheese3d/step0_extract_neural_data.sh`
and `step1_extract_eval_frames.sh` (both CPU-only). `step2_extract_img_tokens.sh` (GPU) then
extracts SABLE latents from the eval-layout frames — see "Cheese3D: Running the pipeline" below.

**Known data issue**: `/work/hdd/bfsr/xdai3/cheese3d/videos_ephys/
20250523_B1_ephys-record_awake_000_TR_18-24-03.mp4` is truncated (ffprobe: "moov atom not
found"; ~30MB vs. an expected ~50MB). An intact copy of all six cameras for this session exists
under `/work/hdd/bfsr/xdai3/cheese3d/videos/` instead (same durations across all cameras,
~1233.97s) — `extract_cheese3d_eval_frames.py` defaults `--raw-video-dir` there for this reason.
This also affects `TR`, the `cheese3d_right_camera` in
`configs/sable/sable_cheese3d_ephys_session.yaml` — worth checking wherever else this session's
raw `TR` video might get read from `videos_ephys/`.

---

## Cheese3D: Trigger CSV and Trial Windows

`extract_cheese3d_neural_data.py` reads the trigger-synchronized alignment already produced by
`align_ephys.py` (`spike/<eid>_100fps.npz`):

| Array | Shape | Description |
|---|---|---|
| `frame_indices` | `(123394, 6)` | 0-based MP4 frame index per trigger, per camera (`BC, L, R, TC, TL, TR` order) |
| `spike_counts` | `(123394, 8)` | Spike count per trigger, per unit |
| `timestamps_s` | `(123394,)` | Session-relative trigger time (seconds) |
| `cluster_names` | `(8,)` | Unit names |

Non-overlapping 1s windows are built over `[0, timestamps_s[-1])`; per-window spike counts are
summed from the trigger rows falling inside the window, and the representative per-camera frame
index is taken from the window's first trigger row (matching `align_ephys.py`'s own
`downsample(..., reduce='first')` behavior when it down-samples to a lower target fps). Units are
kept when their mean firing rate over the full session exceeds `--fr-thresh` (Hz; default `0.2`
— deliberately lower than IBL's convention, since Cheese3D units run 0.4–30Hz vs. IBL's typical
population, see `/work/hdd/bfsr/xdai3/cheese3d/README.md`). Trials are shuffled (seed 42) and
split 70/10/20 train/val/test.

### `<eid>_aligned.npz` keys

| Key pattern | Content |
|---|---|
| `train_spikes`, `val_spikes`, `test_spikes` | `[K, 1, n_units_kept]` int32 — one timestep (`T=1`) per 1s trial |
| `train_intervals`, `val_intervals`, `test_intervals` | `[K, 2]` float64 — `[t_start, t_end]` seconds |

`params.json` (written alongside) records `fr_thresh`, `trial_len_sec`, `cluster_names_kept`,
split sizes, and the seed.

### `frame_manifest.json`

One file per session, listing every trial in every split with its raw per-camera frame index:

```json
{
  "eid": "20250523_B1_ephys-record_awake_000",
  "view_names": ["BC", "L", "R", "TC", "TL", "TR"],
  "splits": {
    "train": [
      {
        "neural_trial_idx": 0,
        "neural_bin_idx": 0,
        "neural_interval_sec": [355.0, 356.0],
        "frame_index": {"BC": 35522, "L": 35522, "R": 35523, "TC": 35522, "TL": 35523, "TR": 35522}
      }
    ],
    "val": [...],
    "test": [...]
  }
}
```

Keeping all six cameras' frame indices here (not just whichever pair a given SABLE config
trains on) means a future config with a different camera pairing can reuse this manifest
directly, without re-deriving trial windows from the trigger CSV.

---

## Cheese3D: Eval-Frame Extraction

`extract_cheese3d_eval_frames.py` reads `frame_manifest.json` and, for each camera (all six by
default; restrict with `--cameras`), does a single ffmpeg `select`-filter decode pass over the
raw session video to pull out exactly the frames named in the manifest — far cheaper than
seeking per frame. Frames are resized to `320x256` (matching the existing `cheese3d_cam`
convention) and written to:

```
{output_dir}/{eid}/{camera}/{split}/interval{trial_idx}timebin0.png
{output_dir}/{eid}/{camera}/{split}/interval{trial_idx}timebin0.npy   # static per-camera calibration
```

The `.npy` sidecar (copied verbatim from the already-extracted `cheese3d_cam` calibration file
for that camera — identical for every frame) is required even though
`training.use_camera_params: false` in `sable_cheese3d_ephys_session.yaml`, because
`training.load_gt_camera_params_for_vis: true` in that same config unconditionally loads it for
visualization, and `beast predict` has no config-override mechanism to disable that per run —
see `Cheese3DDataset._load_camera_params` in `beast/data/sable_dataset.py`.

Because `beast.data.sable_dataset.SABLEDataset._discover_eval_split_records` (the eval-layout
reader shared by `IBLTwoViewDataset` and `Cheese3DDataset`) keys frame indices by stereo *role*
(`left_source_frame_index` / `right_source_frame_index`), not by camera name, the script also
writes a role-keyed `frame_index_mapping.json` into the `--left-camera`/`--right-camera` (and
optional `--center-camera`) directories only:

```
{camera_dir}/{split}/frame_index_mapping.json
```

```json
{
  "interval0timebin0.png": {
    "left_source_frame_index": 35523,
    "neural_trial_idx": 0,
    "neural_bin_idx": 0,
    "neural_interval_sec": [355.0, 356.0]
  }
}
```

The other extracted cameras' frames sit on disk without a mapping file until a future run
assigns them a role (re-running this script with different `--left-camera`/`--right-camera`
values is cheap — no video re-decode needed if those cameras were already extracted; only
the mapping-file-writing step reruns).

---

## Cheese3D: `Cheese3DDataset` Eval-Layout Support

`Cheese3DDataset._load_records` (in `beast/data/sable_dataset.py`) falls back to the eval layout
per session automatically: when a session's configured left-camera directory has no flat
`img*.png` files, it looks for `{split}/frame_index_mapping.json` instead (via the same
`SABLEDataset._discover_eval_split_records` that `IBLTwoViewDataset` uses for IBL's own eval
layout). Eval-layout sessions carry a fixed on-disk train/val/test split and bypass
`training.val_split_ratio` entirely. Like `IBLTwoViewDataset`, eval-layout support is two-camera
only for now — a configured `cheese3d_center_camera` is ignored (with a warning) for
eval-layout sessions.

---

## Cheese3D: Running the pipeline

```bash
sbatch scripts/sable_scripts/encoding_decoding/cheese3d/step0_extract_neural_data.sh
sbatch scripts/sable_scripts/encoding_decoding/cheese3d/step1_extract_eval_frames.sh
sbatch --export=ALL,MODEL_DIR=/path/to/trained/cheese3d_ephys_session/checkpoint \
  scripts/sable_scripts/encoding_decoding/cheese3d/step2_extract_img_tokens.sh
```

From there, stages 3 onward (PCA compression, encoding, decoding, unprojection, render, video)
reuse the existing generic scripts under `scripts/sable_scripts/encoding_decoding/{img_token,
encoding}/` unchanged — see [`neural_encoding_decoding.md`](neural_encoding_decoding.md) — with
`EID=20250523_B1_ephys-record_awake_000` and `NEURAL_INPUT_DIR` pointed at step 0's
`--neural-output-dir`.

---

## See Also

- [`neural_encoding_decoding.md`](neural_encoding_decoding.md) — the downstream encoding/decoding
  pipeline that consumes the artifacts documented here.
- [`cheese3d_dataset.md`](cheese3d_dataset.md) — `Cheese3DDataset`'s training-layout behavior.
- `/u/xdai3/project3d/erayzer_cls/E-RayZer-private/docs/ibl_neural_behavior_extraction.md` — the
  IBL analog (external repo; not yet ported into this doc).
