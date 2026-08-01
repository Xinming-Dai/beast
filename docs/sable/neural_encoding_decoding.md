# Neural Encoding/Decoding from SABLE Image Tokens

This guide walks through the pipeline that connects neural spiking data (e.g. IBL Neuropixels,
Cheese3D) to SABLE's learned per-patch image tokens: fitting encoding models (spikes-from-latents)
and decoding models (latents-from-spikes), compressing image tokens with PCA to make decoding
tractable, and finally reconstructing / rendering decoded tokens back into images and videos. The
pipeline is dataset-agnostic — any dataset with SABLE latents and aligned spiking data can use it.

All stages live under `beast.sable_encoding_decoding` and are optional — they are not imported
by any core `beast` training/inference code.

---

## SABLE's Latent Export Gates

`Sable.forward` can optionally export four latent tensors, each gated by a `return_*` flag that
can be set per-batch (`data['return_*'] = True`) or statically in config (`model.return_*: true`).
Every gate is `None` unless requested, and only ever populated during inference/eval, never
during training:

| Result field | Gate | What it is |
|---|---|---|
| `result.frame_z` | `return_frame_cls_tokens` | CLS token *after* the VGGT/geometry encoder (3D-aware) |
| `result.dino_z` | `return_dino_cls` | Raw DINO featurizer CLS token, *before* the geometry encoder |
| `result.combined_z` | `return_combined_z` | `cat([frame_z, dino_z], dim=-1)` — the `cat` latent kind below |
| `result.img_tokens` | `return_img_tokens` | Full per-patch geometry-encoder tokens (post-fusion) |

The easiest way to extract them for a whole dataset is `beast.inference.extract_sable_latents`, or
the CLI:

```bash
beast predict --model $MODEL_ROOT --input $DATASET_PATH --extract-latents \
    --return-all-z --return-img-tokens
```

`--return-all-z` is shorthand for `--return-frame-z --return-dino-z --return-cat-z`; each can also
be requested individually (`--return-frame-z`, `--return-dino-z`, `--return-cat-z`,
`--return-img-tokens`, any combination). This saves one `.npz` per batch per session under
`<output_dir>/<latent_type>/<session_id>/<split>/` (a batch whose rows span a session boundary is
split into one file per session), then combines each session's batches into
`<output_dir>/<latent_type>/<session_id>/<latent_type>_trials.npz` — the exact layout Stage 1's
`--latent_kind` table below expects. `img_tokens` is never combined this way (see the note under
Stage 2) — its raw per-batch shards are consumed directly by `img_token.run_pca_and_save`. Once a
session's combine succeeds, its per-batch files are deleted automatically (except for
`img_tokens`); resuming a run whose session was already combined and cleaned up skips it entirely
rather than rerunning the model.

---

## Installation

The pipeline needs a few extra packages beyond base beast (Ray Tune for hyperparameter search,
`facemap`, `torcheval`, `accelerate`, `torchmetrics[image]`, `imageio[ffmpeg]`). Install the
optional extra:

```bash
pip install -e ".[sable_encoding_decoding]"
```

This mirrors how the `vda` extra is installed elsewhere in this repo — base `beast` installs
are unaffected if you skip this.

---

## Overview

| Stage | Module | What it does |
|---|---|---|
| 1 | `neural.run_encoding_decoding` | Fit RRR + CNN/TCN encoding or decoding models between spikes and a chosen latent (frame embedding, pose, image tokens, ...) |
| 2 | `img_token.run_pca_and_save` | Compress per-patch SABLE image tokens with PCA so they're small enough to decode from spikes |
| 3 | `neural.run_encoding_decoding` (reused) | Decode PCA-compressed image tokens from spikes, same CLI as stage 1 with `--latent_kind img_tokens_compressed*` |
| 4 | `img_token.unproject` | Un-PCA + un-normalize the decoded compressed tokens back to full-dimensional image tokens |
| 5 | `render.decode_and_render` | Feed (real or decoded) image tokens through SABLE's decoder, render images, compute PSNR/SSIM |
| 6 | `video.video_generator` | Stitch a folder of rendered frames into an MP4 |

A typical use case runs stages 1-2-3-4-5 to answer "how well can spikes decode the SABLE latent
that reconstructs the actual image," then stage 6 to turn the reconstructed frames into a
video for visual inspection.

---

## Stage 1 & 3: Neural Encoding / Decoding

`beast.sable_encoding_decoding.neural.run_encoding_decoding` fits both a reduced-rank-regression
(RRR) model and a CNN/TCN model, sweeping hyperparameters via Ray Tune, then evaluates on held
out trials. In **encoding** mode it predicts spike rates from a latent; in **decoding** mode it
predicts the latent from spike rates.

```bash
python -m beast.sable_encoding_decoding.neural.run_encoding_decoding \
    --eid $EID \
    --neural_input_dir $NEURAL_ROOT \
    --latent_input_dir $LATENT_ROOT \
    --eval_task encoding \
    --latent_kind frame
```

Key flags (see `neural/utils.py::get_encoding_decoding_args` for the full list):

| Flag | Meaning |
|---|---|
| `--eid` | session / animal id (subfolder name under both `--neural_input_dir` and `--latent_input_dir`) |
| `--neural_input_dir` | root directory of neural (spike) data; the session is read from `<neural_input_dir>/<eid>` |
| `--latent_input_dir` | root directory of latent data |
| `--eval_task` | `encoding` or `decoding` |
| `--latent_kind` | which latent to use — see table below. Omit to read `<latent_input_dir>/<eid>/z_trials.npz` directly |
| `--model_config` | training/inference YAML; required only for `--latent_kind psae` or `mu_u` (reads `model.auto_encoder.num_latents`) |
| `--result_name` | override the output `.npy` basename (default derived from `--eval_task`/`--latent_kind`) |
| `--tune_storage_path` | Ray Tune experiment root; defaults under the latent-kind subdirectory |
| `--seed` | RNG seed (default 42) |

`--latent_kind` layout (each resolves to `<latent_input_dir>/<subdir>/<eid>/<trials_npz>`):

| `--latent_kind` | Subdirectory | Trials file |
|---|---|---|
| `frame` | `frame_z` | `frame_z_trials.npz` |
| `mu_s` | `pose_mu_s_z` | `pose_mu_s_z_trials.npz` |
| `psae` | `psae_z` | `psae_z_trials.npz` (full latent; requires `--model_config`) |
| `mu_u` | `psae_z` | `psae_z_trials.npz`, sliced to the unsupervised tail (requires `--model_config`) |
| `dino` | `dino_z` | `dino_z_trials.npz` |
| `cat` | `cat_z` | `cat_z_trials.npz` |
| `behavior` | `behavior_z` | `behavior_z_trials.npz` (Cheese3D raw TL/TR keypoint traces; see [`neural_extraction.md`](neural_extraction.md)) |
| `img_tokens_compressed*` (e.g. `img_tokens_compressed_3_comp`) | `<latent_kind>` | `img_tokens_compressed_trials.npz` (PCA-compressed image tokens from stage 2; CNN/TCN only, no RRR) |
| (omitted) | `<latent_input_dir>` directly | `z_trials.npz` |

**Output**: an `.npy` (basename `encoding_results`/`decoding_results`, optionally suffixed with
`_<latent_kind>`) saved under the resolved latent session directory, containing both the RRR
and CNN/TCN results (test-set metrics + predictions).

To decode PCA-compressed image tokens (stage 3), rerun this same command with
`--eval_task decoding --latent_kind img_tokens_compressed` (or whatever name you chose for the
PCA output subdirectory in stage 2) — the output is
`decoding_results_img_tokens_compressed.npy`, consumed by stage 4.

---

## Stage 2: Image-Token PCA Compression

Full per-patch SABLE image tokens are too high-dimensional to decode directly from spikes.
`beast.sable_encoding_decoding.img_token.run_pca_and_save` assembles trial-aligned image-token
tensors from SABLE inference output, fits a PCA on the **train** split, and projects
train/val/test into the compressed space. In a multisession layout
(`<input-dir>/<session_name>/<train|val|test>/...`), each session is fit and applied
**independently** — a session's own train split fits its own PCA basis, never pooled with other
sessions.

```bash
python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \
    --input-dir $MODEL_ROOT/img_tokens \
    --stage all \
    --n-feat-keep 3 \
    --model-root $MODEL_ROOT
```

Key flags (see `img_token/run_pca_and_save.py::parse_args`):

| Flag | Meaning |
|---|---|
| `--input-dir` | inference directory of `img_tokens_batch*.npz` / `img_tokens_chunk*.npz` shards to assemble into trials |
| `--session-names` | space-separated session/EID names to process (each fit independently); defaults to auto-discovering every immediate subfolder of `--input-dir` |
| `--combined-trials-{train,val,test}-npz` | skip assembling a given split from `--input-dir`; use a pre-assembled trials `.npz` instead |
| `--stage` | `1` (fit PCA on train only), `2` (project val/test using stage-1's PCA + train data), or `all` (both in one pass, default) |
| `--n-feat-keep` | number of PCA components to keep (default 3) |
| `--model-root` | anchor directory; when `--output-*` are omitted, defaults are written under `<model-root>/img_tokens_compressed/` |
| `--output-pca-npz` | explicit output path for the PCA bundle |
| `--output-trials-npz` | explicit output path for the compressed trials `.npz` |
| `--pair-metadata` | optional `pair_metadata.json` for IBL neural-aligned precache session partitioning |
| `--include-splits` | comma-separated splits to include (default `train,val,test`) |

Run stage 1 alone when val/test image tokens are not yet available (e.g. inference is still
running), then stage 2 once they are — stage 2 reuses each session's stage-1 PCA fit rather than
refitting. Both stages must be given the same `--session-names` (or the same auto-discoverable
`--input-dir` tree) so the two runs line up session-for-session.

`--input-dir` points at the raw `img_tokens_batch*.npz` shards `extract_sable_latents` writes
under `<output_dir>/img_tokens/<session_id>/<split>/` — `extract_sable_latents` intentionally
never combines or deletes `img_tokens` batches (unlike `frame_z`/`dino_z`/`combined_z`), since
this stage reads them directly rather than from a combined trials `.npz`.

**Output**: two files per session, under `<model-root>/img_tokens_compressed/<session_name>/`:

- `img_tokens_pca_joint.npz` — that session's fitted PCA bundle (portable PCA arrays + pickled
  sklearn `PCA` + train-session normalization stats).
- `img_tokens_compressed_trials.npz` — per-split (`train_z_trials_time`, `val_z_trials_time`,
  `test_z_trials_time`) PCA-compressed trial tensors, plus `trial_split`, interval, and
  `neural_trial_idx` metadata.

Feed `img_tokens_compressed_trials.npz`'s parent directory into stage 1's `--latent_input_dir`
(with `--latent_kind img_tokens_compressed`) to run stage 3.

---

## Stage 4: Unprojecting Decoded Tokens

Once stage 3 has decoded PCA-compressed image tokens from spikes, un-PCA and de-normalize them
back to full-dimensional image tokens with `beast.sable_encoding_decoding.img_token.unproject`.

```bash
python -m beast.sable_encoding_decoding.img_token.unproject \
    --eid $EID \
    --decoding-npy $LATENT_ROOT/img_tokens_compressed/$EID/decoding_results_img_tokens_compressed.npy \
    --pca-npz $MODEL_ROOT/img_tokens_compressed/img_tokens_pca_joint.npz \
    --compressed-trials-npz $MODEL_ROOT/img_tokens_compressed/img_tokens_compressed_trials.npz \
    --out-root $MODEL_ROOT/img_tokens_compressed_estimated
```

Key flags (see `img_token/unproject.py::parse_args`):

| Flag | Meaning |
|---|---|
| `--eid` | session id |
| `--decoding-npy` | `decoding_results_img_tokens_compressed.npy` from stage 3 |
| `--pca-npz` | `img_tokens_pca_joint.npz` from stage 2 |
| `--compressed-trials-npz` | `img_tokens_compressed_trials.npz` from stage 2 (for token count `L`, compressed dim `k_comp`, and trial metadata) |
| `--out-root` | output root, e.g. `$MODEL_ROOT/img_tokens_compressed_estimated` |
| `--neural-trial-index` | optional comma-separated neural trial ids (test split only) — restricts output to those trials |

**Output**: per-trial batch `.npz` files under `<out-root>/<eid>/test/`, one file per test
trial, matching inference-style batching so they can be fed straight into stage 5 as
`--z-source`. With `--neural-trial-index`, files are instead named
`img_tokens_estimated_neuraltrialXXXX.npz`.

---

## Stage 5: Decode + Render

`beast.sable_encoding_decoding.render.decode_and_render` loads image tokens (real, from SABLE
inference, or decoded/unprojected from stage 4) and runs them through the SABLE decoder
(`Sable.predict_frame_from_all_tokens`) to reconstruct images, save point clouds, and/or compute
PSNR/SSIM against ground truth.

```bash
python -m beast.sable_encoding_decoding.render.decode_and_render \
    --z-source $MODEL_ROOT/img_tokens_compressed_estimated/$EID/test \
    --out-dir $MODEL_ROOT/render_out \
    --model-dir $MODEL_ROOT \
    --dataset-path $DATASET_PATH
```

`--model-dir` replaces the source pipeline's separate `--config`/`--checkpoint` flags: it
points at a directory containing `config.yaml` and a `*best.ckpt`, loaded via beast's
`beast.api.model.Model.from_dir` convention (the same convention used everywhere else in beast
for loading a trained model).

Key flags (see `render/decode_and_render.py::parse_args`):

| Flag | Meaning |
|---|---|
| `--z-source` | a single `.npz` (`z`/`z_trials`/`*_z_trials_time`), or a directory of direct-child `img_tokens*.npz` files — one decode run per file, sorted by name |
| `--out-dir` | output root for renders / metrics / point clouds |
| `--model-dir` | directory with `config.yaml` + `*best.ckpt`; required unless `--combine-metrics-only` |
| `--camera-npz` | optional `img_tokens_camera_parameters.npz` sidecar, used when `--z-source` files don't carry camera tensors |
| `--dataset-path`, `--correspondence-cache-root`, `--vda-cache-root`, `--eid`, `--include-splits`, `--batch-size`, `--num-workers`, `--ibl-precache-valid-index` | dataloader construction overrides, mirroring `beast/inference.py::infer_sable`'s dataloader setup |
| `--sync-batch-index` | dataloader batch index for the first `--z-source` file (directory mode increments per file) |
| `--neural-trial-index` / `--neural-trial-range` | restrict to specific neural trial ids (mutually exclusive) |
| `--max-render-samples`, `--max-render-views` | caps on how much to render per file |
| `--metrics-only` | skip point clouds/renders, only compute PSNR/SSIM to an `.npz` |
| `--metrics-npz` | output path for `--metrics-only` (default `<out-dir>/psnr_ssim_metrics.npz`) |
| `--combine-metrics-only` | skip model/dataloader entirely; just combine already-computed per-token metric shards under `<out-dir>/metrics_shards/` |
| `--no-resume` | ignore existing metrics shards / render completion markers; redo everything |

**Output layout** under `--out-dir`: rendered image grids and gaussian point clouds per
token file, resumable per-token metric shards under `metrics_shards/`, and (with
`--metrics-only` or after a full run) a combined `psnr_ssim_metrics.npz` with shape `[N, T, 2]`
(PSNR and SSIM per sample/timestep).

---

## Stage 6: Video Generation

Once you have a folder of rendered frames (e.g. from stage 5), stitch them into an MP4 with
`beast.sable_encoding_decoding.video.video_generator`.

```bash
python -m beast.sable_encoding_decoding.video.video_generator \
    --input-dir $MODEL_ROOT/render_out/frames \
    --output $MODEL_ROOT/render_out/video.mp4 \
    --fps 24
```

Key flags:

| Flag | Meaning |
|---|---|
| `-i`, `--input-dir` | folder of frame images, naturally sorted by filename (`frame_2` before `frame_10`) |
| `-o`, `--output` | output `.mp4` path (default: `<input-dir>/video.mp4`) |
| `--fps` | frames per second (default 24) |
| `--recursive` | include images in subdirectories |
| `--extensions` | comma-separated suffixes to include (default `png,jpg,jpeg,webp,tif,tiff`) |
| `--crf` | H.264 CRF quality (default 23; lower is higher quality) |

Uses `imageio`'s FFMPEG plugin when available (from the `sable_encoding_decoding` extra);
falls back automatically to `cv2.VideoWriter` (`opencv-python-headless` is already a base
beast dependency) if the ffmpeg plugin isn't installed.

---

## See Also

- [`CLAUDE_add_sable_encoding_decoding.md`](CLAUDE_add_sable_encoding_decoding.md) — internal
  porting notes: file-by-file source mapping, bug fixes found during the port, and deviations
  from the source repo.
- [`extract_data_for_sable.md`](extract_data_for_sable.md) — preparing the raw IBL dataset that
  feeds SABLE training/inference in the first place.
- [`neural_extraction.md`](neural_extraction.md) — building the `<eid>_aligned.npz` and
  eval-layout frames this pipeline consumes, per source dataset (currently: Cheese3D).
