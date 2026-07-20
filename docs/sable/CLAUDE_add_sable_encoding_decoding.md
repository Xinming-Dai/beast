# Adding E-RayZer's IBL Neural Encoding/Decoding Pipeline to Beast

Source repo: `/u/xdai3/project3d/erayzer_cls/E-RayZer-private`
Target repo: `/u/xdai3/project3d/SBALE_repo/beast`

This port was done in four phases and lands entirely under
`beast/sable_encoding_decoding/`, a self-contained optional analysis subpackage (see
"Self-Containment" below).

## Files to Move

All paths in the "Source" column are relative to `src/` in the source repo. All paths in
"Destination" are relative to `beast/` in the target repo.

### Phase 1 — `neural/` (RRR + CNN/TCN encoding & decoding)

| Source | Destination |
|---|---|
| `analyses/utils/utils.py` | `sable_encoding_decoding/neural/utils.py` |
| `analyses/models/rrr_encoder.py` | `sable_encoding_decoding/neural/rrr_encoder.py` |
| `analyses/models/rrr_decoder.py` | `sable_encoding_decoding/neural/rrr_decoder.py` |
| `analyses/utils/encoder.py` | `sable_encoding_decoding/neural/encoder.py` |
| `analyses/utils/decoder.py` | `sable_encoding_decoding/neural/decoder.py` |
| `test.py` | `sable_encoding_decoding/neural/run_encoding_decoding.py` |
| — (new file, no source) | `sable_encoding_decoding/neural/_rrr_common.py` |

### Phase 2 — `img_token/` (PCA compression, trial assembly, unprojection)

| Source | Destination |
|---|---|
| `eval/combine_depth_fused_z_batches.py` (partial — see Deviations) | `sable_encoding_decoding/img_token/trials_assembly.py` |
| `analyses/img_decoder/data_compression.py` | `sable_encoding_decoding/img_token/pca_compression.py` |
| `analyses/img_decoder/saved_img_tokens_io.py` | `sable_encoding_decoding/img_token/saved_tokens_io.py` |
| `analyses/img_decoder/run_img_tokens_pca_and_save.py` | `sable_encoding_decoding/img_token/run_pca_and_save.py` |
| `analyses/img_decoder/neural_decoder_for_img.py` | `sable_encoding_decoding/img_token/neural_decoder.py` |
| `analyses/img_decoder/unproject_pca_compressed_img_tokens.py` | `sable_encoding_decoding/img_token/unproject.py` |

### Phase 3 — `render/` (decode + render + metrics)

| Source | Destination |
|---|---|
| `analyses/img_decoder/decode_saved_img_tokens_render.py` | `sable_encoding_decoding/render/decode_and_render.py` |
| `analyses/img_decoder/decode_saved_img_tokens_utils.py` | `sable_encoding_decoding/render/decode_utils.py` |
| `analyses/img_decoder/decoding_metrics.py` | `sable_encoding_decoding/render/metrics.py` |

### Phase 4 — `video/` (frame sequence to MP4)

| Source | Destination |
|---|---|
| `analyses/img_decoder/video_generator.py` | `sable_encoding_decoding/video/video_generator.py` |

## `__init__.py` Files Created

- `beast/sable_encoding_decoding/__init__.py`
- `beast/sable_encoding_decoding/neural/__init__.py`
- `beast/sable_encoding_decoding/img_token/__init__.py`
- `beast/sable_encoding_decoding/render/__init__.py`
- `beast/sable_encoding_decoding/video/__init__.py`

Each has a one-line module docstring describing the subpackage's scope; no other content.

## Import Rewriting Required

The source files in `src/analyses/`, `src/eval/`, and `src/test.py` are largely
self-contained (stdlib + numpy/torch/ray/sklearn), unlike the model port
(`CLAUDE_add_erayzer.md`) which had heavy `erayzer_core.*` cross-imports. The rewrites
needed here are narrower:

| Old prefix / pattern | New |
|---|---|
| `analyses.models.rrr_encoder` / `analyses.models.rrr_decoder` | `beast.sable_encoding_decoding.neural.rrr_encoder` / `.rrr_decoder` |
| `analyses.utils.utils` | `beast.sable_encoding_decoding.neural.utils` |
| `analyses.utils.encoder` / `analyses.utils.decoder` | `beast.sable_encoding_decoding.neural.encoder` / `.decoder` |
| `analyses.img_decoder.saved_img_tokens_io` | `beast.sable_encoding_decoding.img_token.saved_tokens_io` |
| `analyses.img_decoder.data_compression` | `beast.sable_encoding_decoding.img_token.pca_compression` |
| `analyses.img_decoder.neural_decoder_for_img` | `beast.sable_encoding_decoding.img_token.neural_decoder` |
| `analyses.img_decoder.decoding_metrics` | `beast.sable_encoding_decoding.render.metrics` |
| `analyses.img_decoder.decode_saved_img_tokens_utils` | `beast.sable_encoding_decoding.render.decode_utils` |
| `beast.models.sable.Sable` (already existed — see Deviations) | unchanged, imported directly |
| `beast.api.model.Model` / `beast.inference` (new dependency for `render/decode_and_render.py`) | used for checkpoint loading and dataloader construction, mirroring `beast/inference.py::infer_sable` |

## CLAUDE.md Conventions Applied

Every moved file was rewritten, not blind-copied, to match `CLAUDE.md`:

- **Docstrings**: every module got a module-level docstring; every function/class got a
  Google-style docstring with `Args:`/`Returns:`/`Raises:` sections. The source repo's files
  were inconsistently documented (some had none, some had one-line comments); `neural/utils.py`,
  `img_token/trials_assembly.py` (originally the 2046-line
  `combine_depth_fused_z_batches.py`), and `render/decode_and_render.py` needed the most work
  since their source versions had almost no docstrings on internal helpers.
- **Type hints**: full modern hints (`X | Y`, `list[X]`, `dict[K, V]`) added throughout;
  no `typing.Optional`/`List`/`Dict`/`Union` imports remain.
- **Quotes**: all double-quoted strings converted to single-quoted (source repo used double
  quotes throughout).
- **Imports**: converted to absolute `beast.sable_encoding_decoding.*` form; sorted and grouped
  (stdlib, third-party, local).
- **`pathlib.Path`**: all `os.path.join`/`os.makedirs`/`os.path.exists` call sites converted to
  `Path` equivalents. `img_token/trials_assembly.py` and `render/decode_and_render.py` had the
  most path-string manipulation to convert.
- **f-strings**: all `.format()` and `%`-style string interpolation converted to f-strings,
  including log/print/error messages.
- **Trailing commas**: added to all multi-line call/def argument lists.
- **Line length / whitespace**: wrapped to 99 columns; no trailing whitespace; every file ends
  with exactly one newline.

## Deviations from Source

### `neural/_rrr_common.py` (new file)

`np2tensor`, `np2param`, `tensor2np`, and `get_device` were each duplicated verbatim in both
`rrr_encoder.py` and `rrr_decoder.py` in the source repo. Factored out into a shared
`neural/_rrr_common.py` module; both files now import from it instead of duplicating the
functions.

### Bug fix: `neural/utils.py::neg_log_likelihood`

The source version referenced an undefined `logger` (no `logger` import existed anywhere in
that file) inside a rarely-hit branch (zero-rate prediction). This would have raised
`NameError` the first time it executed. Replaced with `warnings.warn`.

### Bug fix: `neural/rrr_decoder.py::RRRGD.state_dict()`

Referenced an undefined `self.N`. The class only ever sets `self.ncoef`; fixed to reference
`self.ncoef`.

### Bug fix: `neural/decoder.py::train_rrr_decoder`

A print statement referenced an undefined `threshold` variable. Fixed the message to report
the actual threshold value used in that code path.

### `img_token/trials_assembly.py` scope reduction

Ported from the ~2046-line `eval/combine_depth_fused_z_batches.py`. Only the confirmed-unused
CLI tail was dropped: `parse_args`, `main`, `run_combine_depth_fused_z`, and
`_run_combine_per_split_npzs`. Verified via `grep` across the source tree that nothing else
calls into that tail — the assembly *functions* (`assemble_z_trials_time_from_inference_batches`
and friends) are what downstream code (`run_pca_and_save.py`) actually imports and uses.

### `render/decode_and_render.py` scaffolding, not decoder-logic, substitution

`Sable.predict_frame_from_all_tokens` **already existed** in `beast/models/sable.py`, carried
over verbatim from the original erayzer-to-sable port (see `CLAUDE_add_erayzer.md` and later
sable-specific work) — no decoder-logic substitution was needed here. What *was* substituted:

- Checkpoint/config loading now goes through `beast.api.model.Model.from_dir` (directory with
  `config.yaml` + a `*best.ckpt`) instead of the source's separate `--config`/`--checkpoint`
  flag pair.
- Dataloader construction mirrors `beast/inference.py::infer_sable`.
- The source's single `save_and_vis_gaussian_pointclouds` helper was split into two beast-native
  calls: `beast.inference.save_gaussian_pointclouds` (expects a `dict` — built via
  `vars(result)` since `predict_frame_from_all_tokens` returns a `types.SimpleNamespace`) and
  `beast.models.model_utils.train_vis.save_training_visuals` (expects the `SimpleNamespace`
  directly, since it uses attribute access).

### `render/decode_and_render.py::_apply_dataloader_overrides`

Reimplements the intent of `img_token/saved_tokens_io.py::apply_dataloader_overrides` but using
dict-indexing instead of attribute access, since beast configs are plain nested `dict`s, not
attribute-style objects (unlike the source repo's config objects).

### No `--finetune-ckpt-out`/`--finetune-lr` flags

Earlier planning assumed the source `decode_saved_img_tokens_render.py` might have finetuning
flags to port. Confirmed via `grep` that no such flags exist in the actual source file —
nothing was ported there.

## Self-Containment

No core `beast/` module (`beast.models`, `beast.data`, `beast.api`, `beast.cli`) imports
anything from `beast.sable_encoding_decoding`. The dependency runs one direction only:
`sable_encoding_decoding.render.decode_and_render` imports `beast.api.model.Model`,
`beast.inference`, and `beast.models.sable.Sable`. Base `beast` installs (without the
`sable_encoding_decoding` extra) are entirely unaffected by this subpackage's presence.

## New 3rd-Party Dependencies

Added to `pyproject.toml` as the `sable_encoding_decoding` optional extra (parallel to the
existing `vda` extra — see `[project.optional-dependencies]`):

| Package | Used by | Notes |
|---|---|---|
| `ray[tune]` | `neural/encoder.py`, `neural/decoder.py`, `neural/run_encoding_decoding.py`, `img_token/neural_decoder.py` | Ray Tune orchestrates the `_with_tune` hyperparameter search variants |
| `facemap` | `neural/utils.py` | behavioral feature extraction utility referenced by the source repo |
| `torcheval` | `neural/rrr_encoder.py`, `neural/rrr_decoder.py` | metrics (e.g. R2) during RRR gradient descent |
| `accelerate` | `neural/encoder.py`, `neural/decoder.py` | device placement helpers used by the CNN/TCN training loops |
| `torchmetrics[image]` | `render/metrics.py` | PSNR/SSIM evaluation |
| `imageio[ffmpeg]` | `video/video_generator.py` | primary video-encoding backend; `cv2.VideoWriter` is the fallback when the ffmpeg plugin is unavailable |

Install with:

```bash
pip install -e ".[sable_encoding_decoding]"
```

Base beast installs are unaffected — matching how the `vda` extra is documented elsewhere in
this repo.
