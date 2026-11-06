# SABLE fixes after 642ebcd

The `qihang-iclr` branch corrects the following behavior from `642ebcd`:

- The geometry encoder now receives `[CLS_v, patches_v]` for each view. Its local
  attention and exported `frame_z` therefore use the correct view grouping. `dino_z`
  still exports the frozen DINO CLS before the geometry encoder; `frame_z` exports
  the updated CLS after it.
- `context_full_mask` hides a view from both the pose encoder and DINO, then zeros
  its patch tokens and DINO CLS. This applies in training and evaluation, including
  when random patch masking is disabled. Original images remain reconstruction
  targets. Random patch masking remains a training-only operation.
- Pseudo-center evaluation preserves the configured real target views instead of
  adding the dummy black center image to the loss. If the only held-out view is
  outside the target set, real target pixels remain supervised.
- Session discovery deduplicates configured session names before splitting frames,
  preventing repeated copies of a frame from entering both train and validation.
  The three Cheese3D configs with a repeated B20 session have also been corrected.
- Masked geometry loss averages over valid xyz coordinates. An all-ones mask now
  produces the same value and gradients as the unmasked mean squared error,
  rather than a loss three times larger.
- Geometry depth, foreground masks, and point-cloud shapes follow the views that
  generated the Gaussians. A single source view skips two-view alignment. For
  cross-camera reconstruction, source patch masks are not applied to target pixels.
  Source depth is exported as `gaussian_depth_output`; `depth_output` is populated
  only when its cameras match the rendered targets.

`model.gaussians.init_gs: false` continues to disable geometry supervision, including
in configurations that set a nonzero `gs_reg_loss_weight`. This now produces a clear
warning and skips loading the unused online VDA model. Enable `init_gs` explicitly
when geometry supervision is intended; Cheese3D requires online VDA in that case,
because its dataset supplies zero placeholders rather than precomputed depth.
When geometry is disabled, loading older checkpoints ignores their unused frozen
`vda.*` weights while retaining strict validation for the remaining model weights.

## Validation

The focused tests use the real small SABLE pose and geometry transformers and
Gaussian decoder, with substitutes for pretrained DINO loading and GPU rendering.
They cover token grouping, held-out image invariance in train/eval, pseudo-center
loss targets, single-view geometry gradients, masked-loss normalization, duplicate
session split isolation, and the existing SABLE utility/sampler checks.

From the repository root, run with the `sable` Python environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -q \
  tests/models/test_sable.py tests/models/test_sable_regressions.py \
  tests/data/test_sable_dataset.py tests/test_train_sable.py
```

Plugin autoload is disabled to avoid unrelated ROS plugins in the local environment.
`--noconftest` avoids the repository-wide fixture download, which these tests do not
need. Full pretrained-model training and GPU rendering still need a cluster run.

## Existing experiments

The parameter shapes are unchanged, but the corrected CLS layout and holdout rules
change predictions. Existing checkpoints were trained with the previous behavior;
retraining is needed to evaluate the corrected training procedure. Re-extract
SABLE-derived features and rerun downstream encoding/decoding for corrected runs.
Do not combine cached features from the old and corrected implementations. The
session deduplication also changes synthetic train/validation membership, so prior
validation scores from duplicated-session configurations are not directly comparable.
