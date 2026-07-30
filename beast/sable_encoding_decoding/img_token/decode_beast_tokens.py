"""Decode saved beast (ViT-MAE) img_tokens back into reconstructed frames.

Unlike Sable's `decode_and_render.py`, this does not need camera extrinsics/intrinsics or a
Gaussian-splat renderer: beast's `VisionTransformer` is a plain per-frame autoencoder, so
decoding a saved token grid is just running it back through the model's own MAE decoder and
unpatchifying the result (mirrors `predict_frame_from_all_tokens` in the original E-RayZer
`erayzer.py`). The decode logic lives here, as a standalone function taking the already-loaded
model as a parameter, rather than as a method on `beast.models.vits.VisionTransformer`.

Two input modes are supported for real (directly-extracted) tokens:

- **Combined npz** (`--img-tokens-npz` / `--ids-restore-npz`): the single-file output of
  `combine_eval_layout_img_tokens` (no `--batch-size`).
- **Shards** (`--input-dir` / `--session-id`): the `img_tokens_batch*.npz` shard tree written by
  `extract_eval_layout_img_token_batches` (`combine-eval-layout-img-tokens --batch-size`), which
  avoids ever materializing the full combined array. Left/right camera tokens are merged into one
  `2*L` axis in each shard (see `beast.inference.extract_eval_layout_img_token_batches`); this
  module un-merges that axis back to `(2, L)` before decoding, since beast's MAE decoder — unlike
  Sable's cross-view renderer — runs per-view, not on a fused multi-camera token set.

PSNR/SSIM metrics (`--target-images-npz`) are only supported in combined-npz mode today, since
they're keyed off reading `neural_trial_idx` / `neural_bin_idx` / `trial_split` straight from that
one file (`beast.sable_encoding_decoding.render.metrics`).

Decoding neurally-estimated tokens (after `unproject.py`'s PCA round trip) additionally needs a
trial-indexed `ids_restore` lookup — since an estimated trial has no `ids_restore` of its own, it
must be fetched from the original extraction's sidecar by `trial_split`/`neural_trial_idx`,
mirroring the original E-RayZer pipeline's `_load_ids_restore_from_sidecar`. That lookup isn't
implemented here yet.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from beast.api.model import Model
from beast.inference import ImagePredictionHandler
from beast.logging import log_step
from beast.models.vits import VisionTransformer
from beast.sable_encoding_decoding.img_token.saved_tokens_io import load_img_tokens_trials_npz
from beast.sable_encoding_decoding.img_token.trials_assembly import (
    assemble_z_trials_time_from_inference_batches,
)
from beast.sable_encoding_decoding.render.metrics import (
    collect_psnr_ssim_metrics_block,
    resolve_metrics_npz_path,
    save_psnr_ssim_metrics_npz,
)


def load_ids_restore_trials_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a combined-format ids_restore sidecar `.npz` and return `(ids_restore, meta)`.

    Mirrors `saved_tokens_io.load_img_tokens_trials_npz`, but reads the
    `train_ids_restore`/`val_ids_restore`/`test_ids_restore` keys written by
    `beast.inference.combine_eval_layout_img_tokens`. These arrays are stored as `float32` (a
    limitation of the shared `_per_split_kw_for_aux` helper they're written with), so this
    function casts them back to `int64` before returning.

    Args:
        path: path to the ids_restore sidecar `.npz`.

    Returns:
        Tuple `(ids_restore, meta)`: `ids_restore` is an `int64` array shaped like the matching
        img_tokens block minus its feature dim (e.g. `[K, T, V, L]`); `meta` has the same shape
        as `saved_tokens_io.load_img_tokens_trials_npz`'s.

    Raises:
        KeyError: if no non-empty `{split}_ids_restore` array is present.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        blocks: list[np.ndarray] = []
        split_labels: list[str] = []
        for split in ('train', 'val', 'test'):
            key = f'{split}_ids_restore'
            if key not in keys:
                continue
            arr = np.asarray(d[key])
            if arr.shape[0] == 0:
                continue
            blocks.append(arr)
            split_labels.extend([split] * int(arr.shape[0]))

        if not blocks:
            raise KeyError(
                f'{path}: no non-empty train/val/test_ids_restore arrays; got {sorted(keys)}',
            )

        ids_restore = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
        ids_restore = ids_restore.astype(np.int64)

        neural_trial_idx = None
        if 'neural_trial_idx' in keys:
            neural_trial_idx = np.asarray(d['neural_trial_idx'], dtype=np.int64)

    meta: dict[str, Any] = {'path': str(path), 'keys': sorted(keys), 'trial_split': split_labels}
    if neural_trial_idx is not None:
        meta['neural_trial_idx'] = neural_trial_idx
    return ids_restore, meta


def load_img_tokens_and_ids_restore_from_shards(
    input_dir: Path,
    session_id: str,
    splits: str = 'train,val,test',
    time_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble `(img_tokens, ids_restore)` from `img_tokens_batch*.npz` shards.

    Reads shards via `assemble_z_trials_time_from_inference_batches` (unchanged), which merges
    them into `z_trials_time` shaped `(N, T, 2*L, D)` plus an `'ids_restore'` aux array shaped
    `(N, T, 2*L)` (recognized via `extra_aux_keys`, since it isn't one of
    `IMG_TOKEN_CAM_BATCH_KEYS`). Both are then un-merged back to per-camera form, `(N, T, 2, L, D)`
    / `(N, T, 2, L)`, undoing the plain concatenation `extract_eval_layout_img_token_batches`
    applied at write time.

    Args:
        input_dir: root directory holding `<session_id>/<split>/img_tokens_batch*.npz` shards —
            i.e. the `img_tokens/` folder itself (same as the `--input-dir` passed to
            `run_pca_and_save.py`, or `<--output-dir passed to combine-eval-layout-img-tokens
            --batch-size>/img_tokens`).
        session_id: session/EID name (shard subdirectory of `input_dir`).
        splits: comma-separated splits to include.
        time_bins: timebins per trial (must match what step0 extracted).

    Returns:
        Tuple `(img_tokens, ids_restore)`: `img_tokens` is `float32` shaped `(N, T, 2, L, D)`;
        `ids_restore` is `int64` shaped `(N, T, 2, L)`.

    Raises:
        RuntimeError: if no `'ids_restore'` aux array is found in the shards.
        ValueError: if the merged token axis isn't evenly divisible by 2 cameras.
    """
    assembly = assemble_z_trials_time_from_inference_batches(
        input_dir=Path(input_dir) / session_id,
        session_id=session_id,
        include_splits=splits,
        time_bins=time_bins,
        file_prefix='img_tokens',
        extra_aux_keys=frozenset({'ids_restore'}),
    )
    if not assembly.aux_trials or 'ids_restore' not in assembly.aux_trials:
        raise RuntimeError(
            f"No 'ids_restore' aux array found in img_tokens shards under "
            f'{Path(input_dir) / session_id}; expected it alongside z in every '
            'img_tokens_batch*.npz (see extract_eval_layout_img_token_batches).',
        )

    z = np.asarray(assembly.z_trials_time, dtype=np.float32)
    restore = np.asarray(assembly.aux_trials['ids_restore'], dtype=np.float32).astype(np.int64)

    n, t, merged, d = z.shape
    if merged % 2 != 0:
        raise ValueError(f'merged token axis {merged} is not evenly divisible by 2 cameras')
    n_tok = merged // 2

    img_tokens = z.reshape(n, t, 2, n_tok, d)
    ids_restore = restore.reshape(n, t, 2, n_tok)
    return img_tokens, ids_restore


def predict_frame_from_all_tokens(
    model: VisionTransformer,
    img_tokens: torch.Tensor,
    ids_restore: torch.Tensor,
    data: dict | None = None,
) -> dict:
    """Decode a saved token grid back into a reconstructed frame.

    Mirrors `predict_frame_from_all_tokens` in the original E-RayZer
    `erayzer_core/model/erayzer.py`, but for beast's plain ViT-MAE autoencoder: no camera
    parameters or Gaussian-splat rendering, just the model's own MAE decoder + unpatchify.

    Args:
        model: a loaded `beast.models.vits.VisionTransformer` (only its `vit_mae` submodule is
            used).
        img_tokens: full encoder output sequence (CLS token plus patches), shape
            `(batch, num_patches + 1, hidden_size)`, as saved by
            `predict_images(..., save_img_tokens=True)`. The CLS token must be included: the
            decoder's self-attention over it affects the patch reconstructions too.
        ids_restore: matching restore indices, shape `(batch, num_patches)`.
        data: optional dict passed through to the result unchanged (e.g. for carrying the
            ground-truth image alongside the reconstruction).

    Returns:
        Dict with `'render'` (reconstructed images, `(batch, channels, height, width)`) and,
        when `data` is given, its contents merged in.
    """
    decoder_outputs = model.vit_mae.decoder(img_tokens, ids_restore)
    logits = decoder_outputs.logits
    render = model.vit_mae.unpatchify(logits)

    result = dict(data) if data is not None else {}
    result['render'] = render
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the beast img-token decode entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back to
            reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True, help='trained beast model directory')
    combined = ap.add_argument_group('combined-npz mode')
    combined.add_argument(
        '--img-tokens-npz',
        type=Path,
        default=None,
        help='img_tokens trials .npz to decode (combine_eval_layout_img_tokens, no --batch-size)',
    )
    combined.add_argument(
        '--ids-restore-npz',
        type=Path,
        default=None,
        help='matching ids_restore sidecar .npz (see combine_eval_layout_img_tokens)',
    )
    sharded = ap.add_argument_group('shard mode')
    sharded.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help=(
            'root of img_tokens_batch*.npz shards (the --output-dir passed to '
            'combine-eval-layout-img-tokens --batch-size)'
        ),
    )
    sharded.add_argument(
        '--session-id', type=str, default=None, help='session/EID name (shard subdirectory)',
    )
    ap.add_argument(
        '--splits',
        type=str,
        default='train,val,test',
        help='shard mode only: comma-separated splits to decode',
    )
    ap.add_argument(
        '--time-bins',
        type=int,
        default=60,
        help='shard mode only: timebins per trial (must match what step0 extracted)',
    )
    ap.add_argument('--out-dir', type=Path, required=True, help='directory for decoded outputs')
    ap.add_argument(
        '--target-images-npz',
        type=Path,
        default=None,
        help=(
            'optional npz with a "image" array of ground-truth frames, for PSNR/SSIM metrics. '
            'Combined-npz mode only.'
        ),
    )
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', type=str, default='cuda:0')
    args = ap.parse_args(argv)

    have_combined = args.img_tokens_npz is not None or args.ids_restore_npz is not None
    have_sharded = args.input_dir is not None or args.session_id is not None
    if have_combined == have_sharded:
        ap.error(
            'pass either --img-tokens-npz + --ids-restore-npz (combined-npz mode) or '
            '--input-dir + --session-id (shard mode), not both/neither',
        )
    if have_combined and (args.img_tokens_npz is None or args.ids_restore_npz is None):
        ap.error('combined-npz mode requires both --img-tokens-npz and --ids-restore-npz')
    if have_sharded and (args.input_dir is None or args.session_id is None):
        ap.error('shard mode requires both --input-dir and --session-id')
    if have_sharded and args.target_images_npz is not None:
        ap.error('--target-images-npz (PSNR/SSIM metrics) is not supported in shard mode yet')
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the beast img-token decode pipeline end to end (CLI entry point)."""
    args = parse_args(argv)

    if args.input_dir is not None:
        log_step(
            f'Assembling img_tokens + ids_restore from shards: {args.input_dir}/{args.session_id}',
            level='info',
        )
        img_tokens, ids_restore = load_img_tokens_and_ids_restore_from_shards(
            args.input_dir, args.session_id, splits=args.splits, time_bins=args.time_bins,
        )
    else:
        log_step(f'Loading img_tokens from: {args.img_tokens_npz}', level='info')
        img_tokens, _tokens_meta = load_img_tokens_trials_npz(args.img_tokens_npz)
        log_step(f'Loading ids_restore from: {args.ids_restore_npz}', level='info')
        ids_restore, _ = load_ids_restore_trials_npz(args.ids_restore_npz)
    if img_tokens.shape[:-1] != ids_restore.shape:
        raise ValueError(
            f'img_tokens {img_tokens.shape} and ids_restore {ids_restore.shape} shape mismatch',
        )

    log_step(f'Loading model from: {args.model_dir}', level='info')
    loaded = Model.from_dir(args.model_dir)
    model = loaded.model
    model.to(args.device)
    model.eval()

    handler = ImagePredictionHandler(args.out_dir, args.out_dir)

    # collapse (K, T, V) into a single decode batch dim; L, D stay per-token
    k, t, v, l, d = img_tokens.shape
    flat_tokens = img_tokens.reshape(k * t * v, l, d)
    flat_restore = ids_restore.reshape(k * t * v, l)

    target = None
    if args.target_images_npz is not None:
        with np.load(args.target_images_npz, allow_pickle=True) as td:
            image = np.asarray(td['image'], dtype=np.float32)
            target = image.reshape(k * t * v, *image.shape[-3:])

    psnr_blocks, ssim_blocks, trial_blocks, bin_blocks, split_blocks = [], [], [], [], []
    num_decoded = 0
    with torch.no_grad():
        for start in range(0, flat_tokens.shape[0], args.batch_size):
            end = min(start + args.batch_size, flat_tokens.shape[0])
            tokens_batch = torch.from_numpy(flat_tokens[start:end]).to(args.device)
            restore_batch = torch.from_numpy(flat_restore[start:end]).to(args.device)

            result = predict_frame_from_all_tokens(model, tokens_batch, restore_batch)
            render = result['render']

            for i in range(render.shape[0]):
                row = start + i
                handler.save_reconstruction(render[i], 'decoded', row, Path(f'row{row:06d}.png'))
            num_decoded += render.shape[0]

            if target is not None:
                target_batch = torch.from_numpy(target[start:end]).to(args.device)
                psnr, ssim, trial_idx, bin_idx, split_labels, _ = collect_psnr_ssim_metrics_block(
                    render.unsqueeze(1),
                    target_batch.unsqueeze(1),
                    args.img_tokens_npz,
                    k_trials=end - start,
                    t_bins=1,
                )
                psnr_blocks.append(psnr)
                ssim_blocks.append(ssim)
                trial_blocks.append(trial_idx)
                bin_blocks.append(bin_idx)
                split_blocks.append(split_labels)

    log_step(f'Decoded {num_decoded} frames to: {args.out_dir}', level='info')

    if psnr_blocks:
        metrics_path = resolve_metrics_npz_path(None, args.out_dir)
        save_psnr_ssim_metrics_npz(
            metrics_path,
            psnr_blocks=psnr_blocks,
            ssim_blocks=ssim_blocks,
            neural_trial_blocks=trial_blocks,
            neural_bin_blocks=bin_blocks,
            trial_split_blocks=split_blocks,
            source_file_rows=[str(args.img_tokens_npz)] * sum(b.shape[0] for b in trial_blocks),
        )
        log_step(f'Saved PSNR/SSIM metrics to: {metrics_path}', level='info')


if __name__ == '__main__':
    main()
