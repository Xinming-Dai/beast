"""Decode saved beast (ViT-MAE) img_tokens back into reconstructed frames.

Unlike Sable's `decode_and_render.py`, this does not need camera extrinsics/intrinsics or a
Gaussian-splat renderer: beast's `VisionTransformer` is a plain per-frame autoencoder, so
decoding a saved token grid is just running it back through the model's own MAE decoder and
unpatchifying the result (mirrors `predict_frame_from_all_tokens` in the original E-RayZer
`erayzer.py`). The decode logic lives here, as a standalone function taking the already-loaded
model as a parameter, rather than as a method on `beast.models.vits.VisionTransformer`.

Three input modes are supported:

- **Combined npz** (`--img-tokens-npz` / `--ids-restore-npz`): the single-file output of
  `combine_eval_layout_img_tokens` (no `--batch-size`).
- **Shards** (`--input-dir` / `--session-id`): the `img_tokens_batch*.npz` shard tree written by
  `extract_eval_layout_img_token_batches` (`combine-eval-layout-img-tokens --batch-size`), which
  avoids ever materializing the full combined array.
- **Estimated** (`--estimated-dir` / `--ids-restore-sidecar`): step3 `unproject.py`'s per-trial
  `img_tokens_estimated_neuraltrial*.npz` output (neurally decoded, PCA-unprojected tokens).
  These files carry no `ids_restore` of their own — decoding needs the MAE decoder's un-shuffle
  indices, which are only ever produced at the original extraction — so this mode fetches each
  estimated trial's `ids_restore` from the `img_tokens_camera_parameters.npz` sidecar written by
  step1 `run_pca_and_save.py` (a byproduct of that step's own CPU-only per-split assembly pass),
  matching `(trial_split, neural_trial_idx)`, mirroring the original E-RayZer pipeline's
  `_load_ids_restore_from_sidecar` — and Sable's own camera-sidecar consumer,
  `decode_and_render.py`'s `_load_cameras_from_sidecar`.

In shard and estimated mode, left/right camera tokens are merged into one `2*L` axis on disk (see
`beast.inference.extract_eval_layout_img_token_batches`); this module un-merges that axis back to
`(2, L)` before decoding, since beast's MAE decoder — unlike Sable's cross-view renderer — runs
per-view, not on a fused multi-camera token set. `ids_restore`'s own per-view `L` is one token
shorter than `img_tokens`'s (it excludes the CLS token, see `_unmerge_camera_axis`), so the two
are un-merged independently rather than assumed to share a token count.

PSNR/SSIM metrics are supported in combined-npz mode via a precomputed `--target-images-npz`, and
in estimated mode via `--target-frame-mapping-left` / `--target-frame-mapping-right` (raw frames
resolved per trial/bin through the eval-layout camera input dirs' `frame_index_mapping.json`, see
`beast.sable_encoding_decoding.img_token.target_frames`). Not supported in shard mode.
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
from beast.sable_encoding_decoding.img_token.saved_tokens_io import (
    load_img_tokens_trials_npz,
    sorted_img_tokens_npz_paths,
)
from beast.sable_encoding_decoding.img_token.target_frames import (
    load_frame_index_mapping,
    load_source_frame_index_mapping,
    load_target_images_for_trials,
    load_target_masks_for_trials,
)
from beast.sable_encoding_decoding.img_token.trials_assembly import (
    assemble_z_trials_time_from_inference_batches,
)
from beast.sable_encoding_decoding.render.decode_utils import _print_combined_metrics_summary
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


def _unmerge_camera_axis(
    img_tokens: np.ndarray, ids_restore: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a shard-layout merged `2*L` camera axis back into a leading `(2, L)` view axis.

    `img_tokens` and `ids_restore` have independent per-camera token counts (`ids_restore` is
    one token shorter — it un-shuffles patches only, while `img_tokens` also carries the CLS
    token, see `predict_frame_from_all_tokens`), so each array's merged axis is un-merged using
    its own trailing dimension rather than a shared token count.

    Args:
        img_tokens: shape `(N, T, 2*L_img, D)`.
        ids_restore: shape `(N, T, 2*L_restore)` (`L_restore == L_img - 1`).

    Returns:
        Tuple `(img_tokens, ids_restore)` reshaped to `(N, T, 2, L_img, D)` /
        `(N, T, 2, L_restore)`.

    Raises:
        ValueError: if either merged token axis isn't evenly divisible by 2 cameras.
    """
    n, t, merged_img, d = img_tokens.shape
    _, _, merged_restore = ids_restore.shape
    if merged_img % 2 != 0:
        raise ValueError(f'img_tokens merged axis {merged_img} is not evenly divisible by 2')
    if merged_restore % 2 != 0:
        raise ValueError(f'ids_restore merged axis {merged_restore} is not evenly divisible by 2')
    n_tok_img = merged_img // 2
    n_tok_restore = merged_restore // 2
    return (
        img_tokens.reshape(n, t, 2, n_tok_img, d),
        ids_restore.reshape(n, t, 2, n_tok_restore),
    )


def load_ids_restore_lookup_from_sidecar(
    sidecar_path: Path,
) -> dict[tuple[str, int], np.ndarray]:
    """Build a `(trial_split, neural_trial_idx) -> ids_restore` lookup from a PCA-step sidecar.

    Used to decode step3-estimated tokens, which carry no `ids_restore` of their own: each
    estimated trial's restore indices must be fetched from the original extraction by matching
    trial identity. The sidecar (`img_tokens_camera_parameters.npz`, written by
    `run_pca_and_save.py`'s `_write_camera_sidecar`) already carries `{split}_ids_restore`
    alongside `trial_split` / `neural_trial_idx` as a byproduct of that step's own CPU-only,
    per-split assembly pass — so loading it directly here avoids re-running that assembly (and
    materializing the full high-dimensional `z` token array) a second time on the GPU decode job.

    Args:
        sidecar_path: path to the `img_tokens_camera_parameters.npz` sidecar.

    Returns:
        `{(trial_split, neural_trial_idx): ids_restore}`, where each value has shape
        `(T, 2*L_restore)` (merged camera axis; `L_restore == L - 1`, one token shorter than
        step3's `z` layout since `ids_restore` excludes the CLS token).

    Raises:
        KeyError: if the sidecar lacks `trial_split` / `neural_trial_idx`, or no
            `{split}_ids_restore` array is present.
        ValueError: if two trials share the same `(split, neural_trial_idx)` key.
    """
    path = Path(sidecar_path).resolve()
    with np.load(path, allow_pickle=True) as d:
        if 'trial_split' not in d.files or 'neural_trial_idx' not in d.files:
            raise KeyError(
                f'{path}: sidecar needs trial_split and neural_trial_idx; got {sorted(d.files)}',
            )
        trial_split_labels = [
            str(x).lower() for x in np.asarray(d['trial_split'], dtype=object).reshape(-1)
        ]
        neural_trial_idx = np.asarray(d['neural_trial_idx'], dtype=np.int64).reshape(-1)

        restore_by_split: dict[str, np.ndarray] = {}
        for split in ('train', 'val', 'test'):
            key = f'{split}_ids_restore'
            if key in d.files:
                restore_by_split[split] = np.asarray(d[key], dtype=np.float32).astype(np.int64)
        if not restore_by_split:
            raise KeyError(f"{path}: no '{{split}}_ids_restore' array found; got {sorted(d.files)}")

        split_row_counter: dict[str, int] = {}
        lookup: dict[tuple[str, int], np.ndarray] = {}
        for split, tid in zip(trial_split_labels, neural_trial_idx, strict=True):
            row = split_row_counter.get(split, 0)
            split_row_counter[split] = row + 1
            if split not in restore_by_split:
                continue
            key = (split, int(tid))
            if key in lookup:
                raise ValueError(f'Duplicate trial {key} in sidecar {path}')
            lookup[key] = restore_by_split[split][row]
    return lookup


def load_estimated_tokens_dir(
    estimated_dir: Path,
) -> tuple[np.ndarray, list[str], np.ndarray, list[Path]]:
    """Load step3's per-trial `img_tokens_estimated_neuraltrial*.npz` files from a directory.

    Args:
        estimated_dir: step3 `unproject.py` output directory (a single split dir, or a root
            containing several split subdirectories — searched recursively).

    Returns:
        Tuple `(img_tokens, trial_split_labels, neural_trial_idx, source_paths)`:
        `img_tokens` is `float32` shaped `(K, T, L, D)` (merged camera axis, one row per file,
        sorted by path); `trial_split_labels` has length `K`; `neural_trial_idx` is `int64`
        shaped `(K,)`; `source_paths` has length `K`.

    Raises:
        FileNotFoundError: if no matching `.npz` files are found under `estimated_dir`.
    """
    paths = sorted_img_tokens_npz_paths(Path(estimated_dir))
    if not paths:
        raise FileNotFoundError(f'No img_tokens_estimated*.npz found under {estimated_dir}')

    z_rows, split_labels, trial_ids = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as d:
            z = np.asarray(d['z'], dtype=np.float32)
            if z.ndim == 4 and z.shape[0] == 1:
                z = z[0]
            elif z.ndim != 3:
                raise ValueError(
                    f'{path}: expected z shape (1, T, L, D) or (T, L, D); got {z.shape}',
                )
            split_labels.append(str(np.asarray(d['trial_split']).reshape(-1)[0]).lower())
            trial_ids.append(int(np.asarray(d['neural_trial_idx']).reshape(-1)[0]))
        z_rows.append(z)

    img_tokens = np.stack(z_rows, axis=0)
    return img_tokens, split_labels, np.asarray(trial_ids, dtype=np.int64), paths


def resolve_ids_restore_for_trials(
    trial_split_labels: list[str],
    neural_trial_idx: np.ndarray,
    lookup: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    """Fetch each trial's `ids_restore` from a lookup by `(split, neural_trial_idx)`.

    Args:
        trial_split_labels: per-trial split label, length `K`.
        neural_trial_idx: per-trial neural trial id, shape `(K,)`.
        lookup: `load_ids_restore_lookup_from_sidecar`'s output.

    Returns:
        `ids_restore` stacked to shape `(K, T, 2*L_restore)`, in the same order as the inputs.

    Raises:
        KeyError: if any `(split, neural_trial_idx)` pair has no match in `lookup`.
    """
    missing: list[tuple[str, int]] = []
    rows = []
    for split, tid in zip(trial_split_labels, neural_trial_idx, strict=True):
        key = (str(split).lower(), int(tid))
        if key not in lookup:
            missing.append(key)
            continue
        rows.append(lookup[key])
    if missing:
        raise KeyError(
            f'{len(missing)} trial(s) from the estimated-token directory have no matching '
            f'ids_restore in the sidecar: {missing[:10]}'
            + (' ...' if len(missing) > 10 else ''),
        )
    return np.stack(rows, axis=0)


def load_img_tokens_and_ids_restore_from_shards(
    input_dir: Path,
    session_id: str,
    splits: str = 'train,val,test',
    time_bins: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble `(img_tokens, ids_restore)` from `img_tokens_batch*.npz` shards.

    Reads shards via `assemble_z_trials_time_from_inference_batches` (unchanged), which merges
    them into `z_trials_time` shaped `(N, T, 2*L, D)` plus an `'ids_restore'` aux array shaped
    `(N, T, 2*L_restore)` (`L_restore == L - 1`, one token shorter — `ids_restore` excludes the
    CLS token; recognized via `extra_aux_keys`, since it isn't one of `IMG_TOKEN_CAM_BATCH_KEYS`).
    Both are then un-merged back to per-camera form, `(N, T, 2, L, D)` / `(N, T, 2, L_restore)`,
    undoing the plain concatenation `extract_eval_layout_img_token_batches` applied at write time.

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
        `ids_restore` is `int64` shaped `(N, T, 2, L_restore)` (`L_restore == L - 1`).

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
    return _unmerge_camera_axis(z, restore)


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
    estimated = ap.add_argument_group('estimated mode')
    estimated.add_argument(
        '--estimated-dir',
        type=Path,
        default=None,
        help=(
            'step3 unproject.py output dir (a split dir, or a root of several split dirs) of '
            'per-trial img_tokens_estimated_neuraltrial*.npz files'
        ),
    )
    estimated.add_argument(
        '--ids-restore-sidecar',
        type=Path,
        default=None,
        help=(
            'estimated mode: img_tokens_camera_parameters.npz sidecar written by step1 '
            "run_pca_and_save.py, carrying each trial's ids_restore"
        ),
    )
    estimated.add_argument(
        '--target-frame-mapping-left',
        type=Path,
        default=None,
        help=(
            'estimated mode: left camera eval-layout input dir (the one passed as `beast '
            'predict --input`) for PSNR/SSIM ground-truth frames; requires '
            '--target-frame-mapping-right too'
        ),
    )
    estimated.add_argument(
        '--target-frame-mapping-right',
        type=Path,
        default=None,
        help=(
            'estimated mode: right camera eval-layout input dir, paired with '
            '--target-frame-mapping-left'
        ),
    )
    estimated.add_argument(
        '--use-segmentation-mask',
        action='store_true',
        help=(
            'estimated mode: zero out background pixels (per precomputed SAM3 masks) in both '
            'render and target before saving/metrics; requires --segmentation-root, --eid, and '
            '--target-frame-mapping-left/-right'
        ),
    )
    estimated.add_argument(
        '--segmentation-root',
        type=Path,
        default=None,
        help=(
            'estimated mode: root directory precomputed segmentation masks were written under '
            '(see beast.preprocess.sable.precompute_sam3_masks_eval)'
        ),
    )
    estimated.add_argument(
        '--eid', type=str, default=None, help='estimated mode: session id (mask subdirectory)',
    )
    ap.add_argument(
        '--splits',
        type=str,
        default='train,val,test',
        help='shard/estimated mode only: comma-separated splits to decode',
    )
    ap.add_argument(
        '--time-bins',
        type=int,
        default=60,
        help='shard/estimated mode only: timebins per trial (must match what step0 extracted)',
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
    have_estimated = args.estimated_dir is not None or args.ids_restore_sidecar is not None
    if sum([have_combined, have_sharded, have_estimated]) != 1:
        ap.error(
            'pass exactly one of: --img-tokens-npz + --ids-restore-npz (combined-npz mode), '
            '--input-dir + --session-id (shard mode), or --estimated-dir + '
            '--ids-restore-sidecar (estimated mode)',
        )
    if have_combined and (args.img_tokens_npz is None or args.ids_restore_npz is None):
        ap.error('combined-npz mode requires both --img-tokens-npz and --ids-restore-npz')
    if have_sharded and (args.input_dir is None or args.session_id is None):
        ap.error('shard mode requires both --input-dir and --session-id')
    if have_sharded and args.target_images_npz is not None:
        ap.error('--target-images-npz (PSNR/SSIM metrics) is not supported in shard mode yet')
    if have_estimated and (args.estimated_dir is None or args.ids_restore_sidecar is None):
        ap.error('estimated mode requires --estimated-dir and --ids-restore-sidecar')
    have_target_frames = (
        args.target_frame_mapping_left is not None or args.target_frame_mapping_right is not None
    )
    if have_target_frames and (
        args.target_frame_mapping_left is None or args.target_frame_mapping_right is None
    ):
        ap.error(
            '--target-frame-mapping-left and --target-frame-mapping-right must be given together',
        )
    if have_target_frames and not have_estimated:
        ap.error('--target-frame-mapping-left/-right are estimated-mode only')
    if args.use_segmentation_mask and (
        args.segmentation_root is None or args.eid is None or not have_target_frames
    ):
        ap.error(
            '--use-segmentation-mask requires --segmentation-root, --eid, and '
            '--target-frame-mapping-left/-right',
        )
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the beast img-token decode pipeline end to end (CLI entry point)."""
    args = parse_args(argv)

    trial_split_labels: list[str] | None = None
    neural_trial_idx: np.ndarray | None = None
    if args.estimated_dir is not None:
        log_step(f'Loading estimated img_tokens from: {args.estimated_dir}', level='info')
        img_tokens, trial_split_labels, neural_trial_idx, _paths = load_estimated_tokens_dir(
            args.estimated_dir,
        )
        log_step(
            f'Loading ids_restore lookup from sidecar: {args.ids_restore_sidecar}', level='info',
        )
        restore_lookup = load_ids_restore_lookup_from_sidecar(args.ids_restore_sidecar)
        ids_restore = resolve_ids_restore_for_trials(
            trial_split_labels, neural_trial_idx, restore_lookup,
        )
        img_tokens, ids_restore = _unmerge_camera_axis(img_tokens, ids_restore)
    elif args.input_dir is not None:
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
    if (
        img_tokens.shape[:-2] != ids_restore.shape[:-1]
        or img_tokens.shape[-2] != ids_restore.shape[-1] + 1
    ):
        raise ValueError(
            f'img_tokens {img_tokens.shape} and ids_restore {ids_restore.shape} shape mismatch '
            '(expected matching leading dims and img_tokens token axis == ids_restore token '
            'axis + 1, for the CLS token)',
        )

    log_step(f'Loading model from: {args.model_dir}', level='info')
    loaded = Model.from_dir(args.model_dir)
    model = loaded.model
    model.to(args.device)
    model.eval()

    handler = ImagePredictionHandler(args.out_dir, args.out_dir)

    # collapse (K, T, V) into a single decode batch dim; L, D stay per-token. ids_restore's
    # token axis is one shorter than img_tokens's (no CLS restore index), so flatten each with
    # its own trailing dim.
    k, t, v, l, d = img_tokens.shape
    l_restore = ids_restore.shape[-1]
    flat_tokens = img_tokens.reshape(k * t * v, l, d)
    flat_restore = ids_restore.reshape(k * t * v, l_restore)

    target = None
    target_masks = None
    trial_idx_flat = bin_idx_flat = split_flat = None
    metrics_source = args.img_tokens_npz
    if args.target_images_npz is not None:
        with np.load(args.target_images_npz, allow_pickle=True) as td:
            image = np.asarray(td['image'], dtype=np.float32)
            target = image.reshape(k * t * v, *image.shape[-3:])
    elif args.target_frame_mapping_left is not None:
        assert trial_split_labels is not None and neural_trial_idx is not None
        image_size = int(loaded.config['model']['model_params']['image_size'])
        unique_splits = sorted(set(trial_split_labels))
        mapping_left = {
            sp: load_frame_index_mapping(args.target_frame_mapping_left, sp)
            for sp in unique_splits
        }
        mapping_right = {
            sp: load_frame_index_mapping(args.target_frame_mapping_right, sp)
            for sp in unique_splits
        }
        log_step('Loading ground-truth target frames for PSNR/SSIM metrics', level='info')
        target_full = load_target_images_for_trials(
            trial_split_labels, neural_trial_idx, t, mapping_left, mapping_right, image_size,
        ).numpy()
        target = target_full.reshape(k * t * v, *target_full.shape[-3:])

        trial_idx_full = np.broadcast_to(np.asarray(neural_trial_idx)[:, None, None], (k, t, v))
        bin_idx_full = np.broadcast_to(np.arange(t)[None, :, None], (k, t, v))
        split_full = np.broadcast_to(
            np.asarray(trial_split_labels, dtype=object)[:, None, None], (k, t, v),
        )
        trial_idx_flat = trial_idx_full.reshape(k * t * v).astype(np.int64)
        bin_idx_flat = bin_idx_full.reshape(k * t * v).astype(np.int64)
        split_flat = split_full.reshape(k * t * v).astype(str)
        metrics_source = args.estimated_dir

        if args.use_segmentation_mask:
            mask_index_left = {
                sp: load_source_frame_index_mapping(args.target_frame_mapping_left, sp, 'left')
                for sp in unique_splits
            }
            mask_index_right = {
                sp: load_source_frame_index_mapping(args.target_frame_mapping_right, sp, 'right')
                for sp in unique_splits
            }
            log_step('Loading segmentation masks', level='info')
            masks_full = load_target_masks_for_trials(
                trial_split_labels,
                neural_trial_idx,
                t,
                mask_index_left,
                mask_index_right,
                args.segmentation_root,
                args.eid,
                image_size,
            ).numpy()
            target_masks = masks_full.reshape(k * t * v, *masks_full.shape[-3:])

    psnr_blocks, ssim_blocks, trial_blocks, bin_blocks, split_blocks = [], [], [], [], []
    num_decoded = 0
    with torch.no_grad():
        for start in range(0, flat_tokens.shape[0], args.batch_size):
            end = min(start + args.batch_size, flat_tokens.shape[0])
            tokens_batch = torch.from_numpy(flat_tokens[start:end]).to(args.device)
            restore_batch = torch.from_numpy(flat_restore[start:end]).to(args.device)

            result = predict_frame_from_all_tokens(model, tokens_batch, restore_batch)
            render = result['render']

            if target_masks is not None:
                mask_batch = torch.from_numpy(target_masks[start:end]).to(
                    device=render.device, dtype=render.dtype,
                )
                render = render * mask_batch

            for i in range(render.shape[0]):
                row = start + i
                handler.save_reconstruction(render[i], 'decoded', row, Path(f'row{row:06d}.png'))
            num_decoded += render.shape[0]

            if target is not None:
                target_batch = torch.from_numpy(target[start:end]).to(args.device)
                if target_masks is not None:
                    target_batch = target_batch * mask_batch
                psnr, ssim, trial_idx, bin_idx, split_labels, _ = collect_psnr_ssim_metrics_block(
                    render.unsqueeze(1),
                    target_batch.unsqueeze(1),
                    metrics_source,
                    k_trials=end - start,
                    t_bins=1,
                    neural_trial_idx=(
                        trial_idx_flat[start:end] if trial_idx_flat is not None else None
                    ),
                    neural_bin_idx=(
                        bin_idx_flat[start:end].reshape(-1, 1)
                        if bin_idx_flat is not None
                        else None
                    ),
                    trial_split=split_flat[start:end] if split_flat is not None else None,
                )
                psnr_blocks.append(psnr)
                ssim_blocks.append(ssim)
                trial_blocks.append(trial_idx)
                bin_blocks.append(bin_idx)
                split_blocks.append(split_labels)

    log_step(f'Decoded {num_decoded} frames to: {args.out_dir}', level='info')

    if psnr_blocks:
        metrics_path = resolve_metrics_npz_path(None, args.out_dir)
        saved_metrics = save_psnr_ssim_metrics_npz(
            metrics_path,
            psnr_blocks=psnr_blocks,
            ssim_blocks=ssim_blocks,
            neural_trial_blocks=trial_blocks,
            neural_bin_blocks=bin_blocks,
            trial_split_blocks=split_blocks,
            source_file_rows=[str(metrics_source)] * sum(b.shape[0] for b in trial_blocks),
        )
        log_step(f'Saved PSNR/SSIM metrics to: {metrics_path}', level='info')
        _print_combined_metrics_summary(metrics_path, saved_metrics)


if __name__ == '__main__':
    main()
