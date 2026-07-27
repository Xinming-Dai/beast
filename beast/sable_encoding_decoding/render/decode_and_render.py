"""Load saved img_token latents from an inference `.npz` and render them via the Sable decoder.

Camera tensors (`c2w_input_out`, `fxfycxcy_input_out`, `c2w_target_out`, `fxfycxcy_target_out`)
are read directly from the `.npz` when present. For estimated tokens that do not carry cameras,
`--camera-npz` can point to the img_tokens camera sidecar; cameras are then selected by
`trial_split` + `neural_trial_idx` + `neural_bin_idx`/time-bin order.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from beast.api.model import Model
from beast.data.sable_dataset import collate_with_correspondence_padding
from beast.inference import save_gaussian_pointclouds
from beast.logging import log_step
from beast.models.model_utils.train_vis import save_training_visuals
from beast.sable_encoding_decoding.img_token.saved_tokens_io import (
    load_z_from_npz_file,
    sorted_img_tokens_npz_paths,
)
from beast.sable_encoding_decoding.render.decode_utils import (
    combine_metrics_shards_to_combined_npz,
    delete_metrics_shards_for_sources,
    filter_img_tokens_npz_paths_by_neural_trial,
    is_render_complete,
    metrics_shard_path,
    parse_neural_trial_index_arg,
    parse_neural_trial_range_arg,
    render_done_marker_path,
    save_single_token_metrics_npz,
    write_render_done_marker,
)
from beast.sable_encoding_decoding.render.metrics import (
    collect_psnr_ssim_metrics_block,
    collect_temporal_metrics_block,
    resolve_metrics_npz_path,
)
from beast.train_sable import _resolve_dataset_class

# Keys written into the .npz by the inference save routine (flat and IBL-trial formats).
_CAM_FLAT_KEYS = ('c2w_input_out', 'fxfycxcy_input_out', 'c2w_target_out', 'fxfycxcy_target_out')
_CAM_TRIAL_SUFFIX = '_trials'


def _parse_include_splits(expr: str) -> list[str]:
    """Parse a comma-separated `--include-splits` argument into a validated split list.

    Args:
        expr: comma-separated splits, e.g. `'train,val'`.

    Returns:
        Lowercased, whitespace-trimmed split names.

    Raises:
        argparse.ArgumentTypeError: if `expr` is blank or contains an invalid split name.
    """
    valid_splits = {'train', 'val', 'test'}
    splits = [x.strip().lower() for x in expr.split(',') if x.strip()]
    if not splits:
        raise argparse.ArgumentTypeError('--include-splits must contain at least one split')
    invalid = [x for x in splits if x not in valid_splits]
    if invalid:
        raise argparse.ArgumentTypeError(
            f'Invalid split(s) {invalid}; expected comma-separated train,val,test',
        )
    return splits


def _z_source_npz_paths(z_source: Path) -> list[Path]:
    """Return one `.npz` file, or every direct-child `img_tokens*.npz` under a directory.

    Args:
        z_source: a single `.npz` file, or a directory of `img_tokens*.npz` files.

    Returns:
        Sorted list of `.npz` paths.

    Raises:
        ValueError: if `z_source` is a file that is not `.npz`.
        FileNotFoundError: if `z_source` does not exist, or a directory has no matching files.
    """
    p = Path(z_source).resolve()
    if p.is_file():
        if p.suffix.lower() != '.npz':
            raise ValueError(f'--z-source must be a .npz file or directory; got {p}')
        return [p]
    if p.is_dir():
        paths = sorted_img_tokens_npz_paths(p, recursive=False)
        if not paths:
            raise FileNotFoundError(f'No img_tokens*.npz files directly under {p}')
        return paths
    raise FileNotFoundError(f'--z-source not found: {p}')


def _resolve_npz_path(path: Path) -> Path:
    """Resolve `path` to a single `.npz` file, descending into a directory if needed."""
    path = Path(path).resolve()
    if path.is_dir():
        cands = sorted_img_tokens_npz_paths(path, recursive=True)
        if not cands:
            raise FileNotFoundError(f'No img_tokens*.npz under {path}')
        path = cands[0]
    return path


def _load_cameras_from_npz(path: Path) -> dict[str, np.ndarray] | None:
    """Return camera arrays from the `.npz` if they were saved by the inference run, else `None`.

    Flat format: keys are `c2w_input_out`, ... shape `(B, V, 4, 4)` / `(B, V, 4)`.
    IBL format: keys are `c2w_input_out_trials`, ... shape `(K, T, V, 4, 4)` -> reshaped to
    `(K*T, ...)`.
    """
    try:
        path = _resolve_npz_path(path)
    except FileNotFoundError:
        return None

    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        # flat format
        if all(k in keys for k in _CAM_FLAT_KEYS):
            return {k: np.asarray(d[k], dtype=np.float32) for k in _CAM_FLAT_KEYS}
        # ibl trial format: keys carry a _trials suffix, shape (K, T, V, ...)
        trial_keys = [k + _CAM_TRIAL_SUFFIX for k in _CAM_FLAT_KEYS]
        if all(k in keys for k in trial_keys):
            cams: dict[str, np.ndarray] = {}
            for flat_key, trial_key in zip(_CAM_FLAT_KEYS, trial_keys, strict=True):
                arr = np.asarray(d[trial_key], dtype=np.float32)  # (K, T, V, ...)
                k_trials, t_bins = arr.shape[0], arr.shape[1]
                cams[flat_key] = arr.reshape(k_trials * t_bins, *arr.shape[2:])
            return cams
    return None


def _load_token_index_metadata(
    z_source: Path,
    *,
    k_trials: int,
    t_bins: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return per-token-row split, trial index, and bin index arrays.

    Raises:
        KeyError: if `trial_split`/`neural_trial_idx` are missing from `z_source`.
        ValueError: if metadata array lengths cannot index the `(k_trials, t_bins)` shape.
    """
    path = _resolve_npz_path(z_source)
    flat = k_trials * t_bins
    with np.load(path, allow_pickle=True) as d:
        if 'trial_split' not in d.files or 'neural_trial_idx' not in d.files:
            raise KeyError(
                f'{path}: --camera-npz indexing needs trial_split and neural_trial_idx; '
                f'got {d.files}',
            )
        split_raw = [
            str(x).lower() for x in np.asarray(d['trial_split'], dtype=object).reshape(-1)
        ]
        trial_raw = np.asarray(d['neural_trial_idx'], dtype=np.int64).reshape(-1)
        bin_raw = (
            np.asarray(d['neural_bin_idx'], dtype=np.int64).reshape(-1)
            if 'neural_bin_idx' in d.files
            else None
        )

    if len(trial_raw) == flat:
        trial_idx = trial_raw
    elif len(trial_raw) == k_trials:
        trial_idx = np.repeat(trial_raw, t_bins)
    else:
        raise ValueError(
            f'{path}: neural_trial_idx length {len(trial_raw)} cannot index z shape '
            f'({k_trials}, {t_bins}, ...)',
        )

    if len(split_raw) == flat:
        splits = split_raw
    elif len(split_raw) == k_trials:
        splits = [s for s in split_raw for _ in range(t_bins)]
    else:
        raise ValueError(
            f'{path}: trial_split length {len(split_raw)} cannot index z shape '
            f'({k_trials}, {t_bins}, ...)',
        )

    if bin_raw is not None:
        if len(bin_raw) != flat:
            raise ValueError(f'{path}: neural_bin_idx length {len(bin_raw)} != K*T {flat}')
        bin_idx = bin_raw
    else:
        bin_idx = np.tile(np.arange(t_bins, dtype=np.int64), k_trials)

    return splits, trial_idx, bin_idx


def _load_cameras_from_sidecar(
    camera_npz: Path,
    z_source: Path,
    *,
    k_trials: int,
    t_bins: int,
) -> dict[str, np.ndarray]:
    """Load per-row cameras from a camera sidecar `.npz`, indexed by split/trial/bin.

    Raises:
        KeyError: if the sidecar or `z_source` lacks the split/trial index metadata.
    """
    splits, trial_idx, bin_idx = _load_token_index_metadata(
        z_source,
        k_trials=k_trials,
        t_bins=t_bins,
    )
    camera_npz = Path(camera_npz).resolve()
    cams: dict[str, list[np.ndarray]] = {k: [] for k in _CAM_FLAT_KEYS}
    with np.load(camera_npz, allow_pickle=True) as d:
        if 'trial_split' not in d.files or 'neural_trial_idx' not in d.files:
            raise KeyError(
                f'{camera_npz}: camera sidecar needs trial_split and neural_trial_idx; '
                f'got {d.files}',
            )
        side_splits = [
            str(x).lower() for x in np.asarray(d['trial_split'], dtype=object).reshape(-1)
        ]
        side_trials = np.asarray(d['neural_trial_idx'], dtype=np.int64).reshape(-1)
        split_trial_to_local: dict[str, dict[int, int]] = {}
        for split in sorted(set(side_splits)):
            global_rows = [i for i, s in enumerate(side_splits) if s == split]
            split_trial_to_local[split] = {
                int(side_trials[global_i]): local_i
                for local_i, global_i in enumerate(global_rows)
            }

        for split, trial, bin_i in zip(splits, trial_idx, bin_idx, strict=True):
            if split not in split_trial_to_local:
                raise KeyError(f'{camera_npz}: no camera rows for split {split!r}')
            try:
                local_trial_i = split_trial_to_local[split][int(trial)]
            except KeyError as exc:
                raise KeyError(
                    f'{camera_npz}: no camera row for split={split!r}, '
                    f'neural_trial_idx={int(trial)}',
                ) from exc
            for key in _CAM_FLAT_KEYS:
                side_key = f'{split}_{key}'
                if side_key not in d.files:
                    raise KeyError(f'{camera_npz}: missing camera key {side_key!r}')
                arr = np.asarray(d[side_key], dtype=np.float32)
                if int(bin_i) < 0 or int(bin_i) >= arr.shape[1]:
                    raise IndexError(
                        f'{camera_npz}: neural_bin_idx={int(bin_i)} out of range for '
                        f'{side_key} with T={arr.shape[1]}',
                    )
                cams[key].append(arr[local_trial_i, int(bin_i)])

    return {key: np.stack(vals, axis=0).astype(np.float32) for key, vals in cams.items()}


def _apply_dataloader_overrides(config: dict, args: argparse.Namespace) -> None:
    """Apply CLI dataloader overrides onto a beast dict config in place.

    beast's config is a plain nested dict (not attribute-accessible like the source pipeline's
    config object), so overrides are applied via dict indexing rather than reusing
    `saved_tokens_io.apply_dataloader_overrides`.

    Args:
        config: mutable beast config dict with `'training'` / `'model'` keys.
        args: parsed CLI namespace; only attributes that are set are applied.
    """
    training = config['training']
    if getattr(args, 'dataset_path', None):
        training['dataset_path'] = args.dataset_path
    if getattr(args, 'batch_size', None) is not None:
        training['batch_size_per_gpu'] = args.batch_size
    if getattr(args, 'num_workers', None) is not None:
        training['num_workers'] = args.num_workers
    if getattr(args, 'include_splits', None) is not None:
        training['ibl_precache_splits'] = args.include_splits
    if getattr(args, 'ibl_session_eids', None) is not None:
        training['ibl_inference_session_eids'] = args.ibl_session_eids
    if getattr(args, 'vda_cache_root', None) is not None:
        config['model'].setdefault('vda', {})
        config['model']['vda']['cache_root'] = args.vda_cache_root
    if getattr(args, 'correspondence_cache_root', None):
        config['model'].setdefault('merge_pcd', {})
        config['model']['merge_pcd'].setdefault('use_correspondences', {})
        config['model']['merge_pcd']['use_correspondences']['cache_root'] = (
            args.correspondence_cache_root
        )
    if getattr(args, 'ibl_precache_valid_index', None) is not None:
        training['ibl_precache_valid_index'] = args.ibl_precache_valid_index


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the decode-and-render entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__.strip())
    p.add_argument(
        '--z-source',
        type=Path,
        required=True,
        help=(
            'img_tokens .npz (z, z_trials, or train/val/test_z_trials_time) or a directory of '
            'direct-child img_tokens*.npz files (e.g. img_tokens_batch*, img_tokens_estimated_*). '
            'Directory mode runs one decode per file (sorted by name); each file uses dataloader '
            'batch index --sync-batch-index + file_index.'
        ),
    )
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument(
        '--combine-metrics-only',
        action='store_true',
        help=(
            'Load per-token shards from <out-dir>/metrics_shards/ for this --z-source list, '
            'write the combined metrics .npz, and exit (no model or dataloader).'
        ),
    )
    p.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore existing metrics shards / render markers and redo every token file.',
    )
    p.add_argument(
        '--camera-npz',
        type=Path,
        default=None,
        help=(
            'Optional img_tokens_camera_parameters.npz sidecar. Used when --z-source does not '
            'contain camera tensors; rows are selected by trial_split/neural_trial_idx/bin.'
        ),
    )
    p.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help=(
            'Directory containing config.yaml and a *best.ckpt, loaded via '
            'beast.api.model.Model.from_dir. Required unless --combine-metrics-only.'
        ),
    )
    p.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    p.add_argument('--dataset-path', type=str, default=None)
    p.add_argument('--correspondence-cache-root', type=str, default=None)
    p.add_argument('--vda-cache-root', type=str, default=None)
    p.add_argument('--ibl-session-eids', type=str, default=None)
    p.add_argument(
        '--include-splits',
        type=_parse_include_splits,
        default=None,
        metavar='SPLITS',
        help='Comma-separated IBL precache splits to load for the dataloader (train,val,test).',
    )
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--ibl-precache-valid-index', action='store_true', default=None)
    p.add_argument(
        '--no-ibl-precache-valid-index',
        action='store_false',
        dest='ibl_precache_valid_index',
    )
    p.add_argument(
        '--sync-batch-index',
        type=int,
        default=0,
        help=(
            'Dataloader batch index for the first --z-source .npz. If --z-source is a directory, '
            'the i-th sorted *.npz uses batch index sync_batch_index + i.'
        ),
    )
    nt_group = p.add_mutually_exclusive_group()
    nt_group.add_argument(
        '--neural-trial-index',
        type=parse_neural_trial_index_arg,
        default=None,
        metavar='IDS',
        help=(
            'Comma-separated neural_trial_idx values; keep only token .npz files whose '
            'neural_trial_idx is constant and matches (filename neuraltrialXXXX checked when '
            'present).'
        ),
    )
    nt_group.add_argument(
        '--neural-trial-range',
        type=parse_neural_trial_range_arg,
        default=None,
        metavar='RANGE',
        help='Inclusive neural_trial_idx range: two numbers LO,HI or one number N meaning 0..N.',
    )
    p.add_argument(
        '--max-render-samples',
        type=int,
        default=64,
        help='Max samples (rows) to render per .npz file; capped by K*T in each file.',
    )
    p.add_argument('--max-render-views', type=int, default=2)
    p.add_argument(
        '--metrics-only',
        '--save-metrics-only',
        action='store_true',
        help=(
            'Decode all rows and save only PSNR/SSIM metrics to an .npz; skips pointclouds '
            'and render visualizations. Metrics are saved with shape [N, T, 2].'
        ),
    )
    p.add_argument(
        '--metrics-npz',
        type=Path,
        default=None,
        help='Output .npz for --metrics-only (default: <out-dir>/psnr_ssim_metrics.npz).',
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the decode-and-render pipeline end to end (CLI entry point)."""
    args = parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = _z_source_npz_paths(args.z_source)
    z_src = Path(args.z_source).resolve()
    if z_src.is_dir():
        log_step(
            f'Found {len(npz_paths)} img_tokens*.npz file(s) under {z_src} (sorted by name).',
            level='info',
        )
    else:
        log_step(f'Single --z-source .npz: {npz_paths[0]}', level='info')

    if args.neural_trial_index is not None:
        npz_paths = filter_img_tokens_npz_paths_by_neural_trial(
            npz_paths, allowed_indices=args.neural_trial_index,
        )
        log_step(
            f'After --neural-trial-index filter: {len(npz_paths)} file(s) '
            f'(ids {sorted(args.neural_trial_index)}).',
            level='info',
        )
    elif args.neural_trial_range is not None:
        lo, hi = args.neural_trial_range
        npz_paths = filter_img_tokens_npz_paths_by_neural_trial(
            npz_paths, inclusive_range=args.neural_trial_range,
        )
        log_step(
            f'After --neural-trial-range filter: {len(npz_paths)} file(s) '
            f'(trial id [{lo}, {hi}] inclusive).',
            level='info',
        )

    n_npz = len(npz_paths)

    if args.combine_metrics_only:
        metrics_npz = resolve_metrics_npz_path(args.metrics_npz, out_dir)
        merged, missing = combine_metrics_shards_to_combined_npz(
            out_dir, npz_paths, metrics_npz, allow_missing=True,
        )
        if merged is None:
            log_step(
                f'No metrics shards under {metrics_shard_path(out_dir, npz_paths[0]).parent} '
                'for this --z-source list; nothing written.',
                level='warning',
            )
            sys.exit(1)
        if missing:
            ms = ','.join(str(i) for i in missing[:48])
            more = '...' if len(missing) > 48 else ''
            log_step(
                f'Combined metrics omit {len(missing)} missing shard(s) at source indices '
                f'[{ms}{more}].',
                level='warning',
            )
        n_removed = delete_metrics_shards_for_sources(out_dir, npz_paths)
        log_step(
            f'Removed {n_removed} metrics shard(s) after writing combined metrics.', level='info',
        )
        log_step(
            f'[combine-metrics-only] Wrote {metrics_npz}  '
            f"psnr_shape={merged['psnr'].shape}  ssim_shape={merged['ssim'].shape}  "
            f"avg_psnr={float(merged['average_psnr']):.4f}  "
            f"avg_ssim={float(merged['average_ssim']):.4f}",
            level='info',
        )
        return

    if not args.model_dir:
        raise SystemExit(
            '--model-dir is required unless --combine-metrics-only '
            '(omit it only when merging existing metric shards).',
        )

    wrapped = Model.from_dir(args.model_dir)
    config = wrapped.config
    config['inference'] = True
    config['evaluation'] = config.get('evaluation', False)
    _apply_dataloader_overrides(config, args)

    device = torch.device(args.device)
    model = wrapped.model.to(device)
    model.eval()
    p0 = next(model.parameters())
    log_step(f'Loaded model from {args.model_dir}', level='info')

    training = config['training']
    dataset_cls = _resolve_dataset_class(
        training.get('dataset_name', 'beast.data.sable_dataset.SABLEDataset'),
    )
    include_splits = args.include_splits if args.include_splits is not None else ['train', 'val']
    dataset = dataset_cls(config, include_splits=include_splits)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(training.get('batch_size_per_gpu', 1)),
        shuffle=False,
        num_workers=int(training.get('num_workers', 4)),
        collate_fn=collate_with_correspondence_padding,
        drop_last=False,
    )
    log_step('Built inference dataloader.', level='info')

    total_vis = 0
    dataloader_iter = iter(dataloader)
    start_batch_idx = int(args.sync_batch_index)
    for skipped_idx in range(start_batch_idx):
        try:
            next(dataloader_iter)
        except StopIteration as exc:
            raise IndexError(
                f'Dataloader ended before --sync-batch-index={start_batch_idx} '
                f'(stopped while skipping batch {skipped_idx}).',
            ) from exc
    allow_resume = not args.no_resume
    for file_i, npz_path in enumerate(npz_paths):
        batch_idx = start_batch_idx + file_i

        if args.metrics_only and allow_resume and metrics_shard_path(out_dir, npz_path).is_file():
            log_step(
                f'[{file_i + 1}/{n_npz}] Resume: skip decode; metrics shard exists: '
                f'{metrics_shard_path(out_dir, npz_path).name}',
                level='info',
            )
            try:
                next(dataloader_iter)
            except StopIteration as exc:
                raise IndexError(
                    f'Dataloader ended before batch_idx={batch_idx}; {n_npz} .npz file(s) '
                    f'require consecutive batches starting at {start_batch_idx}.',
                ) from exc
            continue

        if (
            not args.metrics_only
            and allow_resume
            and is_render_complete(render_done_marker_path(out_dir, batch_idx))
        ):
            log_step(
                f'[{file_i + 1}/{n_npz}] Resume: skip decode; render marker exists for '
                f'batch_{batch_idx:04d}',
                level='info',
            )
            try:
                next(dataloader_iter)
            except StopIteration as exc:
                raise IndexError(
                    f'Dataloader ended before batch_idx={batch_idx}; {n_npz} .npz file(s) '
                    f'require consecutive batches starting at {start_batch_idx}.',
                ) from exc
            continue

        log_step(f'[{file_i + 1}/{n_npz}] Loading and decoding: {npz_path}', level='info')
        z = load_z_from_npz_file(npz_path)
        log_step(
            f'[{file_i + 1}/{n_npz}] load_z: done shape={tuple(z.shape)} dtype={z.dtype}',
            level='info',
        )
        k, t_bins, l_tok, d_feat = z.shape
        flat = k * t_bins

        try:
            batch = next(dataloader_iter)
        except StopIteration as exc:
            raise IndexError(
                f'Dataloader ended before batch_idx={batch_idx}; {n_npz} .npz file(s) '
                f'require consecutive batches starting at {start_batch_idx}.',
            ) from exc
        # move batch to device (plain dict comprehension; no move_batch_to_device helper in beast)
        batch = {k_: v.to(device) if torch.is_tensor(v) else v for k_, v in batch.items()}
        log_step(f'[{file_i + 1}/{n_npz}] batch moved to {device}', level='info')

        b_img = int(batch['image'].shape[0])
        if flat != b_img:
            raise ValueError(
                f'{npz_path}: z spans K*T={flat} rows but the dataloader batch has B={b_img}. '
                f'--sync-batch-index={args.sync_batch_index} (file uses batch_idx={batch_idx}) '
                'may not match this .npz file, or --batch-size is inconsistent.',
            )

        m = flat if args.metrics_only else min(int(args.max_render_samples), flat)
        log_step(
            f'  batch_idx={batch_idx}  K*T={flat}  render m={m}  ({npz_path.name})',
            level='info',
        )

        npz_cams = _load_cameras_from_npz(npz_path)
        if npz_cams is None and args.camera_npz is not None:
            if file_i == 0:
                log_step(f'Loading cameras from sidecar: {args.camera_npz}', level='info')
            npz_cams = _load_cameras_from_sidecar(
                args.camera_npz,
                npz_path,
                k_trials=k,
                t_bins=t_bins,
            )
        elif npz_cams is not None and file_i == 0:
            log_step('Loading cameras from .npz (c2w_*_out / fxfycxcy_*_out).', level='info')
        if npz_cams is None:
            raise FileNotFoundError(
                'No camera tensors found in the token .npz. Pass --camera-npz pointing to '
                'img_tokens_camera_parameters.npz for estimated-token files.',
            )

        def _to_tensor(
            key: str, *, _cams: dict[str, np.ndarray] = npz_cams, _m: int = m,
        ) -> torch.Tensor:
            """Load one camera key, slice to `m` rows, and move to the model's device/dtype."""
            return torch.from_numpy(_cams[key][:_m]).to(device=device, dtype=p0.dtype)

        c2w_input = _to_tensor('c2w_input_out')
        fxfycxcy_input = _to_tensor('fxfycxcy_input_out')
        c2w_target = _to_tensor('c2w_target_out')
        fxfycxcy_target = _to_tensor('fxfycxcy_target_out')
        v_input = int(c2w_input.shape[1])

        if l_tok % v_input != 0:
            raise ValueError(
                f'{npz_path}: token count L={l_tok} is not divisible by v_input={v_input}. '
                'The .npz file may not have been produced with this model configuration.',
            )
        n_tokens_per_view = l_tok // v_input

        z_flat = z.reshape(flat, l_tok, d_feat)[:m]
        all_tokens = (
            torch.from_numpy(z_flat)
            .to(device=p0.device, dtype=p0.dtype)
            .reshape(m, v_input, n_tokens_per_view, d_feat)
        )

        data = {
            key: val[:m] if torch.is_tensor(val) and val.shape[0] >= m else val
            for key, val in batch.items()
        }

        log_step(
            f'[{file_i + 1}/{n_npz}] predict_frame: starting m={m} v_input={v_input} '
            f'n_tokens_per_view={n_tokens_per_view}',
            level='info',
        )
        with torch.no_grad():
            result = model.predict_frame_from_all_tokens(
                all_tokens,
                c2w_input,
                fxfycxcy_input,
                c2w_target,
                fxfycxcy_target,
                data,
            )
        log_step(
            f'[{file_i + 1}/{n_npz}] predict_frame: done '
            f'render.shape={tuple(result.render.shape)}',
            level='info',
        )

        if 'target_indices' in data:
            tidx = data['target_indices']
            imgs = data['image']
            result.target_image = torch.stack([imgs[i, tidx[i]] for i in range(m)], dim=0)
        else:
            result.target_image = data['image'][:m]

        if args.metrics_only:
            log_step(
                f'[{file_i + 1}/{n_npz}] metrics: computing PSNR/SSIM/temporal '
                f'render={tuple(result.render.shape)} target={tuple(result.target_image.shape)}',
                level='info',
            )
            (
                psnr_block,
                ssim_block,
                neural_trial_idx,
                neural_bin_idx,
                trial_split,
                source_files,
            ) = collect_psnr_ssim_metrics_block(
                result.render,
                result.target_image,
                npz_path,
                k_trials=k,
                t_bins=t_bins,
            )
            temporal_blocks = collect_temporal_metrics_block(
                result.render,
                result.target_image,
                npz_path,
                k_trials=k,
                t_bins=t_bins,
            )
            save_single_token_metrics_npz(
                metrics_shard_path(out_dir, npz_path),
                psnr_block=psnr_block,
                ssim_block=ssim_block,
                neural_trial_idx=neural_trial_idx,
                neural_bin_idx=neural_bin_idx,
                trial_split=trial_split,
                source_files=source_files,
                temporal_blocks=temporal_blocks,
            )
            log_step(
                f'[{file_i + 1}/{n_npz}] Metrics collected: {npz_path.name}  '
                f'psnr_shape={psnr_block.shape}  ssim_shape={ssim_block.shape}  '
                f'temporal_metrics={list(temporal_blocks)}',
                level='info',
            )
            continue

        batch_out_dir = out_dir / f'batch_{batch_idx:04d}'
        vis_dir = batch_out_dir / 'render_visuals'
        batch_out_dir.mkdir(parents=True, exist_ok=True)
        # beast has no combined save_and_vis_gaussian_pointclouds helper: call the pointcloud
        # save and visual save routines separately (2-call replacement for the source's 1 call).
        save_gaussian_pointclouds(vars(result), batch_out_dir, batch_idx=batch_idx, max_samples=m)
        log_step(
            f'Saved gaussian pointclouds for batch {batch_idx} to {batch_out_dir}', level='info',
        )

        vis_paths = save_training_visuals(
            vis_dir,
            result=result,
            batch=data,
            step=batch_idx,
            max_samples=m,
            max_views=args.max_render_views,
        )
        total_vis += len(vis_paths)
        log_step(
            f'Saved {len(vis_paths)} render visual(s) for batch {batch_idx} to {vis_dir}',
            level='info',
        )
        write_render_done_marker(render_done_marker_path(out_dir, batch_idx), npz_path)
        log_step(f'[{file_i + 1}/{n_npz}] Done: {npz_path.name}', level='info')

    if args.metrics_only:
        metrics_npz = resolve_metrics_npz_path(args.metrics_npz, out_dir)
        metrics, missing = combine_metrics_shards_to_combined_npz(
            out_dir, npz_paths, metrics_npz, allow_missing=True,
        )
        if metrics is None:
            log_step(
                f'No metrics shards under {metrics_shard_path(out_dir, npz_paths[0]).parent}; '
                'combined file not written.',
                level='warning',
            )
            return
        if missing:
            ms = ','.join(str(i) for i in missing[:48])
            more = '...' if len(missing) > 48 else ''
            log_step(
                f'Combined metrics omit {len(missing)} missing shard(s) at source indices '
                f'[{ms}{more}]; re-run without --no-resume to compute them.',
                level='warning',
            )
        n_removed = delete_metrics_shards_for_sources(out_dir, npz_paths)
        log_step(
            f'Removed {n_removed} metrics shard(s) after writing combined metrics.', level='info',
        )
        log_step(
            f'Saved PSNR/SSIM metrics to {metrics_npz}  '
            f"psnr_shape={metrics['psnr'].shape}  ssim_shape={metrics['ssim'].shape}  "
            f"avg_psnr={float(metrics['average_psnr']):.4f}  "
            f"avg_ssim={float(metrics['average_ssim']):.4f}",
            level='info',
        )
        return

    log_step(
        f'Finished all {n_npz} .npz file(s). Total render visual(s) written: {total_vis}',
        level='info',
    )


if __name__ == '__main__':
    main()
