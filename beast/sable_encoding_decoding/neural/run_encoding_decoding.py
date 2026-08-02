"""CLI entry point for neural encoding/decoding: RRR + CNN, orchestrated by Ray Tune.

Runnable as `python -m beast.sable_encoding_decoding.neural.run_encoding_decoding`.
"""

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import ray
import yaml

from beast.sable_encoding_decoding.neural.decoder import (
    train_cnn_decoder_with_tune,
    train_rrr_decoder_with_tune,
)
from beast.sable_encoding_decoding.neural.encoder import (
    train_cnn_encoder_with_tune,
    train_rrr_encoder_with_tune,
)
from beast.sable_encoding_decoding.neural.utils import (
    LATENT_KIND_LAYOUT,
    get_encoding_decoding_args,
    is_img_tokens_compressed_family,
    set_seed,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RTOL_INTERVALS = 1e-9
_ATOL_INTERVALS = 1e-12


def _default_result_basename(eval_task: str, latent_kind: str | None) -> str:
    """Build the default result basename from the task and latent kind.

    Args:
        eval_task: `'encoding'` or `'decoding'`.
        latent_kind: parsed `--latent_kind`, or `None`.

    Returns:
        Basename such as `encoding_results` or `encoding_results_dino`.
    """
    base = 'encoding_results' if eval_task == 'encoding' else 'decoding_results'
    if latent_kind is not None:
        return f'{base}_{latent_kind}'
    return base


# Parent folder of <latent_root>/<subdir>/<eid> when using layout subdirs
# (see LATENT_KIND_LAYOUT in utils.py).
_SUBDIR_TO_RESULT_SUFFIX = {
    'frame_z': 'frame',
    'pose_mu_s_z': 'mu_s',
    'dino_z': 'dino',
    'psae_z': 'psae',
    'combined_z': 'combined',
    'behavior_z': 'behavior',
}


def _latent_kind_for_result_basename(
    latent_kind: str | None, latent_session_dir: str,
) -> str | None:
    """Label for output filenames: explicit `--latent_kind`, else inferred from session path.

    Args:
        latent_kind: parsed `--latent_kind`, or `None`.
        latent_session_dir: resolved latent session directory.

    Returns:
        Latent kind label, or `None` if it cannot be inferred.
    """
    if latent_kind is not None:
        return latent_kind
    parent = Path(latent_session_dir).parent.name
    if parent.startswith('img_tokens_compressed'):
        return parent
    return _SUBDIR_TO_RESULT_SUFFIX.get(parent)


def _load_psae_z_partition_for_mu_u(model_config_path: str) -> tuple[int, int]:
    """Full PSAE z width (`num_latents`) and `dim_supervised` (`mu_s`; tail is `mu_u`).

    Args:
        model_config_path: path to the training/inference YAML config.

    Returns:
        Tuple `(num_latents, dim_supervised)`.

    Raises:
        KeyError: if `model.auto_encoder.num_latents` is missing from the config.
    """
    with Path(model_config_path).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = cfg['model']
    ae = model.get('auto_encoder') or {}
    if 'num_latents' not in ae:
        raise KeyError(
            'model.auto_encoder.num_latents is required in --model_config for '
            '--latent_kind psae or mu_u',
        )
    num_latents = int(ae['num_latents'])
    lp_ae = ae.get('latent_partition') or {}
    lp_root = model.get('latent_partition') or {}
    dim_supervised = int(lp_ae.get('dim_supervised', lp_root.get('dim_supervised', 6)))
    return num_latents, dim_supervised


def _latent_session_dir_and_trials_npz(
    latent_root: str, eid: str, latent_kind: str | None,
) -> tuple[str, str]:
    """Return (session directory, trials npz filename) for loading latents.

    Args:
        latent_root: `--latent_input_dir`.
        eid: session id.
        latent_kind: parsed `--latent_kind`, or `None`.

    Returns:
        Tuple `(session_dir, trials_npz_filename)`.
    """
    if latent_kind is None:
        return str(Path(latent_root) / eid), 'z_trials.npz'
    if is_img_tokens_compressed_family(latent_kind):
        return (
            str(Path(latent_root) / latent_kind / eid),
            'img_tokens_compressed_trials.npz',
        )
    subdir, fname = LATENT_KIND_LAYOUT[latent_kind]
    return str(Path(latent_root) / subdir / eid), fname


def _permutation_npz_path(permutation_dir: str, eid: str, latent_kind: str | None) -> Path:
    """Resolve the frame-permutation table path matching a session's latent layout.

    Args:
        permutation_dir: `--permutation_dir`.
        eid: session id.
        latent_kind: parsed `--latent_kind`, or `None`.

    Returns:
        Path to `<permutation_dir>/<subdir>/<eid>/permutation.npz`.

    Raises:
        KeyError: if `latent_kind` has no entry in `LATENT_KIND_LAYOUT` and is not an
            `img_tokens_compressed*` kind.
    """
    if is_img_tokens_compressed_family(latent_kind):
        subdir = latent_kind
    elif latent_kind is None:
        subdir = ''
    else:
        subdir = LATENT_KIND_LAYOUT[latent_kind][0]
    return Path(permutation_dir) / subdir / eid / 'permutation.npz'


def _apply_frame_permutation(z: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Shuffle `z` along its flattened trial*time frame axis using a stored permutation.

    Args:
        z: latent tensor of shape `[N_trials, T_time_bins, V_views, D_latent]`.
        perm: permutation indices of length `N_trials * T_time_bins`.

    Returns:
        `z` with rows reordered by `perm` along the flattened frame axis, reshaped back to
        `z`'s original shape.

    Raises:
        ValueError: if `perm`'s length does not match `z`'s trial*time frame count.
    """
    n, t, v, d = z.shape
    if perm.shape[0] != n * t:
        raise ValueError(
            f'permutation length {perm.shape[0]} != N_trials*T_time_bins ({n}*{t}={n * t}) '
            f'for z of shape {z.shape}; the permutation table was likely generated for a '
            'different --latent_kind or a stale trials npz.',
        )
    return z.reshape(n * t, v, d)[perm].reshape(n, t, v, d)


def _resolve_tune_storage_path(
    latent_root: str, latent_kind: str | None, explicit: str | None,
) -> str | None:
    """Ray Tune root: under frame_z / dino_z / combined_z when `latent_kind` is set.

    Args:
        latent_root: `--latent_input_dir`.
        latent_kind: parsed `--latent_kind`, or `None`.
        explicit: value of `--tune_storage_path`, or `None`.

    Returns:
        Resolved Ray Tune storage path, or `None` to fall back to Ray's own default.
    """
    if latent_kind is None:
        return explicit
    if explicit is not None:
        return explicit
    if is_img_tokens_compressed_family(latent_kind):
        return str(Path(latent_root) / latent_kind)
    subdir = LATENT_KIND_LAYOUT[latent_kind][0]
    return str(Path(latent_root) / subdir)


def _resolve_ray_session_temp_dir() -> str:
    """Ray defaults to /tmp/ray; on some HPC nodes /tmp is missing — use a writable path.

    Returns:
        A writable absolute directory path for Ray's `_temp_dir`.

    Raises:
        RuntimeError: if no writable candidate directory could be found.
    """
    bases: list[Path] = []
    ray_tmp = os.environ.get('RAY_TMPDIR')
    if ray_tmp:
        bases.append(Path(ray_tmp).expanduser().resolve())
    for key in ('SLURM_TMPDIR', 'TMPDIR'):
        v = os.environ.get(key)
        if v:
            bases.append(Path(v).expanduser().resolve())
    bases.append(Path(tempfile.gettempdir()).resolve())
    bases.append(Path.home() / '.cache')
    bases.append(_REPO_ROOT / '.ray_tmp')

    for base in bases:
        ray_dir = base / 'ray'
        try:
            ray_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            probe = ray_dir / '.ray_write_probe'
            probe.write_text('ok', encoding='ascii')
            probe.unlink()
            return str(ray_dir)
        except OSError:
            continue
    raise RuntimeError(
        'Could not create a writable Ray temp directory (see FileNotFoundError for /tmp/ray '
        'on some clusters). Set RAY_TMPDIR to an existing, writable absolute path.',
    )


def _log_interval_mismatch_detail(
    *,
    split_name: str,
    latent_row: int,
    neural_trial_idx: int,
    la: np.ndarray,
    ne: np.ndarray,
) -> None:
    """Print diagnostics when latent vs neural interval rows disagree.

    Args:
        split_name: `'train'`, `'val'`, or `'test'`.
        latent_row: row index within the latent intervals array.
        neural_trial_idx: corresponding row index within the neural intervals array.
        la: latent interval row.
        ne: neural interval row.
    """
    la = np.asarray(la)
    ne = np.asarray(ne)
    print('[intervals check] mismatch detail (latent z_trials.npz vs neural *_aligned.npz)')
    print(f'  split={split_name!r}  latent_row={latent_row}  neural_trial_idx={neural_trial_idx}')
    print(f'  latent: shape={la.shape} dtype={la.dtype}')
    print(f'  latent: {np.array2string(la, precision=17, floatmode="unique")}')
    print(f'  neural: shape={ne.shape} dtype={ne.dtype}')
    print(f'  neural: {np.array2string(ne, precision=17, floatmode="unique")}')
    if la.shape != ne.shape:
        print('  reason: shape mismatch (cannot subtract / allclose)')
        return
    la64 = la.astype(np.float64, copy=False)
    ne64 = ne.astype(np.float64, copy=False)
    la_has_nan = bool(np.isnan(la64).any())
    ne_has_nan = bool(np.isnan(ne64).any())
    print(f'  latent has_nan={la_has_nan}  neural has_nan={ne_has_nan}')
    diff = np.abs(la64 - ne64)
    print(f'  abs(latent - neural): {np.array2string(diff, precision=17, floatmode="unique")}')
    if np.all(np.isnan(diff)):
        print('  max abs diff: nan (all elements)')
    else:
        print(f'  max abs diff: {float(np.nanmax(diff))}')
    rel = diff / (np.maximum(np.abs(ne64), np.abs(la64)) + 1e-300)
    if np.all(np.isnan(rel)):
        print('  max relative diff (rough): nan (all elements)')
    else:
        print(f'  max relative diff (rough): {float(np.nanmax(rel))}')
    ok = np.isclose(la64, ne64, rtol=_RTOL_INTERVALS, atol=_ATOL_INTERVALS, equal_nan=True)
    print(f'  np.isclose (equal_nan=True) elementwise: {ok}')
    print(
        f'  tolerance used: rtol={_RTOL_INTERVALS:g} atol={_ATOL_INTERVALS:g} '
        '(equal_nan=False in assert)',
    )


def _maybe_assert_latent_intervals_match_neural(
    latent_data_dict: Any, neural_data_dict: Any,
) -> None:
    """If combined z_trials.npz carries intervals + neural_trial_idx, check vs aligned npz.

    Combined outputs may omit incomplete trials, so we index neural intervals by
    `neural_trial_idx` (same row order as `z_trials_time`: train block, then val, then test).

    Mismatches and missing metadata only emit warnings — the run continues.

    Args:
        latent_data_dict: loaded `z_trials.npz` (an `np.lib.npyio.NpzFile`).
        neural_data_dict: loaded `<eid>_aligned.npz` (an `np.lib.npyio.NpzFile`).
    """
    keys = ('train_intervals', 'val_intervals', 'test_intervals', 'neural_trial_idx')
    missing = [k for k in keys if k not in latent_data_dict.files]
    if missing:
        warnings.warn(
            f'z_trials.npz missing keys {missing}; skipping latent vs neural interval check.',
            stacklevel=2,
        )
        return
    nti = np.asarray(latent_data_dict['neural_trial_idx'])
    n_tr = len(latent_data_dict['train_intervals'])
    n_va = len(latent_data_dict['val_intervals'])
    n_te = len(latent_data_dict['test_intervals'])
    if n_tr + n_va + n_te != len(nti):
        warnings.warn(
            f'neural_trial_idx length {len(nti)} != train+val+test interval counts '
            f'({n_tr}+{n_va}+{n_te}); skipping latent vs neural interval check.',
            stacklevel=2,
        )
        return
    print(
        '[intervals check] comparing latent intervals to neural aligned npz — '
        f'train/val/test rows=({n_tr},{n_va},{n_te}), neural_trial_idx len={len(nti)}',
    )
    splits = (
        ('train', n_tr, 0),
        ('val', n_va, n_tr),
        ('test', n_te, n_tr + n_va),
    )
    for name, n_loc, off in splits:
        ne_all = np.asarray(neural_data_dict[f'{name}_intervals'])
        for i in range(n_loc):
            j = int(nti[off + i])
            if j < 0 or j >= len(ne_all):
                warnings.warn(
                    f'neural_trial_idx={j} out of range for neural {name}_intervals '
                    f'(len={len(ne_all)}); skipping interval check.',
                    stacklevel=2,
                )
                return
            la = np.asarray(latent_data_dict[f'{name}_intervals'][i])
            ne = np.asarray(ne_all[j])
            if la.shape != ne.shape:
                _log_interval_mismatch_detail(
                    split_name=name, latent_row=i, neural_trial_idx=j, la=la, ne=ne,
                )
                warnings.warn(
                    f'Interval shape mismatch for {name} latent row {i} '
                    f'(neural_trial_idx={j}); continuing without interval validation.',
                    stacklevel=2,
                )
                return
            allclose = np.allclose(
                la, ne, rtol=_RTOL_INTERVALS, atol=_ATOL_INTERVALS, equal_nan=False,
            )
            if not allclose:
                _log_interval_mismatch_detail(
                    split_name=name, latent_row=i, neural_trial_idx=j, la=la, ne=ne,
                )
                warnings.warn(
                    f'Interval mismatch for {name} latent row {i} (neural_trial_idx={j}); '
                    'continuing without interval validation.',
                    stacklevel=2,
                )
                return
    print('OK: latent train/val/test_intervals match neural aligned npz via neural_trial_idx.')


def main() -> None:
    """Run the neural encoding/decoding pipeline end to end (RRR + CNN via Ray Tune)."""
    ray_temp = _resolve_ray_session_temp_dir()
    print(f'Ray session temp dir (_temp_dir): {ray_temp}')
    ray.init(num_cpus=2, num_gpus=1, _temp_dir=ray_temp)
    args = get_encoding_decoding_args()

    if args.latent_kind == 'mu_u' and not args.model_config:
        raise ValueError(
            '--model_config is required when --latent_kind is mu_u '
            '(to slice psae_z into mu_u via num_latents and latent_partition.dim_supervised).',
        )
    if args.latent_kind == 'psae' and not args.model_config:
        raise ValueError(
            '--model_config is required when --latent_kind is psae '
            '(to verify z_trials_time last dim matches model.auto_encoder.num_latents).',
        )

    eid = args.eid
    neural_input_dir = Path(args.neural_input_dir) / eid
    latent_input_dir, latent_trials_npz = _latent_session_dir_and_trials_npz(
        args.latent_input_dir, eid, args.latent_kind,
    )
    seed = args.seed
    eval_task = args.eval_task
    tune_storage_path = _resolve_tune_storage_path(
        args.latent_input_dir, args.latent_kind, args.tune_storage_path,
    )
    if tune_storage_path:
        print(f'Ray Tune storage_path: {tune_storage_path}')

    if eval_task == 'encoding':
        print('Neural Encoding:')
    elif eval_task == 'decoding':
        print('Neural Decoding:')
    else:
        raise ValueError(f'Invalid evaluation task: {eval_task}')

    latent_suffix = _latent_kind_for_result_basename(args.latent_kind, latent_input_dir)
    if args.result_name is not None:
        result_basename = args.result_name.strip()
        if result_basename.lower().endswith('.npy'):
            result_basename = result_basename[:-4]
        if not result_basename:
            result_basename = _default_result_basename(eval_task, latent_suffix)
    else:
        result_basename = _default_result_basename(eval_task, latent_suffix)
    save_path = Path(latent_input_dir) / result_basename
    print(f'Saving results to: {save_path}.npy')
    trials_npz_path = Path(latent_input_dir) / latent_trials_npz
    print(f'Loading latent trials from: {trials_npz_path}')

    set_seed(seed)

    result_dict = {}

    print(f'Processing {eid}')

    neural_aligned_npz = neural_input_dir / f'{eid}_aligned.npz'
    neural_data_dict = np.load(neural_aligned_npz, allow_pickle=True)

    if is_img_tokens_compressed_family(args.latent_kind):
        # phase 2: img_token compressed latents are decoded via a dedicated module that
        # does not exist yet; the import path below is where it will live.
        from beast.sable_encoding_decoding.img_token.neural_decoder import (
            build_img_token_neural_decoding_data,
            build_img_token_neural_encoding_data,
            resolve_compressed_trials_npz_path,
        )

        trials_load_path = str(
            resolve_compressed_trials_npz_path(str(trials_npz_path), session_id=eid),
        )
        latent_data_dict = np.load(trials_load_path, allow_pickle=True)

        required_split_keys_img = tuple(f'{s}_z_trials_time' for s in ('train', 'val', 'test'))
        if not all(k in latent_data_dict.files for k in required_split_keys_img):
            raise KeyError(
                f'{trials_load_path}: require {", ".join(required_split_keys_img)}; '
                f'got {sorted(latent_data_dict.files)}',
            )

        if eval_task == 'encoding':
            train_data = build_img_token_neural_encoding_data(
                eid, str(neural_aligned_npz), trials_load_path, session_id=eid,
            )
        else:
            train_data = build_img_token_neural_decoding_data(
                eid, str(neural_aligned_npz), trials_load_path, session_id=eid,
            )

        print(
            f"Train X Shape: {train_data[eid]['X'][0].shape}, "
            f"Train Y Shape: {train_data[eid]['y'][0].shape}",
        )
        print(
            f"Val X Shape: {train_data[eid]['X'][1].shape}, "
            f"Val Y Shape: {train_data[eid]['y'][1].shape}",
        )
        print(
            f"Test X Shape: {train_data[eid]['X'][2].shape}, "
            f"Test Y Shape: {train_data[eid]['y'][2].shape}",
        )

        _maybe_assert_latent_intervals_match_neural(latent_data_dict, neural_data_dict)

        if eval_task == 'encoding':
            cnn_result = train_cnn_encoder_with_tune(
                train_data, num_samples=30, tune_storage_path=tune_storage_path,
            )
            print(
                f"CNN Encoding {eid} Test BPS: {cnn_result[eid]['bps']} "
                f"Test R2: {cnn_result[eid]['r2']}",
            )
            result_dict[eid] = {'cnn': cnn_result[eid]}
        elif eval_task == 'decoding':
            cnn_result = train_cnn_decoder_with_tune(
                train_data, num_samples=30, tune_storage_path=tune_storage_path,
            )
            print(f"CNN Decoding {eid} Test R2: {cnn_result['test'][eid]['r2']}")
            print(f"CNN Decoding {eid} Val R2: {cnn_result['val'][eid]['r2']}")
            result_dict[eid] = {
                'cnn': {'test': cnn_result['test'][eid], 'val': cnn_result['val'][eid]},
            }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, result_dict)
        return

    latent_data_dict = np.load(trials_npz_path, allow_pickle=True)

    if 'z_trials_time' in latent_data_dict.files:
        raise ValueError(
            f'{trials_npz_path}: single-stack z_trials_time is no longer supported; '
            'use neural-aligned combine outputs with train_z_trials_time, val_z_trials_time, '
            'and test_z_trials_time.',
        )
    required_split_keys = tuple(f'{s}_z_trials_time' for s in ('train', 'val', 'test'))
    if not all(k in latent_data_dict.files for k in required_split_keys):
        raise KeyError(
            f'{trials_npz_path}: require {", ".join(required_split_keys)}; '
            f'got {sorted(latent_data_dict.files)}',
        )

    train_emb_raw = np.asarray(latent_data_dict['train_z_trials_time'], dtype=np.float32)
    val_emb_raw = np.asarray(latent_data_dict['val_z_trials_time'], dtype=np.float32)
    test_emb_raw = np.asarray(latent_data_dict['test_z_trials_time'], dtype=np.float32)

    if args.permutation_dir is not None:
        permutation_npz_path = _permutation_npz_path(args.permutation_dir, eid, args.latent_kind)
        print(f'Applying frame permutation from: {permutation_npz_path}')
        permutation_data_dict = np.load(permutation_npz_path, allow_pickle=True)
        train_emb_raw = _apply_frame_permutation(
            train_emb_raw, permutation_data_dict['perm_train'],
        )
        val_emb_raw = _apply_frame_permutation(val_emb_raw, permutation_data_dict['perm_val'])
        test_emb_raw = _apply_frame_permutation(test_emb_raw, permutation_data_dict['perm_test'])

    ref = next((x for x in (train_emb_raw, val_emb_raw, test_emb_raw) if x.size > 0), None)
    if ref is None:
        raise ValueError(f'{trials_npz_path}: all split tensors are empty.')
    _, _, _, D = ref.shape

    if args.latent_kind in ('psae', 'mu_u'):
        num_latents, dim_supervised = _load_psae_z_partition_for_mu_u(args.model_config)
        if D != num_latents:
            raise ValueError(
                f'z_trials_time per-view feature dim D={D} != '
                f'model.auto_encoder.num_latents={num_latents} '
                '(psae z is concat(mu_s, mu_u) along the last dim); ensure --model_config '
                'matches the run that produced psae_z.',
            )
        if args.latent_kind == 'mu_u':
            train_emb_raw = train_emb_raw[..., dim_supervised:]
            val_emb_raw = val_emb_raw[..., dim_supervised:]
            test_emb_raw = test_emb_raw[..., dim_supervised:]

    def _flatten_views(z: np.ndarray) -> np.ndarray:
        """Flatten embeddings across camera view directions (matches former concat path)."""
        k0, t0, v0, d0 = z.shape
        return np.asarray(z.reshape(k0, t0, v0 * d0), dtype=np.float32)

    train_embedding = _flatten_views(train_emb_raw)
    val_embedding = _flatten_views(val_emb_raw)
    test_embedding = _flatten_views(test_emb_raw)

    _maybe_assert_latent_intervals_match_neural(latent_data_dict, neural_data_dict)

    train_neural = neural_data_dict['train_spikes']
    val_neural = neural_data_dict['val_spikes']
    test_neural = neural_data_dict['test_spikes']

    train_data = {eid: {'X': [], 'y': [], 'setup': {}}}

    if eval_task == 'encoding':
        train_data[eid]['X'].append(train_embedding)
        train_data[eid]['X'].append(val_embedding)
        train_data[eid]['X'].append(test_embedding)
        train_data[eid]['y'].append(train_neural)
        train_data[eid]['y'].append(val_neural)
        train_data[eid]['y'].append(test_neural)
    elif eval_task == 'decoding':
        train_data[eid]['X'].append(train_neural)
        train_data[eid]['X'].append(val_neural)
        train_data[eid]['X'].append(test_neural)
        train_data[eid]['y'].append(train_embedding)
        train_data[eid]['y'].append(val_embedding)
        train_data[eid]['y'].append(test_embedding)

    print(
        f"Train X Shape: {train_data[eid]['X'][0].shape}, "
        f"Train Y Shape: {train_data[eid]['y'][0].shape}",
    )
    print(
        f"Val X Shape: {train_data[eid]['X'][1].shape}, "
        f"Val Y Shape: {train_data[eid]['y'][1].shape}",
    )
    print(
        f"Test X Shape: {train_data[eid]['X'][2].shape}, "
        f"Test Y Shape: {train_data[eid]['y'][2].shape}",
    )

    if eval_task == 'encoding':
        rrr_result = train_rrr_encoder_with_tune(
            train_data, num_samples=30, tune_storage_path=tune_storage_path,
        )
        cnn_result = train_cnn_encoder_with_tune(
            train_data, num_samples=30, tune_storage_path=tune_storage_path,
        )
    elif eval_task == 'decoding':
        rrr_result = train_rrr_decoder_with_tune(
            train_data, num_samples=30, tune_storage_path=tune_storage_path,
        )
        cnn_result = train_cnn_decoder_with_tune(
            train_data, num_samples=30, tune_storage_path=tune_storage_path,
        )
    else:
        raise ValueError(f'Invalid evaluation task: {eval_task}')

    if eval_task == 'encoding':
        print(
            f"RRR Encoding {eid} Test BPS: {rrr_result[eid]['bps']} "
            f"Test R2: {rrr_result[eid]['r2']}",
        )
        print(
            f"CNN Encoding {eid} Test BPS: {cnn_result[eid]['bps']} "
            f"Test R2: {cnn_result[eid]['r2']}",
        )
    elif eval_task == 'decoding':
        print(f"RRR Decoding {eid} Test R2: {rrr_result[eid]['r2']}")
        print(f"CNN Decoding {eid} Test R2: {cnn_result['test'][eid]['r2']}")
        print(f"CNN Decoding {eid} Val R2: {cnn_result['val'][eid]['r2']}")

    if eval_task == 'encoding':
        result_dict[eid] = {'rrr': rrr_result[eid], 'cnn': cnn_result[eid]}
    else:
        result_dict[eid] = {
            'rrr': rrr_result[eid],
            'cnn': {'test': cnn_result['test'][eid], 'val': cnn_result['val'][eid]},
        }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, result_dict)


if __name__ == '__main__':
    main()
