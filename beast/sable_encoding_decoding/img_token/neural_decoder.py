"""Neural decoding/encoding data assembly for PCA-compressed img tokens (TCN / KeypointsNetwork).

Bridges `img_token_compressed` trials (`beast.sable_encoding_decoding.img_token.pca_compression`
outputs, assembled by `trials_assembly`) with the neural decoder/encoder training entry points in
`beast.sable_encoding_decoding.neural.decoder`.
"""

from pathlib import Path
from typing import Any

import numpy as np

from beast.sable_encoding_decoding.img_token.pca_compression import split_z_by_trial_split
from beast.sable_encoding_decoding.img_token.trials_assembly import _session_subdir_key
from beast.sable_encoding_decoding.neural.decoder import train_cnn_decoder_with_tune

# --------------------------------------------------------------------------- #
# Neural decoding (TCN / KeypointsNetwork via Ray Tune)
# --------------------------------------------------------------------------- #


def _flatten_token_components(z: np.ndarray) -> np.ndarray:
    """`(K, T, L, D)` -> `(K, T, L*D)`; `train_cnn_decoder` uses `embed_size = y.shape[-1]`."""
    k, t, n_tok, d = z.shape
    return z.reshape(k, t, n_tok * d).astype(np.float32)


def build_decoding_data_dict(
    eid: str,
    neural_train: np.ndarray,
    neural_val: np.ndarray,
    neural_test: np.ndarray,
    img_token_compressed_train: np.ndarray,
    img_token_compressed_val: np.ndarray,
    img_token_compressed_test: np.ndarray,
) -> dict[str, Any]:
    """Build a decoding data dict: `X` = neural `(K, T, C)`, `y` = flattened compressed tokens.

    Args:
        eid: session id, used as the sole key of the returned dict.
        neural_train: train-split neural array `(K, T, C)`.
        neural_val: val-split neural array `(K, T, C)`.
        neural_test: test-split neural array `(K, T, C)`.
        img_token_compressed_train: train-split compressed tokens `(K, T, L, D)`.
        img_token_compressed_val: val-split compressed tokens `(K, T, L, D)`.
        img_token_compressed_test: test-split compressed tokens `(K, T, L, D)`.

    Returns:
        `{eid: {'X': [train, val, test], 'y': [train, val, test], 'setup': {}}}` with `y` flattened
        to `(K, T, L*D)`.
    """
    return {
        eid: {
            'X': [neural_train, neural_val, neural_test],
            'y': [
                _flatten_token_components(img_token_compressed_train),
                _flatten_token_components(img_token_compressed_val),
                _flatten_token_components(img_token_compressed_test),
            ],
            'setup': {},
        },
    }


def build_encoding_data_dict(
    eid: str,
    img_token_compressed_train: np.ndarray,
    img_token_compressed_val: np.ndarray,
    img_token_compressed_test: np.ndarray,
    neural_train: np.ndarray,
    neural_val: np.ndarray,
    neural_test: np.ndarray,
) -> dict[str, Any]:
    """Build an encoding data dict: `X` = flattened compressed tokens, `y` = neural `(K, T, C)`.

    Args:
        eid: session id, used as the sole key of the returned dict.
        img_token_compressed_train: train-split compressed tokens `(K, T, L, D)`.
        img_token_compressed_val: val-split compressed tokens `(K, T, L, D)`.
        img_token_compressed_test: test-split compressed tokens `(K, T, L, D)`.
        neural_train: train-split neural array `(K, T, C)`.
        neural_val: val-split neural array `(K, T, C)`.
        neural_test: test-split neural array `(K, T, C)`.

    Returns:
        `{eid: {'X': [train, val, test], 'y': [train, val, test], 'setup': {}}}` with `X` flattened
        to `(K, T, L*D)`.
    """
    return {
        eid: {
            'X': [
                _flatten_token_components(img_token_compressed_train),
                _flatten_token_components(img_token_compressed_val),
                _flatten_token_components(img_token_compressed_test),
            ],
            'y': [neural_train, neural_val, neural_test],
            'setup': {},
        },
    }


def split_compressed_by_trial_split(
    x_compressed: np.ndarray,
    trial_split_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience alias for `split_z_by_trial_split` with compressed latents `(N, T, L, k)`."""
    return split_z_by_trial_split(x_compressed, trial_split_labels)


def resolve_compressed_trials_npz_path(
    path: str | Path,
    session_id: str | None = None,
) -> Path:
    """Resolve the real `.npz` path under the combine layout.

    Handles either a direct file path, or `parent/<session_subdir>/<basename>` when
    neural-aligned batches use session partitioning.

    Args:
        path: nominal trials `.npz` path.
        session_id: session id used to derive the session subdirectory fallback.

    Returns:
        The resolved, existing `.npz` path.

    Raises:
        FileNotFoundError: if neither the direct path nor the session-subdir fallback exists.
    """
    path = Path(path).expanduser()
    rp = path.resolve()
    if rp.is_file():
        return rp
    if session_id is not None:
        candidate = rp.parent / _session_subdir_key(session_id) / rp.name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f'Compressed trials npz not found at {rp}'
        + (
            f' or {rp.parent / _session_subdir_key(session_id or "") / rp.name}'
            if session_id is not None
            else ''
        ),
    )


def load_aligned_spike_splits(
    neural_aligned_npz: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train / val / test spike tensors from `{eid}_aligned.npz`.

    Args:
        neural_aligned_npz: path to the aligned neural `.npz` bundle.

    Returns:
        Tuple `(train_spikes, val_spikes, test_spikes)`, each float32.

    Raises:
        FileNotFoundError: if `neural_aligned_npz` does not exist.
    """
    p = Path(neural_aligned_npz)
    if not p.is_file():
        raise FileNotFoundError(neural_aligned_npz)
    with np.load(p, allow_pickle=True) as d:
        return (
            np.asarray(d['train_spikes'], dtype=np.float32),
            np.asarray(d['val_spikes'], dtype=np.float32),
            np.asarray(d['test_spikes'], dtype=np.float32),
        )


def load_compressed_trials_npz(
    path: str | Path,
    session_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load `img_tokens_compressed_trials.npz` (or same layout) train/val/test trial blocks.

    Args:
        path: nominal path to the compressed trials `.npz` (resolved via
            `resolve_compressed_trials_npz_path`).
        session_id: session id used to derive the session subdirectory fallback.

    Returns:
        Tuple `(train, val, test, meta)` where each array has shape `(K_split, T, L, D)` and `meta`
        contains `path`, `keys`, and optionally `meta_json`.

    Raises:
        KeyError: if no `{split}_z_trials_time` array is present, or if only the legacy
            single-stack `z_trials_time` key is present.
    """
    path = resolve_compressed_trials_npz_path(path, session_id=session_id)
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        meta_json_raw = d['meta_json'] if 'meta_json' in keys else None
        if any(f'{s}_z_trials_time' in keys for s in ('train', 'val', 'test')):
            first_arr = None
            for s in ('train', 'val', 'test'):
                k = f'{s}_z_trials_time'
                if k in keys:
                    first_arr = np.asarray(d[k], dtype=np.float32)
                    break
            if first_arr is None:
                raise KeyError(f'{path}: no *_z_trials_time arrays')
            t0, v0, d0 = int(first_arr.shape[1]), int(first_arr.shape[2]), int(first_arr.shape[3])

            def _split_arr(split: str) -> np.ndarray:
                """Load `{split}_z_trials_time`, or an empty placeholder if absent."""
                k = f'{split}_z_trials_time'
                if k not in keys:
                    return np.empty((0, t0, v0, d0), dtype=np.float32)
                return np.asarray(d[k], dtype=np.float32)

            tr, va, te = _split_arr('train'), _split_arr('val'), _split_arr('test')
        else:
            if 'z_trials_time' in keys:
                raise KeyError(
                    f'{path}: single-stack z_trials_time is no longer supported; use combine '
                    'outputs with train_z_trials_time / val_z_trials_time / test_z_trials_time.',
                )
            raise KeyError(
                f'{path}: expected train_z_trials_time / val_z_trials_time / '
                f'test_z_trials_time, got {sorted(keys)}',
            )
    meta: dict[str, Any] = {'path': str(path.resolve()), 'keys': sorted(keys)}
    if meta_json_raw is not None:
        meta['meta_json'] = str(meta_json_raw.tolist())
    return tr, va, te, meta


def build_img_token_neural_decoding_data(
    eid: str,
    neural_aligned_npz: str | Path,
    compressed_trials_npz: str | Path,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Load aligned neural spikes + compressed PCA token trials into a decoding data dict.

    Args:
        eid: session id, used as the key of the returned dict.
        neural_aligned_npz: path to the `{eid}_aligned.npz` bundle.
        compressed_trials_npz: nominal path to the compressed trials `.npz`.
        session_id: session id used to resolve the session-subdir layout; defaults to `eid`.

    Returns:
        `{eid: {'X': [...], 'y': [...], 'setup': {}}}` (see `build_decoding_data_dict`).
    """
    sid = session_id if session_id is not None else eid
    tr_n, va_n, te_n = load_aligned_spike_splits(neural_aligned_npz)
    y_tr, y_va, y_te, _ = load_compressed_trials_npz(compressed_trials_npz, session_id=sid)
    return build_decoding_data_dict(eid, tr_n, va_n, te_n, y_tr, y_va, y_te)


def build_img_token_neural_encoding_data(
    eid: str,
    neural_aligned_npz: str | Path,
    compressed_trials_npz: str | Path,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build an encoding data dict: flattened compressed tokens as `X`, spikes as `y`.

    Args:
        eid: session id, used as the key of the returned dict.
        neural_aligned_npz: path to the `{eid}_aligned.npz` bundle.
        compressed_trials_npz: nominal path to the compressed trials `.npz`.
        session_id: session id used to resolve the session-subdir layout; defaults to `eid`.

    Returns:
        `{eid: {'X': [...], 'y': [...], 'setup': {}}}` (see `build_encoding_data_dict`).
    """
    sid = session_id if session_id is not None else eid
    tr_n, va_n, te_n = load_aligned_spike_splits(neural_aligned_npz)
    x_tr, x_va, x_te, _ = load_compressed_trials_npz(compressed_trials_npz, session_id=sid)
    return build_encoding_data_dict(eid, x_tr, x_va, x_te, tr_n, va_n, te_n)


def train_tcn_neural_to_compressed(
    data_dict: dict[str, Any],
    *,
    num_samples: int = 30,
    tune_storage_path: str | None = None,
) -> dict[str, Any]:
    """Run Ray Tune hyperparameter search, then retrain on train+val and evaluate on test.

    Args:
        data_dict: mapping `eid -> {'X': [train, val, test], 'y': [...]}`.
        num_samples: number of Ray Tune trials.
        tune_storage_path: Ray Tune experiment root directory; `None` uses Ray's default.

    Returns:
        Mapping `eid -> result dict` from `train_cnn_decoder_with_tune`.
    """
    return train_cnn_decoder_with_tune(
        data_dict,
        num_samples=num_samples,
        tune_storage_path=tune_storage_path,
    )


def compressed_pred_from_decoder_result(
    result: dict[str, Any], eid: str, n_tokens: int, n_comp: int,
) -> np.ndarray:
    """Reshape a TCN decoder's `pred` on the test set to `(K_test, T, L, n_comp)`.

    Uses denormalized `pred` from `train_cnn_decoder`; for PCA unproject you should use the same
    scale as targets — the decoder returns denormalized `y` in the original compressed-target
    units.

    Args:
        result: mapping `eid -> {'pred': ..., ...}`, as returned by
            `train_tcn_neural_to_compressed`.
        eid: session id key into `result`.
        n_tokens: number of tokens `L` used to flatten the compressed targets.
        n_comp: number of PCA components per token.

    Returns:
        Array of shape `(K, T, n_tokens, n_comp)`, float64.

    Raises:
        ValueError: if `pred`'s last dim does not equal `n_tokens * n_comp`.
    """
    pred = result[eid]['pred']  # (K, T, L*n_comp)
    k, t, le = pred.shape
    if le != n_tokens * n_comp:
        raise ValueError(f'Expected pred.shape[-1]={n_tokens * n_comp}, got {le}')
    return pred.reshape(k, t, n_tokens, n_comp).astype(np.float64)
