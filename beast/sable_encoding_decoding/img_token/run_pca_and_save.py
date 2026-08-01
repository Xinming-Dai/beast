"""Fit PCA on train img-token latents and save compressed trials + PCA bundle.

Load batched ``img_tokens`` tensors from inference (see
``assemble_z_trials_time_from_inference_batches``), typically under ``…/img_tokens`` or a
folder populated after merging shards elsewhere. Fit PCA on the **train** trials only
(normalize over train trials/time), apply the same norm + PCA transform to val/test, then
save:

1. ``save_pca_and_norm`` artifact (portable PCA arrays + pickled sklearn ``PCA``, train-session
   norm, train-only ``x_compressed``).
2. Compressed trials ``.npz`` (per-split ``*_z_trials_time``, ``trial_split``, intervals,
   ``neural_trial_idx``) with PCA-compressed last dimension.

To merge inference batches into aligned trial tensors in a separate step, run
``trials_assembly.py`` first and point ``--input-dir`` at the same tree that script read from
(or your combined layout as supported by assembly).

The pipeline supports incremental execution via ``--stage``:

- ``--stage 1``: Fit PCA on train only; write ``--output-pca-npz`` + a train-only
  ``--output-trials-npz`` (``train_z_trials_time`` only). Run this when val/test tokens are
  not yet available.
- ``--stage 2``: Load PCA + train norm from ``--output-pca-npz``, copy ``train_z_trials_time``
  from the stage-1 ``--output-trials-npz`` (no re-assembly from disk), project val/test tokens,
  and write the final ``--output-trials-npz``. Optionally load val/test from
  ``--combined-trials-val-npz`` / ``--combined-trials-test-npz`` instead of assembling from
  ``--input-dir``.
- ``--stage all`` (default): Run both stages in a single pass (original behaviour).

You may pass uncompressed trials ``.npz`` files produced by ``trials_assembly.py``
(``--output-train`` / ``--output-val`` / ``--output-test`` per-split mode) via
``--combined-trials-train-npz``, ``--combined-trials-val-npz``, ``--combined-trials-test-npz``
to skip assembling those splits from inference batches under ``--input-dir`` (fewer scans /
RAM). Omit ``--input-dir`` only when every split required for the chosen ``--stage`` has a
corresponding combined NPZ and you use explicit outputs or ``--model-root``.

If ``--output-pca-npz`` and ``--output-trials-npz`` are both omitted, they default to::

    ``<MODEL_ROOT>/img_tokens_compressed/<session_name>/img_tokens_pca_joint.npz``
    ``<MODEL_ROOT>/img_tokens_compressed/<session_name>/img_tokens_compressed_trials.npz``

where ``MODEL_ROOT`` is set explicitly via ``--model-root``, or inferred as the parent of
``--input-dir`` when its last path segment is ``img_tokens``. When ``--input-dir`` is omitted,
you must pass ``--model-root`` for default output paths.

When ``--input-dir`` is given, each session is fit **independently**: ``--session-names`` selects
which ``<input-dir>/<session_name>/`` subfolders to process (space-separated), or, when omitted,
every immediate subfolder of ``--input-dir`` is auto-discovered and processed. Each session gets
its own PCA basis fit on that session's train split only — sessions are never pooled together.

Example — two-pass (explicit outputs or use ``--model-root`` + omit ``--output-*``)::

    # Stage 1: train only (val/test tokens not yet extracted); processes every session found
    # under $MODEL_ROOT/img_tokens/ unless --session-names restricts it
    python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \\
      --input-dir $MODEL_ROOT/img_tokens \\
      --session-names <EID1> <EID2> \\
      --model-root $MODEL_ROOT \\
      --n-feat-keep 3 --stage 1

    # Stage 2: project val/test once those tokens exist
    python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \\
      --input-dir $MODEL_ROOT/img_tokens \\
      --session-names <EID1> <EID2> \\
      --model-root $MODEL_ROOT \\
      --n-feat-keep 3 --stage 2

Example — single pass, auto-discovering every session under ``--input-dir``::

    python -m beast.sable_encoding_decoding.img_token.run_pca_and_save \\
      --input-dir .../eval_results_.../img_tokens \\
      --model-root ...

Loading the saved PCA bundle later::

    import pickle
    import numpy as np

    d = np.load('img_tokens_pca_joint.npz', allow_pickle=True)
    pca = pickle.loads(d['pca_sklearn_pickle'][0])
"""

import argparse
import json
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from beast.sable_encoding_decoding.img_token.pca_compression import (
    feature_pca_fit_on_train,
    feature_pca_project_train_normalized,
    save_pca_and_norm,
)
from beast.sable_encoding_decoding.img_token.trials_assembly import (
    IMG_TOKEN_CAM_BATCH_KEYS,
    ZTrialsAssembly,
    _session_subdir_key,
    assemble_z_trials_time_from_inference_batches,
    find_depth_fused_batch_npz_files,
    find_depth_fused_chunk_files,
    maybe_check_against_neural,
)


def _log(msg: str, *, enabled: bool) -> None:
    """Print `msg` when `enabled`."""
    if enabled:
        print(msg, flush=True)


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string (e.g. ``'1.5 MiB'``)."""
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024.0:
            return f'{n:.1f} {unit}'
        n /= 1024.0
    return f'{n:.1f} PiB'


def _per_split_z_trials_kw(
    z_trials_time: np.ndarray,
    trial_split_labels: list[str],
    include_splits: str,
) -> dict[str, np.ndarray]:
    """Per-split ``*_z_trials_time`` arrays (empty split → 0 trials)."""
    if z_trials_time.ndim != 4 or not trial_split_labels:
        return {}
    t = int(z_trials_time.shape[1])
    v = int(z_trials_time.shape[2])
    d = int(z_trials_time.shape[3])
    include_list = [x.strip().lower() for x in include_splits.split(',') if x.strip()]
    z_split = np.array([str(s).lower() for s in trial_split_labels])
    kw: dict[str, np.ndarray] = {}
    for sp in include_list:
        if sp not in ('train', 'val', 'test'):
            continue
        mask = z_split == sp
        key = f'{sp}_z_trials_time'
        if np.any(mask):
            kw[key] = np.asarray(z_trials_time[mask], dtype=np.float32)
        else:
            kw[key] = np.empty((0, t, v, d), dtype=np.float32)
    return kw


def _stack_split_intervals_from_rows(
    trial_split_labels: list[str], per_trial_iv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild train/val/test interval stacks from per-trial interval rows."""
    train_iv: list[np.ndarray] = []
    val_iv: list[np.ndarray] = []
    test_iv: list[np.ndarray] = []
    for lab, iv in zip(trial_split_labels, per_trial_iv, strict=True):
        sp = str(lab).lower()
        row = np.asarray(iv, dtype=np.float64).reshape(2)
        if sp == 'train':
            train_iv.append(row)
        elif sp == 'val':
            val_iv.append(row)
        elif sp == 'test':
            test_iv.append(row)
        else:
            raise ValueError(f'Unexpected trial split label {lab!r} (expected train/val/test)')

    def _stack_or_empty(rows: list[np.ndarray]) -> np.ndarray:
        if not rows:
            return np.empty((0, 2), dtype=np.float64)
        return np.stack(rows, axis=0)

    return (
        _stack_or_empty(train_iv),
        _stack_or_empty(val_iv),
        _stack_or_empty(test_iv),
    )


def _per_trial_iv_from_saved_combined_npz(data: Any) -> np.ndarray:
    """Rebuild per-trial ``(N, 2)`` intervals from ``*_intervals`` + ``trial_split``."""
    ts = [str(x).lower() for x in np.asarray(data['trial_split'], dtype=object)]
    nt = np.asarray(data['neural_trial_idx'], dtype=np.int64)
    tr_iv = np.asarray(data['train_intervals'], dtype=np.float64).reshape(-1, 2)
    va_iv = np.asarray(data['val_intervals'], dtype=np.float64).reshape(-1, 2)
    te_iv = np.asarray(data['test_intervals'], dtype=np.float64).reshape(-1, 2)
    out = np.empty((len(ts), 2), dtype=np.float64)
    for i, sp in enumerate(ts):
        j = int(nt[i])
        if sp == 'train':
            out[i] = tr_iv[j]
        elif sp == 'val':
            out[i] = va_iv[j]
        elif sp == 'test':
            out[i] = te_iv[j]
        else:
            raise ValueError(f'Unexpected split {sp!r} in trial_split')
    return out


def assembly_from_combined_trials_npz(path: Path) -> ZTrialsAssembly:
    """Load an uncompressed trials ``.npz`` produced by `trials_assembly` (same schema as disk).

    Args:
        path: path to the combined trials ``.npz`` file.

    Returns:
        A `ZTrialsAssembly` reconstructed from the saved arrays.

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if no non-empty ``*_z_trials_time`` entries are found, or if
            ``trial_split`` / ``neural_trial_idx`` lengths mismatch the ``z_trials_time`` rows.
        KeyError: if required keys are missing from the saved ``.npz``.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    d = np.load(path, allow_pickle=True)
    meta_raw = d.get('meta_json')
    meta: dict[str, Any] = {}
    if meta_raw is not None:
        try:
            meta = json.loads(str(meta_raw))
        except (json.JSONDecodeError, TypeError):
            meta = {}

    if any(f'{s}_z_trials_time' in d.files for s in ('train', 'val', 'test')):
        blocks: list[np.ndarray] = []
        labels: list[str] = []
        for sp in ('train', 'val', 'test'):
            key = f'{sp}_z_trials_time'
            if key not in d.files:
                continue
            zi = np.asarray(d[key], dtype=np.float32)
            if zi.shape[0] == 0:
                continue
            blocks.append(zi)
            labels.extend([sp] * int(zi.shape[0]))
        if not blocks:
            raise ValueError(f'No non-empty *_z_trials_time entries in {path}')
        z_trials_time = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]
    elif 'z_trials_time' in d.files:
        z_trials_time = np.asarray(d['z_trials_time'], dtype=np.float32)
    else:
        raise KeyError(
            f'Expected {{train,val,test}}_z_trials_time or z_trials_time in {path}; got {d.files}',
        )

    if 'trial_split' not in d.files or 'neural_trial_idx' not in d.files:
        raise KeyError(f'Expected trial_split and neural_trial_idx in combined trials npz {path}')

    trial_split_labels = [str(x).lower() for x in np.asarray(d['trial_split'], dtype=object)]
    neural_trial_idx = np.asarray(d['neural_trial_idx'], dtype=np.int64)
    if (
        len(trial_split_labels) != z_trials_time.shape[0]
        or neural_trial_idx.shape[0] != z_trials_time.shape[0]
    ):
        raise ValueError(
            f'trial_split / neural_trial_idx length mismatch vs z in {path}: '
            f'z={z_trials_time.shape[0]} trial_split={len(trial_split_labels)} '
            f'nt={neural_trial_idx.shape[0]}',
        )

    per_trial_iv = _per_trial_iv_from_saved_combined_npz(d)

    return ZTrialsAssembly(
        z_trials_time=z_trials_time,
        trial_split_labels=trial_split_labels,
        meta=meta,
        neural_trial_idx=neural_trial_idx,
        trial_session_ids=None,
        per_trial_iv=per_trial_iv,
        aux_trials=None,
    )


def input_dir_required_for_stage(
    *,
    stage: str,
    combined_train: Path | None,
    combined_val: Path | None,
    combined_test: Path | None,
) -> bool:
    """Return True when batch-tree ``--input-dir`` is still needed given combined-trials paths."""
    if stage == '1':
        return combined_train is None
    if stage == '2':
        return combined_val is None or combined_test is None
    assert stage == 'all'
    return combined_train is None or combined_val is None or combined_test is None


# --------------------------------------------------------------------------- #
# Stage helpers
# --------------------------------------------------------------------------- #


def _load_pca_bundle(
    pca_npz: Path,
) -> tuple[PCA, np.ndarray, np.ndarray, int, int]:
    """Load a sklearn PCA object and norm stats from a ``save_pca_and_norm`` file.

    Prefers the pickled estimator (``pca_sklearn_pickle``) when present; otherwise
    reconstructs from ``pca_mean`` / ``pca_components`` / ``pca_explained_variance``.

    Args:
        pca_npz: path to the ``save_pca_and_norm`` output ``.npz``.

    Returns:
        Tuple ``(pca, x_session_avg, x_session_std, D_ref, k_comp)``.

    Raises:
        FileNotFoundError: if `pca_npz` does not exist.
        TypeError: if the pickled estimator does not deserialize to a sklearn `PCA`.
        ValueError: if stored dimensions are internally inconsistent.
    """
    pca_npz = pca_npz.resolve()
    if not pca_npz.is_file():
        raise FileNotFoundError(
            f'[stage2] PCA bundle not found: {pca_npz}\n'
            'Run --stage 1 first to produce this file.',
        )
    data = np.load(pca_npz, allow_pickle=True)
    if 'pca_sklearn_pickle' in data.files:
        raw = data['pca_sklearn_pickle']
        blob = raw.item() if raw.ndim == 0 else raw[0]
        pca = pickle.loads(blob)
        if not isinstance(pca, PCA):
            raise TypeError(
                f'[stage2] pca_sklearn_pickle deserialized to {type(pca)}, expected sklearn PCA',
            )
    else:
        pca = PCA()
        pca.mean_ = data['pca_mean'].astype(np.float64)
        pca.components_ = data['pca_components'].astype(np.float64)
        n_comp = int(pca.components_.shape[0])
        pca.n_components_ = n_comp
        pca.n_features_in_ = int(pca.components_.shape[1])
        # sklearn>=1.5: PCA.transform() passes explained_variance_ to get_namespace before
        # projecting.
        if 'pca_explained_variance' in data.files:
            ev = np.asarray(data['pca_explained_variance'], dtype=np.float64).reshape(-1)
            if ev.shape[0] != n_comp:
                raise ValueError(
                    f'[stage2] pca_explained_variance length {ev.shape[0]} != '
                    f'n_components {n_comp}',
                )
            pca.explained_variance_ = ev
        else:
            # older bundles omitted this; placeholder is OK when whiten=False (default)
            pca.explained_variance_ = np.ones(n_comp, dtype=np.float64)
    x_session_avg = np.asarray(data['x_session_avg'], dtype=np.float32)
    x_session_std = np.asarray(data['x_session_std'], dtype=np.float32)
    d_ref = int(data['original_feature_dim_D'])
    k_comp = int(data['n_pca_components_saved'])
    nfi = getattr(pca, 'n_features_in_', None)
    if nfi is not None and int(nfi) != d_ref:
        raise ValueError(
            f'[stage2] original_feature_dim_D={d_ref} != PCA n_features_in_={int(nfi)}',
        )
    k_pca = int(pca.components_.shape[0])
    if k_pca != k_comp:
        raise ValueError(
            f'[stage2] n_pca_components_saved={k_comp} != PCA components axis 0 size {k_pca}',
        )
    return pca, x_session_avg, x_session_std, d_ref, k_comp


def _resolve_trials_npz_path(trials_npz: Path, session_id: str | None) -> Path:
    """Return the actual path of the trials npz, handling session subdirectory layout."""
    p = trials_npz.resolve()
    if p.is_file():
        return p
    if session_id is not None:
        candidate = p.parent / _session_subdir_key(session_id) / p.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'[stage2] Stage-1 trials npz not found at {p}'
        + (
            f' or {p.parent / _session_subdir_key(session_id) / p.name}'
            if session_id is not None
            else ''
        )
        + '\nRun --stage 1 first to produce this file.',
    )


def _load_stage1_train_compressed(
    trials_npz: Path,
    session_id: str | None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str] | None, dict]:
    """Load the train-compressed data written by stage 1.

    Args:
        trials_npz: stage-1 ``--output-trials-npz`` path (or basename under a session subdir).
        session_id: session id used to resolve session-subdirectory layout, if any.

    Returns:
        Tuple ``(z_train, split_labels_train, nt_train, iv_train, sids_train, meta)`` where:

        - ``z_train``: ``(N_train, T, L, k)`` float32
        - ``split_labels_train``: list of ``'train'`` strings, length N_train
        - ``nt_train``: ``(N_train,)`` int64 neural_trial_idx
        - ``iv_train``: ``(N_train, 2)`` float64 per-trial intervals
        - ``sids_train``: ``[session_id] * N_train`` or ``None``
        - ``meta``: dict parsed from ``meta_json``

    Raises:
        KeyError: if ``train_z_trials_time`` is missing from the saved file.
        ValueError: if the ``train`` row count disagrees between ``train_z_trials_time`` and
            ``trial_split``.
    """
    path = _resolve_trials_npz_path(trials_npz, session_id)
    data = np.load(path, allow_pickle=True)

    if 'train_z_trials_time' not in data:
        raise KeyError(
            f"[stage2] 'train_z_trials_time' not found in {path}. "
            'Make sure stage 1 completed successfully.',
        )

    z_train = np.asarray(data['train_z_trials_time'], dtype=np.float32)
    n_train = z_train.shape[0]

    # neural_trial_idx is stored for all splits together; filter to train rows
    neural_trial_idx_all = np.asarray(data['neural_trial_idx'], dtype=np.int64)
    trial_split_all = list(data['trial_split'])
    train_mask = np.array([str(s).lower() == 'train' for s in trial_split_all])

    if train_mask.sum() != n_train:
        raise ValueError(
            f'[stage2] Mismatch: train_z_trials_time has {n_train} rows but '
            f"{train_mask.sum()} 'train' entries in trial_split array in {path}.",
        )

    nt_train = neural_trial_idx_all[train_mask]
    iv_train = np.asarray(data['train_intervals'], dtype=np.float64)

    if iv_train.shape != (n_train, 2):
        raise ValueError(
            f'[stage2] Expected train_intervals shape ({n_train}, 2), got {iv_train.shape}.',
        )

    split_labels_train = ['train'] * n_train

    sids_train: list[str] | None = None
    if session_id is not None:
        sids_train = [session_id] * n_train

    meta_raw = data.get('meta_json')
    try:
        meta = json.loads(str(meta_raw)) if meta_raw is not None else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}

    return z_train, split_labels_train, nt_train, iv_train, sids_train, meta


def _write_compressed_trials_npz(
    *,
    assembly: ZTrialsAssembly,
    z_compressed: np.ndarray,
    trials_output: Path,
    meta_extra: dict[str, Any],
    log: Callable[[str], None],
    include_splits: str,
) -> tuple[list[Path], np.ndarray | None]:
    """Write PCA-compressed trials: per-split ``*_z_trials_time``, intervals, trial idx."""
    trials_output = trials_output.resolve()
    latents_parent = trials_output.parent
    trial_fname = trials_output.name

    z_ct = np.asarray(z_compressed, dtype=np.float32)
    trial_split_labels = assembly.trial_split_labels
    neural_trial_idx = assembly.neural_trial_idx
    trial_session_ids = assembly.trial_session_ids
    per_trial_iv = assembly.per_trial_iv

    if trial_split_labels is None:
        raise RuntimeError('trial_split_labels missing; need neural-aligned inference batches.')
    if neural_trial_idx is None or per_trial_iv is None:
        raise RuntimeError('neural_trial_idx / per_trial_iv missing; check inference export.')

    meta_base = {**assembly.meta, **meta_extra}

    written: list[Path] = []
    z_for_neural_check: np.ndarray | None = None

    log(
        f'[write] Writing compressed trials npz under {latents_parent}  '
        f'(basename {trial_fname!r})  z_compressed ~{_fmt_bytes(int(z_ct.nbytes))}',
    )
    t_write0 = time.perf_counter()

    if trial_session_ids is None:
        latents_parent.mkdir(parents=True, exist_ok=True)
        log('[write] mode: single file (no session subdirs)')
        split_kw = _per_split_z_trials_kw(z_ct, trial_split_labels, include_splits)
        flat_kw: dict[str, Any] = {
            **split_kw,
            'meta_json': json.dumps(meta_base),
            'trial_split': np.array(trial_split_labels, dtype=object),
            'neural_trial_idx': np.asarray(neural_trial_idx, dtype=np.int64),
        }
        tr_iv, va_iv, te_iv = _stack_split_intervals_from_rows(trial_split_labels, per_trial_iv)
        flat_kw['train_intervals'] = tr_iv
        flat_kw['val_intervals'] = va_iv
        flat_kw['test_intervals'] = te_iv
        np.savez_compressed(trials_output, **flat_kw)
        written.append(trials_output.resolve())
        log(f'[write] Wrote {trials_output} in {time.perf_counter() - t_write0:.2f}s')
        for k in sorted(split_kw.keys()):
            log(f'  {k} {split_kw[k].shape}')
        z_for_neural_check = z_ct
    else:
        sub_keys = sorted(
            {_session_subdir_key(s) for s in trial_session_ids},
            key=lambda k: (k == '_unknown', k),
        )
        log(f'[write] mode: {len(sub_keys)} session partition(s): {sub_keys}')
        for i_sk, sk in enumerate(sub_keys):
            indices = [
                i for i, s in enumerate(trial_session_ids) if _session_subdir_key(s) == sk
            ]
            idx = np.asarray(indices, dtype=np.int64)
            z_part = np.asarray(z_ct[idx], dtype=np.float32)
            splits_part = [trial_split_labels[i] for i in indices]
            nt_part = np.asarray(neural_trial_idx[idx], dtype=np.int64)
            iv_part = np.asarray(per_trial_iv[idx], dtype=np.float64)

            tr_iv, va_iv, te_iv = _stack_split_intervals_from_rows(splits_part, iv_part)
            skw = _per_split_z_trials_kw(z_part, splits_part, include_splits)
            n_tot = sum(int(skw[k].shape[0]) for k in skw)
            meta_part = {
                **meta_base,
                'partition_session_subdir': sk,
                'z_trials_time_shape': [n_tot] + list(z_part.shape[1:]),
                'num_complete_trials': n_tot,
                'z_trials_time_layout': 'N_trials, T_bins, L_tokens, n_pca_components',
                'z_storage': 'per_split_train_val_test_z_trials_time',
            }
            out_path = latents_parent / sk / trial_fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            t_part = time.perf_counter()
            np.savez_compressed(
                out_path,
                **skw,
                meta_json=json.dumps(meta_part),
                trial_split=np.array(splits_part, dtype=object),
                train_intervals=tr_iv,
                val_intervals=va_iv,
                test_intervals=te_iv,
                neural_trial_idx=nt_part,
            )
            written.append(out_path.resolve())
            shapes = ' '.join(f'{k}={tuple(skw[k].shape)}' for k in sorted(skw.keys()))
            log(
                f'[write]   ({i_sk + 1}/{len(sub_keys)}) {out_path}  '
                f'{time.perf_counter() - t_part:.2f}s  {shapes}  '
                f'intervals train/val/test {tr_iv.shape}/{va_iv.shape}/{te_iv.shape}',
            )

        if len(sub_keys) == 1:
            z_for_neural_check = z_ct
        log(f'[write] All compressed trial files done in {time.perf_counter() - t_write0:.2f}s')

    return written, z_for_neural_check


# --------------------------------------------------------------------------- #
# Camera sidecar helpers
# --------------------------------------------------------------------------- #

_CAMERA_SIDECAR_NAME = 'img_tokens_camera_parameters.npz'


def _camera_sidecar_path(trials_npz: Path, session_id: str | None = None) -> Path:
    """Derive the camera sidecar path alongside ``trials_npz``.

    When `session_id` is given, nests under `<parent>/<session_subdir>/` to match the
    per-session partitioning `_write_compressed_trials_npz` applies to the trials npz itself.
    """
    parent = trials_npz.resolve().parent
    if session_id is not None:
        parent = parent / _session_subdir_key(session_id)
    return parent / _CAMERA_SIDECAR_NAME


def _write_camera_sidecar(
    path: Path,
    cameras_by_split: dict[str, dict[str, np.ndarray]],
    trial_split_labels: list[str],
    neural_trial_idx: np.ndarray,
    per_trial_iv: np.ndarray,
    meta: dict[str, Any],
    log: Callable[[str], None],
) -> None:
    """Write sorted+combined per-split camera arrays alongside the compressed trials npz.

    ``cameras_by_split`` maps split name → dict of camera key → ``(N_split, ...)`` array.
    Writes ``{split}_{key}`` keys (e.g. ``train_c2w_target_out``) mirroring the schema used
    by the camera sidecar writer in `trials_assembly.py`. Arrays are stored uncompressed
    (``np.savez``) — same order as the trials npz, no lossy transform applied.
    """
    if not cameras_by_split:
        return
    cam_kw: dict[str, Any] = {'meta_json': json.dumps({**meta, 'camera_sidecar': True})}
    cam_kw['trial_split'] = np.array(trial_split_labels, dtype=object)
    cam_kw['neural_trial_idx'] = np.asarray(neural_trial_idx, dtype=np.int64)
    tr_iv, va_iv, te_iv = _stack_split_intervals_from_rows(trial_split_labels, per_trial_iv)
    cam_kw['train_intervals'] = tr_iv
    cam_kw['val_intervals'] = va_iv
    cam_kw['test_intervals'] = te_iv
    for split, cam_dict in cameras_by_split.items():
        for key, arr in cam_dict.items():
            cam_kw[f'{split}_{key}'] = np.asarray(arr, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **cam_kw)
    keys_written = [f'{sp}_{k}' for sp, cd in cameras_by_split.items() for k in cd]
    log(f'[cameras] Wrote camera sidecar → {path}  keys: {keys_written}')


def _load_stage1_train_cameras(
    camera_sidecar_path: Path,
    log: Callable[[str], None],
) -> dict[str, np.ndarray] | None:
    """Load ``train_{key}`` arrays from the stage-1 camera sidecar.

    Returns a dict ``{key: arr}`` (without the ``train_`` prefix) or ``None`` if the file
    does not exist or contains no train camera keys.
    """
    p = camera_sidecar_path.resolve()
    if not p.is_file():
        log(
            f'[cameras] Stage-1 camera sidecar not found at {p}; '
            'skipping train cameras in stage 2.',
        )
        return None
    d = np.load(p, allow_pickle=True)
    train_cams: dict[str, np.ndarray] = {}
    for key in IMG_TOKEN_CAM_BATCH_KEYS:
        full_key = f'train_{key}'
        if full_key in d.files:
            train_cams[key] = np.asarray(d[full_key], dtype=np.float32)
    if not train_cams:
        log(f'[cameras] No train_* camera keys found in {p}; skipping train cameras in stage 2.')
        return None
    log(
        f'[cameras] Loaded train cameras from stage-1 sidecar {p}  '
        f'keys: {list(train_cams.keys())}',
    )
    return train_cams


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #


def run_img_tokens_pca_joint(
    *,
    input_dir: Path | None,
    pair_metadata: Path | None,
    session_id: str | None,
    include_splits: str,
    time_bins: int,
    file_prefix: str,
    output_pca_npz: Path,
    output_trials_npz: Path,
    n_feat_keep: int,
    random_state: int,
    neural_npz: Path | None,
    verbose: bool = True,
    split_subdirs: bool | None = None,
    stage: str = 'all',
    combined_trials_train_npz: Path | None = None,
    combined_trials_val_npz: Path | None = None,
    combined_trials_test_npz: Path | None = None,
) -> tuple[list[Path], Path]:
    """Fit (or apply) PCA on img-token trials and write the compressed trials + PCA bundle.

    Args:
        input_dir: inference directory scanned for batch/chunk ``.npz`` files, or ``None``
            when every needed split is supplied via ``combined_trials_*_npz``.
        pair_metadata: optional ``pair_metadata.json`` for neural-aligned precache mode.
        session_id: override session id for pair-metadata mode / session-subdir resolution.
        include_splits: comma-separated splits to include (e.g. ``'train,val,test'``).
        time_bins: bins per trial for neural batch assembly.
        file_prefix: inference filename prefix.
        output_pca_npz: output path for the `save_pca_and_norm` bundle.
        output_trials_npz: output (or stage-1 input) path for the compressed trials `.npz`.
        n_feat_keep: number of PCA components to keep.
        random_state: PCA RNG seed.
        neural_npz: optional `*_aligned.npz` to verify trial/time alignment against.
        verbose: whether to print progress logs.
        split_subdirs: forces per-split root discovery on/off; `None` means `'auto'`.
        stage: `'1'`, `'2'`, or `'all'`.
        combined_trials_train_npz: pre-combined train trials `.npz`, skips disk assembly.
        combined_trials_val_npz: pre-combined val trials `.npz`, skips disk assembly.
        combined_trials_test_npz: pre-combined test trials `.npz`, skips disk assembly.

    Returns:
        Tuple `(written_trial_paths, output_pca_npz)`.

    Raises:
        ValueError: if `--include-splits` omits `'train'` outside of stage `'1'`, or if
            feature dimensions mismatch across splits.
        RuntimeError: if required inputs (e.g. `--input-dir`, `neural_trial_idx`) are missing
            for the requested stage.
    """
    def log(m: str) -> None:
        _log(m, enabled=verbose)

    include_list = [s.strip().lower() for s in include_splits.split(',') if s.strip()]

    # ------------------------------------------------------------------ #
    # Stage 2: load PCA + train data from stage-1 outputs, skip train asm #
    # ------------------------------------------------------------------ #
    if stage == '2':
        log(f'[stage2] Loading PCA bundle from {output_pca_npz} …')
        pca_model, x_session_avg, x_session_std, d_ref, k_comp = _load_pca_bundle(
            output_pca_npz.resolve(),
        )
        log(
            f'[stage2] PCA loaded: D={d_ref}  k_comp={k_comp}  '
            f'components shape={pca_model.components_.shape}',
        )

        log('[stage2] Loading train compressed data from stage-1 trials npz …')
        z_train, split_labels_train, nt_train, iv_train, sids_train, assembly_meta_ref = (
            _load_stage1_train_compressed(output_trials_npz, session_id)
        )
        log(f'[stage2] Train data loaded: z_train={z_train.shape}  N_train={z_train.shape[0]}')

        compressed_chunks: list[np.ndarray] = [z_train]
        trial_split_labels_out: list[str] = list(split_labels_train)
        neural_idx_chunks: list[np.ndarray] = [nt_train]
        iv_chunks: list[np.ndarray] = [iv_train]
        sid_flat: list[str] | None = list(sids_train) if sids_train is not None else None

        # load train cameras from stage-1 sidecar (may be None if not saved)
        cam_sidecar_path = _camera_sidecar_path(output_trials_npz, session_id)
        train_cams = _load_stage1_train_cameras(cam_sidecar_path, log)
        cameras_by_split: dict[str, dict[str, np.ndarray]] = {}
        if train_cams is not None:
            cameras_by_split['train'] = train_cams

        # project val and test splits
        for split in ('val', 'test'):
            if split not in include_list:
                continue
            ct_path = combined_trials_val_npz if split == 'val' else combined_trials_test_npz
            if ct_path is not None:
                log(f'[stage2] Loading split={split!r} from combined trials NPZ {ct_path} …')
                t_asm = time.perf_counter()
                asm = assembly_from_combined_trials_npz(ct_path.resolve())
                log(
                    f'[stage2] split={split!r} loaded in {time.perf_counter() - t_asm:.2f}s  '
                    f'z_trials_time shape={tuple(asm.z_trials_time.shape)}',
                )
            else:
                if input_dir is None:
                    raise RuntimeError(
                        f'[stage2] Provide --combined-trials-{split}-npz or --input-dir to '
                        f'assemble {split!r} from inference batches.',
                    )
                log(f'[stage2] Assembling split={split!r} from disk …')
                t_asm = time.perf_counter()
                asm = assemble_z_trials_time_from_inference_batches(
                    input_dir=input_dir,
                    pair_metadata=pair_metadata,
                    session_id=session_id,
                    include_splits=split,
                    time_bins=time_bins,
                    file_prefix=file_prefix,
                    split_subdirs=split_subdirs,
                )
                log(
                    f'[stage2] split={split!r} done in {time.perf_counter() - t_asm:.2f}s  '
                    f'z_trials_time shape={tuple(asm.z_trials_time.shape)}',
                )
            if asm.trial_split_labels is None:
                raise RuntimeError(
                    f'[stage2] split={split!r}: trial_split_labels missing; '
                    'need neural-aligned batches or chunk files.',
                )
            z = np.asarray(asm.z_trials_time, dtype=np.float32)
            _, _, _, d = z.shape
            if int(d) != d_ref:
                raise ValueError(
                    f'[stage2] Feature dim mismatch for split={split!r}: '
                    f'expected D={d_ref}, got D={int(d)}',
                )
            x_cp = feature_pca_project_train_normalized(
                z,
                x_session_avg,
                x_session_std,
                pca_model,
                n_feat_keep=n_feat_keep,
            )
            compressed_chunks.append(x_cp)
            n_rows = int(z.shape[0])
            trial_split_labels_out.extend([split] * n_rows)

            nt_part = asm.neural_trial_idx
            iv_part = asm.per_trial_iv
            if nt_part is None or iv_part is None:
                raise RuntimeError(
                    f'[stage2] neural_trial_idx / per_trial_iv missing for split={split!r}.',
                )
            neural_idx_chunks.append(np.asarray(nt_part, dtype=np.int64))
            iv_chunks.append(np.asarray(iv_part, dtype=np.float64))

            sid_src = asm.trial_session_ids
            if sid_src is not None:
                if sid_flat is None:
                    sid_flat = []
                sid_flat.extend(list(sid_src))

            if asm.aux_trials:
                cameras_by_split[split] = {
                    k: np.asarray(v, dtype=np.float32) for k, v in asm.aux_trials.items()
                }

            del z, asm

        z_compressed = np.concatenate(compressed_chunks, axis=0)
        neural_trial_idx = np.concatenate(neural_idx_chunks, axis=0)
        per_trial_iv = np.concatenate(iv_chunks, axis=0)

        assembly = ZTrialsAssembly(
            z_trials_time=z_compressed.astype(np.float32),
            trial_split_labels=trial_split_labels_out,
            meta=assembly_meta_ref,
            neural_trial_idx=neural_trial_idx,
            trial_session_ids=sid_flat,
            per_trial_iv=per_trial_iv,
        )

        meta_extra = {
            'pipeline': 'img_tokens_pca_stage2_apply',
            'pca_npz_path': str(output_pca_npz.resolve()),
            'original_feature_dim_D': int(d_ref),
            'n_pca_components': int(k_comp),
            'z_trials_time_note': 'last_dim is PCA components (not raw D)',
            'stage': '2',
        }

        written, z_chk = _write_compressed_trials_npz(
            assembly=assembly,
            z_compressed=z_compressed,
            trials_output=output_trials_npz,
            meta_extra=meta_extra,
            log=log,
            include_splits=include_splits,
        )

        if cameras_by_split:
            _write_camera_sidecar(
                cam_sidecar_path,
                cameras_by_split,
                trial_split_labels=trial_split_labels_out,
                neural_trial_idx=neural_trial_idx,
                per_trial_iv=per_trial_iv,
                meta=assembly_meta_ref,
                log=log,
            )

        if neural_npz is not None:
            log('[neural-check] Running trial/time alignment vs *_aligned.npz …')
            if z_chk is not None:
                maybe_check_against_neural(
                    z_chk,
                    neural_npz.resolve(),
                    trial_splits=assembly.trial_split_labels,
                )
            else:
                log(
                    '[neural check] skipped: multiple session outputs '
                    '(trial/time counts should be checked per file).',
                )

        paths_str = ', '.join(str(p) for p in written)
        log(f'[done] Stage 2 finished. Compressed trial outputs ({len(written)}): {paths_str}')
        return written, output_pca_npz.resolve()

    # ------------------------------------------------------------------ #
    # Stage 1 or all: must include train for PCA fitting
    # ------------------------------------------------------------------ #
    if stage == '1':
        include_list = ['train']
        log('[stage1] Forcing include_splits=train (PCA fitting on train only).')
    elif 'train' not in include_list:
        raise ValueError("--include-splits must include 'train' for PCA fitting.")

    splits_to_run = [s for s in ('train', 'val', 'test') if s in include_list]

    ct_map: dict[str, Path | None] = {
        'train': combined_trials_train_npz,
        'val': combined_trials_val_npz,
        'test': combined_trials_test_npz,
    }
    needs_disk_assembly = any(ct_map[sp] is None for sp in splits_to_run)
    if needs_disk_assembly:
        if input_dir is None:
            raise RuntimeError(
                '--input-dir is required when any split in '
                f'{splits_to_run!r} is not supplied via --combined-trials-*-npz.',
            )
        in_dir = input_dir.resolve()
        chunk_files = find_depth_fused_chunk_files(in_dir, file_prefix)
        batch_npz_files = find_depth_fused_batch_npz_files(in_dir, file_prefix)
        log(
            f'[load] input_dir={in_dir}  file_prefix={file_prefix!r}  '
            f'found chunk*.npz={len(chunk_files)}  batch*.npz={len(batch_npz_files)}'
            '  (zeros here are OK if using pair-metadata + per-pair _*.npy only)',
        )
        if pair_metadata is not None:
            log(f'[load] pair_metadata={pair_metadata.resolve()}')
    else:
        log('[load] All splits loaded from --combined-trials-*-npz (no inference-tree scan).')
        if pair_metadata is not None:
            log(f'[load] pair_metadata={pair_metadata.resolve()} (not required for combined NPZs)')

    compressed_chunks_: list[np.ndarray] = []
    trial_split_labels_out_: list[str] = []
    neural_idx_chunks_: list[np.ndarray] = []
    iv_chunks_: list[np.ndarray] = []
    sid_flat_: list[str] | None = None
    assembly_meta_ref_: dict[str, Any] | None = None
    cameras_by_split_: dict[str, dict[str, np.ndarray]] = {}

    pca_model_ = None
    x_session_avg_ = None
    x_session_std_ = None
    k_comp_ = 0
    d_ref_: int | None = None

    splits_to_run = [s for s in ('train', 'val', 'test') if s in include_list]

    for split in splits_to_run:
        ct_path = ct_map.get(split)
        if ct_path is not None:
            log(f'[load] Loading split={split!r} from combined trials NPZ {ct_path} …')
            t_asm = time.perf_counter()
            asm = assembly_from_combined_trials_npz(ct_path.resolve())
            log(
                f'[load] split={split!r} loaded in {time.perf_counter() - t_asm:.2f}s  '
                f"mode={asm.meta.get('mode')!r}  "
                f'z_trials_time shape={tuple(asm.z_trials_time.shape)}  '
                f'~{_fmt_bytes(int(asm.z_trials_time.nbytes))}',
            )
        else:
            log(f'[load] Assembling split={split!r} from disk (may take a long time) …')
            t_asm = time.perf_counter()
            assert input_dir is not None
            asm = assemble_z_trials_time_from_inference_batches(
                input_dir=input_dir,
                pair_metadata=pair_metadata,
                session_id=session_id,
                include_splits=split,
                time_bins=time_bins,
                file_prefix=file_prefix,
                split_subdirs=split_subdirs,
            )
            log(
                f'[load] split={split!r} done in {time.perf_counter() - t_asm:.2f}s  '
                f"mode={asm.meta.get('mode')!r}  "
                f'z_trials_time shape={tuple(asm.z_trials_time.shape)}  '
                f'~{_fmt_bytes(int(asm.z_trials_time.nbytes))}',
            )
        if asm.trial_split_labels is None:
            raise RuntimeError(
                'PCA pipeline requires per-trial split labels (neural-aligned batches, chunks, or '
                'pair-metadata). Legacy stacked .npy batches have no splits.',
            )
        z = np.asarray(asm.z_trials_time, dtype=np.float32)
        _, _, _, d = z.shape
        if d_ref_ is None:
            d_ref_ = int(d)
        elif int(d) != d_ref_:
            raise ValueError(
                f'Feature dim mismatch across splits: expected D={d_ref_}, got D={int(d)} '
                f'for split={split!r}',
            )
        if assembly_meta_ref_ is None:
            assembly_meta_ref_ = dict(asm.meta)

        n_rows = int(z.shape[0])
        nt_part = asm.neural_trial_idx
        iv_part = asm.per_trial_iv
        if nt_part is None or iv_part is None:
            raise RuntimeError('neural_trial_idx / per_trial_iv missing; check inference export.')
        neural_idx_chunks_.append(np.asarray(nt_part, dtype=np.int64))
        iv_chunks_.append(np.asarray(iv_part, dtype=np.float64))
        sid_src = asm.trial_session_ids
        if sid_src is not None:
            if sid_flat_ is None:
                sid_flat_ = []
            sid_flat_.extend(list(sid_src))

        if asm.aux_trials:
            cameras_by_split_[split] = {
                k: np.asarray(v, dtype=np.float32) for k, v in asm.aux_trials.items()
            }

        trial_split_labels_out_.extend([split] * n_rows)

        if split == 'train':
            if n_rows == 0:
                raise RuntimeError('Train split has 0 trials; cannot fit PCA.')
            pca_model_, x_ct, x_session_avg_, x_session_std_ = feature_pca_fit_on_train(
                z,
                n_feat_keep=n_feat_keep,
                random_state=random_state,
                verbose_progress=verbose,
            )
            k_comp_ = int(x_ct.shape[-1])
            compressed_chunks_.append(x_ct)

            output_pca_npz = output_pca_npz.resolve()
            log(f'[save] Writing PCA + norm bundle to {output_pca_npz} …')
            t_save = time.perf_counter()
            save_pca_and_norm(
                output_pca_npz,
                pca_model_,
                x_ct,
                x_session_avg_,
                x_session_std_,
                extra={
                    'original_feature_dim_D': np.int32(d_ref_),
                    'n_pca_components_saved': np.int32(k_comp_),
                },
            )
            log(
                f'[save] PCA bundle wrote in {time.perf_counter() - t_save:.2f}s  '
                f'({output_pca_npz})',
            )
        else:
            assert (
                pca_model_ is not None
                and x_session_avg_ is not None
                and x_session_std_ is not None
            )
            x_cp = feature_pca_project_train_normalized(
                z,
                x_session_avg_,
                x_session_std_,
                pca_model_,
                n_feat_keep=n_feat_keep,
            )
            compressed_chunks_.append(x_cp)

        del z, asm

    assert (
        pca_model_ is not None
        and assembly_meta_ref_ is not None
        and x_session_avg_ is not None
        and d_ref_ is not None
    )

    z_compressed_ = np.concatenate(compressed_chunks_, axis=0)
    neural_trial_idx_ = np.concatenate(neural_idx_chunks_, axis=0)
    per_trial_iv_ = np.concatenate(iv_chunks_, axis=0)

    assembly_ = ZTrialsAssembly(
        z_trials_time=z_compressed_.astype(np.float32),
        trial_split_labels=trial_split_labels_out_,
        meta=assembly_meta_ref_,
        neural_trial_idx=neural_trial_idx_,
        trial_session_ids=sid_flat_,
        per_trial_iv=per_trial_iv_,
    )

    if assembly_.trial_session_ids is not None:
        n_sess = len({_session_subdir_key(s) for s in assembly_.trial_session_ids})
        log(f'[sessions] session_ids: {n_sess} partition(s)')
        if session_id is not None:
            unexpected = {_session_subdir_key(s) for s in assembly_.trial_session_ids} - {
                _session_subdir_key(session_id),
            }
            if unexpected:
                raise ValueError(
                    f'run_img_tokens_pca_joint(session_id={session_id!r}): loaded trials '
                    f'embed unexpected session id(s) {sorted(unexpected)}; the scanned '
                    'directory may have been mis-tagged during extraction.',
                )

    stage_label = stage  # "1" or "all"
    meta_extra_ = {
        'pipeline': 'img_tokens_pca_train_fit',
        'pca_npz_path': str(output_pca_npz),
        'original_feature_dim_D': int(d_ref_),
        'n_pca_components': int(k_comp_),
        'z_trials_time_note': 'last_dim is PCA components (not raw D)',
        'stage': stage_label,
    }

    # for stage 1, only write train split into the trials npz
    write_splits = 'train' if stage == '1' else include_splits

    written_, z_chk_ = _write_compressed_trials_npz(
        assembly=assembly_,
        z_compressed=z_compressed_,
        trials_output=output_trials_npz,
        meta_extra=meta_extra_,
        log=log,
        include_splits=write_splits,
    )

    if cameras_by_split_:
        cam_sidecar = _camera_sidecar_path(output_trials_npz, session_id)
        cam_splits = (
            {'train': cameras_by_split_['train']}
            if stage == '1' and 'train' in cameras_by_split_
            else cameras_by_split_
        )
        _write_camera_sidecar(
            cam_sidecar,
            cam_splits,
            trial_split_labels=trial_split_labels_out_,
            neural_trial_idx=neural_trial_idx_,
            per_trial_iv=per_trial_iv_,
            meta=assembly_meta_ref_,
            log=log,
        )

    if stage == '1':
        log(
            '[stage1] Train-only trials npz written. '
            'Run --stage 2 once val/test img_tokens are available.',
        )

    if neural_npz is not None and stage != '1':
        log('[neural-check] Running trial/time alignment vs *_aligned.npz …')
        if z_chk_ is not None:
            maybe_check_against_neural(
                z_chk_,
                neural_npz.resolve(),
                trial_splits=assembly_.trial_split_labels,
            )
        else:
            log(
                '[neural check] skipped: multiple session outputs '
                '(trial/time counts should be checked per file).',
            )

    paths_str = ', '.join(str(p) for p in written_)
    log(
        f'[done] Finished. PCA file: {output_pca_npz}  '
        f'Compressed trial outputs ({len(written_)}): {paths_str}',
    )
    return written_, output_pca_npz


def resolve_output_npz_paths(
    *,
    input_dir: Path | None,
    model_root: Path | None,
    output_pca_npz: Path | None,
    output_trials_npz: Path | None,
) -> tuple[Path, Path]:
    """Resolve PCA + trials output paths, defaulting under ``<anchor>/img_tokens_compressed/``.

    Args:
        input_dir: `--input-dir` value, used to infer the anchor when its basename is
            `'img_tokens'` (parent becomes the anchor).
        model_root: explicit `--model-root` anchor; takes priority over `input_dir`.
        output_pca_npz: explicit `--output-pca-npz`, or `None`.
        output_trials_npz: explicit `--output-trials-npz`, or `None`.

    Returns:
        Tuple `(output_pca_npz, output_trials_npz)`, both resolved absolute paths.

    Raises:
        SystemExit: if only one of `output_pca_npz` / `output_trials_npz` is given, or if
            neither explicit paths nor an anchor (`model_root` / suitable `input_dir`) can be
            resolved.
    """
    default_dir = Path('img_tokens_compressed')
    default_pca = Path('img_tokens_pca_joint.npz')
    default_trials = Path('img_tokens_compressed_trials.npz')

    has_pca = output_pca_npz is not None
    has_trials = output_trials_npz is not None

    if has_pca ^ has_trials:
        raise SystemExit(
            'Provide both --output-pca-npz and --output-trials-npz, or omit both and set '
            "--model-root / use --input-dir named 'img_tokens' (parent becomes MODEL_ROOT).",
        )

    if has_pca and has_trials:
        return (
            Path(output_pca_npz).expanduser().resolve(),
            Path(output_trials_npz).expanduser().resolve(),
        )

    anchor: Path | None = None
    if model_root is not None:
        anchor = Path(model_root).expanduser().resolve()
    elif input_dir is not None:
        ind = Path(input_dir).expanduser().resolve()
        if ind.name == 'img_tokens':
            anchor = ind.parent
    if anchor is None:
        raise SystemExit(
            'When omitting output paths: pass --model-root <MODEL_ROOT>, or --input-dir such '
            'that its basename is img_tokens (parent directory is MODEL_ROOT).',
        )

    comp = anchor / default_dir
    return (comp / default_pca).resolve(), (comp / default_trials).resolve()


def discover_session_names(input_dir: Path) -> list[str]:
    """Sorted names of `input_dir`'s immediate subdirectories (session/EID folders).

    Args:
        input_dir: root directory expected to contain one subfolder per session, e.g.
            ``<input_dir>/<EID>/<train|val|test>/...``.

    Returns:
        Sorted list of subdirectory names.

    Raises:
        NotADirectoryError: if `input_dir` does not exist.
        RuntimeError: if `input_dir` has no subdirectories.
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    names = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    if not names:
        raise RuntimeError(
            f'No session subdirectories found under {input_dir}; expected '
            '<input_dir>/<session_name>/<train|val|test>/... or pass --session-names explicitly.',
        )
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the img-token PCA fit/apply entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help=(
            'Inference directory scanned for img_tokens_batch*.npz / chunk*.npz / pair tensors. '
            'Typically …/img_tokens. Omit only when every split needed for this run is supplied '
            'via --combined-trials-*-npz (see --stage); default output paths still require '
            '--model-root if --input-dir is omitted.'
        ),
    )
    p.add_argument(
        '--combined-trials-train-npz',
        type=Path,
        default=None,
        help=(
            'Trials .npz from trials_assembly.py (train split). '
            'Stage 1 / stage all: skip assembling train from --input-dir when set.'
        ),
    )
    p.add_argument(
        '--combined-trials-val-npz',
        type=Path,
        default=None,
        help=(
            'Combined trials .npz for val only. Stage 2 / all: skip assembling val from '
            '--input-dir when set.'
        ),
    )
    p.add_argument(
        '--combined-trials-test-npz',
        type=Path,
        default=None,
        help=(
            'Combined trials .npz for test only. Stage 2 / all: skip assembling test from '
            '--input-dir when set.'
        ),
    )
    p.add_argument(
        '--pair-metadata',
        type=Path,
        default=None,
        help='Optional pair_metadata.json (recommended for IBL neural-aligned precache).',
    )
    p.add_argument(
        '--session-names',
        nargs='+',
        metavar='SESSION',
        default=None,
        help=(
            'Session/EID names to process, space-separated. Each is fit independently: PCA is '
            'fit on that session\'s own train split and applied to that session\'s own val/test '
            '(sessions are never pooled). Defaults to auto-discovering every immediate '
            'subdirectory of --input-dir. Ignored when --input-dir is omitted.'
        ),
    )
    p.add_argument(
        '--include-splits',
        type=str,
        default='train,val,test',
        help='Comma-separated splits to include.',
    )
    p.add_argument(
        '--time-bins', type=int, default=60, help='Bins per trial for neural batch assembly.',
    )
    p.add_argument(
        '--file-prefix', type=str, default='img_tokens', help='Inference filename prefix.',
    )
    p.add_argument(
        '--model-root',
        type=Path,
        default=None,
        help=(
            'When --output-pca-npz / --output-trials-npz are omitted: anchor directory whose '
            'img_tokens_compressed/ subtree receives default PCA and trials paths.'
        ),
    )
    p.add_argument(
        '--output-pca-npz',
        type=Path,
        default=None,
        help=(
            'Output path for save_pca_and_norm .npz. Defaults with --model-root or '
            '.../img_tokens parent.'
        ),
    )
    p.add_argument(
        '--output-trials-npz',
        type=Path,
        default=None,
        help=(
            'Destination basename: PCA trials npz written under '
            '``parent / <session_subdir> / name`` when multiple sessions partition output. '
            'In stage 2 this path is also read to load the stage-1 train data. '
            'Defaults alongside --output-pca-npz under img_tokens_compressed/.'
        ),
    )
    p.add_argument('--n-feat-keep', type=int, default=3, help='PCA components to keep (cap at D).')
    p.add_argument('--random-state', type=int, default=777, help='PCA RNG seed.')
    p.add_argument(
        '--neural-npz',
        type=Path,
        default=None,
        help=(
            'Optional *_aligned.npz — verify trial counts / T vs spikes after save '
            '(skipped in stage 1).'
        ),
    )
    p.add_argument(
        '--quiet',
        action='store_true',
        help='Disable progress logs (assemble / PCA phases / writes).',
    )
    p.add_argument(
        '--split-subdirs',
        choices=('auto', 'on', 'off'),
        default='auto',
        help=(
            'Neural-batch ``.npz`` discovery: ``auto`` adjusts for train/val/test subdirectories; '
            '``on`` / ``off`` force per-split roots vs tree-wide discovery.'
        ),
    )
    p.add_argument(
        '--stage',
        choices=['1', '2', 'all'],
        default='all',
        help=(
            "Pipeline stage: '1' = fit PCA on train only and write train-only trials npz; "
            "'2' = load PCA from --output-pca-npz, copy train_z_trials_time from the existing "
            '--output-trials-npz (no re-assembly from disk), project val/test, and write the '
            "final trials npz (optional combined NPZs); 'all' = run both stages in one pass "
            '(default).'
        ),
    )
    return p.parse_args(argv)


def main() -> None:
    """Run the img-token PCA fit/apply pipeline end to end (CLI entry point)."""
    args = parse_args()
    ct_train = (
        Path(args.combined_trials_train_npz).expanduser().resolve()
        if args.combined_trials_train_npz
        else None
    )
    ct_val = (
        Path(args.combined_trials_val_npz).expanduser().resolve()
        if args.combined_trials_val_npz
        else None
    )
    ct_test = (
        Path(args.combined_trials_test_npz).expanduser().resolve()
        if args.combined_trials_test_npz
        else None
    )

    if args.input_dir is None and input_dir_required_for_stage(
        stage=args.stage,
        combined_train=ct_train,
        combined_val=ct_val,
        combined_test=ct_test,
    ):
        raise SystemExit(
            'Provide --input-dir unless every split required for --stage='
            f'{args.stage!r} is covered by --combined-trials-*-npz; '
            'when omitting --input-dir you still need --model-root or explicit '
            '--output-pca-npz / --output-trials-npz for defaults.',
        )

    out_pca, out_tri = resolve_output_npz_paths(
        input_dir=args.input_dir,
        model_root=args.model_root,
        output_pca_npz=args.output_pca_npz,
        output_trials_npz=args.output_trials_npz,
    )
    split_flag = {'auto': None, 'on': True, 'off': False}[args.split_subdirs]

    if args.input_dir is None:
        # combined-trials-only invocation: no per-session directory tree to loop over.
        run_img_tokens_pca_joint(
            input_dir=None,
            pair_metadata=args.pair_metadata,
            session_id=None,
            include_splits=args.include_splits,
            time_bins=args.time_bins,
            file_prefix=args.file_prefix,
            output_pca_npz=out_pca,
            output_trials_npz=out_tri,
            n_feat_keep=args.n_feat_keep,
            random_state=args.random_state,
            neural_npz=args.neural_npz,
            verbose=not args.quiet,
            split_subdirs=split_flag,
            stage=args.stage,
            combined_trials_train_npz=ct_train,
            combined_trials_val_npz=ct_val,
            combined_trials_test_npz=ct_test,
        )
        return

    session_names = (
        list(args.session_names)
        if args.session_names
        else discover_session_names(args.input_dir)
    )
    print(f'[sessions] Processing {len(session_names)} session(s): {session_names}', flush=True)

    for i, session_name in enumerate(session_names):
        print(
            f'[session {i + 1}/{len(session_names)}] {session_name}',
            flush=True,
        )
        run_img_tokens_pca_joint(
            input_dir=args.input_dir / session_name,
            pair_metadata=args.pair_metadata,
            session_id=session_name,
            include_splits=args.include_splits,
            time_bins=args.time_bins,
            file_prefix=args.file_prefix,
            output_pca_npz=out_pca.parent / session_name / out_pca.name,
            output_trials_npz=out_tri,
            n_feat_keep=args.n_feat_keep,
            random_state=args.random_state,
            neural_npz=args.neural_npz,
            verbose=not args.quiet,
            split_subdirs=split_flag,
            stage=args.stage,
            combined_trials_train_npz=ct_train,
            combined_trials_val_npz=ct_val,
            combined_trials_test_npz=ct_test,
        )


if __name__ == '__main__':
    main()
