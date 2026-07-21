"""Assemble saved `img_tokens` / `depth_fused_z` inference batches into trial-aligned arrays.

**IBL neural-aligned video**: each dataset sample is one stereo pair = one **neural time bin**
(60 per 1 s trial). Inference may save under `depth_fused_z/` (or legacy `latents/`):
`depth_fused_z_<session_id>_<pair_idx>.npy` (**`[1, V, D]`**), `depth_fused_z_batch*.npz`
(**per dataloader step**, `z` **[B,V,D]** plus row-wise neural metadata), or, when batching full
trials, `depth_fused_z_chunk*.npz` (**`[N, T, V, D]`** per file with metadata). A session
`pair_metadata.json` is used to assemble **`[N_trials, 60, V, D]`** using `neural_trial_idx` and
`neural_bin_idx` on each pair row. Assembled outputs include `train_intervals` / `val_intervals` /
`test_intervals` (same layout as `<eid>_aligned.npz`) from each pair row's `neural_interval_sec`.

**Neural batch `.npz` split folders**: when inference saves under `<root>/train`, `<root>/val`,
`<root>/test` (e.g. IBL img_tokens), batches can be merged **one subdirectory at a time** (lower
peak RAM than globbing the entire input root once).

**Legacy** (no `pair_metadata`): globs `depth_fused_z_batch*.npy`, concatenates to **`[N, V, D]`**,
then **broadcasts** the same latent across `T` (default 60). Only `z_trials_time` is written (no
per-split keys, not the intermediate batch stack). Use only when each batch is one trial, not 60
separate pairs.

`file_prefix img_tokens`: camera tensors (`c2w_*_out`, `fxfycxcy_*_out`), sorted with the same
trial order as `z`, are written to a `combined_camera_parameters.npz` sidecar beside the trials
output (not inside the large latent file).
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class ZTrialsAssembly(NamedTuple):
    """Result of assembling per-batch / chunk inference latents into `z_trials_time`.

    Attributes:
        z_trials_time: assembled latent array, typically `[N, T, V, D]`.
        trial_split_labels: per-trial split label (`'train'`/`'val'`/`'test'`), or `None` for the
            legacy broadcast mode (no per-trial split metadata).
        meta: assembly metadata (mode, shapes, counts) suitable for `json.dumps`.
        neural_trial_idx: per-trial index into the neural aligned npz, or `None`.
        trial_session_ids: per-trial session id, or `None` when only one session is present.
        per_trial_iv: per-trial `[start, end]` interval in seconds, shape `[N, 2]`, or `None`.
        aux_trials: optional extra per-trial arrays (e.g. camera tensors) keyed by name.
    """

    z_trials_time: np.ndarray
    trial_split_labels: list[str] | None
    meta: dict[str, Any]
    neural_trial_idx: np.ndarray | None
    trial_session_ids: list[str] | None
    per_trial_iv: np.ndarray | None
    aux_trials: dict[str, np.ndarray] | None = None


_INFER_LATENTS_PARENTS = (
    'pose_mu_s_z',
    'frame_z',
    'cat_z',
    'dino_z',
    'psae_z',
    'img_tokens',
)

# saved next to `z` in `img_tokens` batch npz by inference; chunk npz uses `{key}_trials`.
IMG_TOKEN_CAM_BATCH_KEYS = (
    'c2w_target_out',
    'fxfycxcy_target_out',
    'c2w_input_out',
    'fxfycxcy_input_out',
)

# combined img_token camera tensors (same sorting / layout as z) go to this filename next to
# the trials npz.
_IMG_TOKEN_CAMERA_COMBINED_FILENAME = 'combined_camera_parameters.npz'


def _img_tokens_sidecar_cameras(
    *,
    file_prefix: str,
    aux_trials: dict[str, np.ndarray] | None,
) -> bool:
    """When True, cameras go to `combined_camera_parameters.npz` instead of the trials npz."""
    return bool(aux_trials) and file_prefix == 'img_tokens'


def _img_token_cam_chunk_trials_key(batch_key: str) -> str:
    """Chunk-npz key storing the per-trial version of `batch_key` (e.g. `<key>_trials`)."""
    return f'{batch_key}_trials'


def _batch_npy_re(file_prefix: str) -> re.Pattern:
    """Regex matching `<file_prefix>_batch<digits>.npy`."""
    return re.compile(rf'^{re.escape(file_prefix)}_batch(\d+)\.npy$')


def _batch_npz_re(file_prefix: str) -> re.Pattern:
    """Regex matching `<file_prefix>_batch<digits>.npz`."""
    return re.compile(rf'^{re.escape(file_prefix)}_batch(\d+)\.npz$')


def _pair_file_re(file_prefix: str) -> re.Pattern:
    """Regex matching `<file_prefix>_<session>_<pair_idx>.npy` (pair_idx may be unpadded)."""
    return re.compile(rf'^{re.escape(file_prefix)}_(.+)_(\d+)\.npy$')


def _chunk_file_re(file_prefix: str) -> re.Pattern:
    """Regex matching `<file_prefix>_chunk<digits>.npz`."""
    return re.compile(rf'^{re.escape(file_prefix)}_chunk(\d+)\.npz$')


_DEFAULT_SPLITS = ('train', 'val', 'test')


def _split_order(split: str) -> int:
    """Sort key for `'train'`/`'val'`/`'test'`; unknown labels sort first (-1)."""
    try:
        return _DEFAULT_SPLITS.index(split.lower())
    except ValueError:
        return -1


def _sorted_batch_files(input_dir: Path, file_prefix: str) -> list[tuple[int, Path]]:
    """Sorted `(batch_idx, path)` for `<file_prefix>_batch*.npy`, preferring known latent dirs."""
    by_idx: dict[int, list[Path]] = defaultdict(list)
    batch_re = _batch_npy_re(file_prefix)
    for p in input_dir.rglob(f'{file_prefix}_batch*.npy'):
        if not p.is_file():
            continue
        m = batch_re.match(p.name)
        if m:
            by_idx[int(m.group(1))].append(p)
    files: list[tuple[int, Path]] = []
    for idx in sorted(by_idx.keys()):
        paths = by_idx[idx]
        under_preferred = [p for p in paths if p.parent.name in _INFER_LATENTS_PARENTS]
        chosen = under_preferred[0] if under_preferred else paths[0]
        files.append((idx, chosen))
    return files


def load_latent_map_pair_files(
    input_dir: Path, file_prefix: str,
) -> dict[tuple[str, int], np.ndarray]:
    """Map `(session_id, pair_idx) -> [V, D]` from `<file_prefix>_<sid>_<pair>.npy`."""
    out: dict[tuple[str, int], np.ndarray] = {}
    batch_re = _batch_npy_re(file_prefix)
    pair_re = _pair_file_re(file_prefix)
    for p in input_dir.rglob(f'{file_prefix}_*.npy'):
        if not p.is_file() or batch_re.match(p.name):
            continue
        m = pair_re.match(p.name)
        if not m:
            continue
        sid = m.group(1)
        pidx = int(m.group(2))
        z = np.load(p)
        if z.ndim == 3 and z.shape[0] == 1:
            z = z[0]
        if z.ndim != 2:
            raise ValueError(f'{p}: expected [1,V,D] or [V,D] after load, got {z.shape}')
        out[(sid, pidx)] = np.asarray(z, dtype=np.float32)
    return out


def load_stack_batches(input_dir: Path, file_prefix: str) -> np.ndarray:
    """Concatenate legacy `<file_prefix>_batch*.npy` files into a single `[N, V, D]` array."""
    entries = _sorted_batch_files(input_dir, file_prefix)
    if not entries:
        raise FileNotFoundError(f'No files matching {file_prefix}_batch*.npy under {input_dir}')
    parts = [np.load(p) for _, p in entries]
    for i, a in enumerate(parts):
        if a.ndim != 3:
            raise ValueError(
                f'Expected rank-3 array [B, v_input, D] in batch files; '
                f'got {a.shape} from part {i}',
            )
    z = np.concatenate(parts, axis=0)
    return np.asarray(z, dtype=np.float32)


def find_depth_fused_chunk_files(
    input_dir: Path, file_prefix: str,
) -> list[tuple[int, Path]]:
    """Find sorted chunk files produced by inference for a given latent stream.

    Args:
        input_dir: root directory to search recursively.
        file_prefix: filename prefix used by inference (e.g. `'depth_fused_z'`, `'img_tokens'`).

    Returns:
        Sorted `(batch_idx, path)` pairs for `<file_prefix>_chunk*.npz` under `input_dir`.
    """
    return find_depth_fused_chunk_files_under(input_dir, file_prefix)


def find_depth_fused_chunk_files_under(
    root: Path, file_prefix: str,
) -> list[tuple[int, Path]]:
    """Sorted `(batch_idx, path)` for `<file_prefix>_chunk*.npz` under `root` (recursive)."""
    root = root.resolve()
    found: list[tuple[int, Path]] = []
    chunk_re = _chunk_file_re(file_prefix)
    for p in root.rglob(f'{file_prefix}_chunk*.npz'):
        if not p.is_file():
            continue
        m = chunk_re.match(p.name)
        if not m:
            continue
        found.append((int(m.group(1)), p))
    return sorted(found, key=lambda t: t[0])


def find_depth_fused_batch_npz_files(
    input_dir: Path, file_prefix: str,
) -> list[tuple[int, Path]]:
    """Find sorted per-step neural batch `.npz` files (not chunk files).

    Args:
        input_dir: root directory to search recursively.
        file_prefix: filename prefix used by inference (e.g. `'depth_fused_z'`, `'img_tokens'`).

    Returns:
        Sorted `(batch_idx, path)` pairs for `<file_prefix>_batch*.npz` under `input_dir`.
    """
    return find_depth_fused_batch_npz_files_under(input_dir, file_prefix)


def find_depth_fused_batch_npz_files_under(
    root: Path, file_prefix: str,
) -> list[tuple[int, Path]]:
    """Sorted `(batch_idx, path)` for neural `<file_prefix>_batch*.npz` under `root` (recurse)."""
    root = root.resolve()
    found: list[tuple[int, Path]] = []
    batch_npz_re = _batch_npz_re(file_prefix)
    for p in root.rglob(f'{file_prefix}_batch*.npz'):
        if not p.is_file():
            continue
        m = batch_npz_re.match(p.name)
        if not m:
            continue
        found.append((int(m.group(1)), p))
    return sorted(found, key=lambda t: t[0])


def _has_batch_npz_directly_under_input_root(input_dir: Path, file_prefix: str) -> bool:
    """True iff at least one `<file_prefix>_batch*.npz` is an immediate child of `input_dir`."""
    batch_npz_re = _batch_npz_re(file_prefix)
    for p in input_dir.glob(f'{file_prefix}_batch*.npz'):
        if p.is_file() and batch_npz_re.match(p.name):
            return True
    return False


def infer_split_subdirs_auto(input_dir: Path, file_prefix: str) -> bool:
    """Heuristic: use split-folder batch assembly when train dir has batches but root has none."""
    train_root = input_dir / 'train'
    if not train_root.is_dir():
        return False
    if _has_batch_npz_directly_under_input_root(input_dir, file_prefix):
        return False
    return bool(find_depth_fused_batch_npz_files_under(train_root, file_prefix))


def resolve_use_split_subdirs_batch(
    input_dir: Path, file_prefix: str, split_subdirs: bool | None,
) -> bool:
    """`split_subdirs` `None` -> auto heuristic; `True`/`False` force on/off."""
    if split_subdirs is True:
        return True
    if split_subdirs is False:
        return False
    return infer_split_subdirs_auto(input_dir, file_prefix)


def _safe_session_dirname(sid: str) -> str:
    """Sanitize a session id into a filesystem-safe directory name."""
    s = str(sid).strip()
    return s.replace('/', '_').replace('\\', '_') if s else ''


def _session_subdir_key(sid: str) -> str:
    """Filesystem subdir for this session (empty sid -> `_unknown`)."""
    s = _safe_session_dirname(sid)
    return s if s else '_unknown'


def _per_split_z_trials_kw(
    z_trials_time: np.ndarray,
    trial_split_labels: list[str],
    include_splits: str,
) -> dict[str, np.ndarray]:
    """`train/val/test_z_trials_time` per-split keys (empty split -> 0-trial placeholder)."""
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


def _per_split_kw_for_aux(
    arr: np.ndarray,
    trial_split_labels: list[str],
    include_splits: str,
    base_key: str,
) -> dict[str, np.ndarray]:
    """`train_{base_key}`, `val_{base_key}`, `test_{base_key}` (empty split -> 0 trials)."""
    if arr.ndim < 2 or not trial_split_labels:
        return {}
    trail_shape = tuple(int(x) for x in arr.shape[1:])
    include_list = [x.strip().lower() for x in include_splits.split(',') if x.strip()]
    z_split = np.array([str(s).lower() for s in trial_split_labels])
    kw: dict[str, np.ndarray] = {}
    for sp in include_list:
        if sp not in ('train', 'val', 'test'):
            continue
        mask = z_split == sp
        out_key = f'{sp}_{base_key}'
        if np.any(mask):
            kw[out_key] = np.asarray(arr[mask], dtype=np.float32)
        else:
            kw[out_key] = np.empty((0, *trail_shape), dtype=np.float32)
    return kw


def _stack_split_intervals_from_rows(
    trial_split_labels: list[str], per_trial_iv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild train/val/test interval stacks from per-trial rows (assemble_* convention)."""
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
        """Stack `rows` along a new leading axis, or return an empty `(0, 2)` array."""
        if not rows:
            return np.empty((0, 2), dtype=np.float64)
        return np.stack(rows, axis=0)

    return (
        _stack_or_empty(train_iv),
        _stack_or_empty(val_iv),
        _stack_or_empty(test_iv),
    )


def assemble_from_inference_chunks(
    *,
    chunk_files: list[tuple[int, Path]],
    include_splits: set[str],
) -> tuple[
    np.ndarray,
    list[str],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    dict[str, np.ndarray] | None,
]:
    """Merge `<prefix>_chunk*.npz` from inference into the same layout as pair-metadata combine."""
    z_rows: list[np.ndarray] = []
    split_labels: list[str] = []
    ntis: list[int] = []
    eids: list[str] = []
    intervals: list[np.ndarray] = []
    seen: set[tuple[str, str, int]] = set()
    n_bins: int | None = None
    aux_chunk_ref: frozenset[str] | None = None
    aux_rows: dict[str, list[np.ndarray]] = {}

    for _batch_idx, path in chunk_files:
        data = np.load(path, allow_pickle=True)
        try:
            z = np.asarray(data['z_trials'], dtype=np.float32)
            if z.ndim != 4:
                raise ValueError(f'{path}: z_trials expected rank 4, got {z.shape}')
            keys_in_file = frozenset(
                k for k in IMG_TOKEN_CAM_BATCH_KEYS
                if _img_token_cam_chunk_trials_key(k) in data.files
            )
            if aux_chunk_ref is None:
                aux_chunk_ref = keys_in_file
            elif keys_in_file != aux_chunk_ref:
                raise ValueError(
                    f'{path}: chunk img_token camera keys {sorted(keys_in_file)} '
                    f'!= {sorted(aux_chunk_ref)} in other chunk files',
                )
            nti = np.asarray(data['neural_trial_idx'])
            ts = np.asarray(data['trial_split'], dtype=object)
            niv = np.asarray(data['neural_interval_sec'], dtype=np.float64)
            meta_raw = data.get('meta_json')
            if meta_raw is not None:
                try:
                    eid = str(json.loads(str(meta_raw)).get('eid') or '')
                except (json.JSONDecodeError, TypeError):
                    eid = ''
            else:
                eid = ''
            if n_bins is None:
                n_bins = int(z.shape[1])
            elif int(z.shape[1]) != n_bins:
                raise ValueError(
                    f'{path}: z_trials T={z.shape[1]} inconsistent with n_bins={n_bins}',
                )
            nloc = int(z.shape[0])
            if nti.shape[0] != nloc or ts.shape[0] != nloc or niv.shape != (nloc, 2):
                raise ValueError(f'{path}: metadata shape mismatch vs z_trials {z.shape}')
            if aux_chunk_ref:
                for ck in aux_chunk_ref:
                    ck_arr = np.asarray(
                        data[_img_token_cam_chunk_trials_key(ck)], dtype=np.float32,
                    )
                    if ck_arr.shape[0] != nloc:
                        raise ValueError(
                            f'{path}: {_img_token_cam_chunk_trials_key(ck)} shape[0] '
                            f'{ck_arr.shape[0]} != z_trials shape[0] {nloc}',
                        )
                    if ck_arr.shape[1] != int(z.shape[1]):
                        raise ValueError(
                            f'{path}: {_img_token_cam_chunk_trials_key(ck)} T '
                            f'{ck_arr.shape[1]} != z_trials T {z.shape[1]}',
                        )
            for j in range(nloc):
                sp = str(ts[j]).lower()
                if sp not in include_splits:
                    continue
                tid = int(nti[j])
                key = (eid, sp, tid)
                if key in seen:
                    raise ValueError(
                        f'Duplicate trial (eid={eid!r}, split={sp!r}, neural_trial_idx={tid}) '
                        'in chunk files',
                    )
                seen.add(key)
                z_rows.append(np.asarray(z[j], dtype=np.float32))
                split_labels.append(sp)
                ntis.append(tid)
                eids.append(eid)
                intervals.append(np.asarray(niv[j], dtype=np.float64).reshape(2))
                if aux_chunk_ref:
                    for ck in aux_chunk_ref:
                        aux_rows.setdefault(ck, []).append(
                            np.asarray(
                                data[_img_token_cam_chunk_trials_key(ck)][j],
                                dtype=np.float32,
                            ),
                        )
        finally:
            data.close()

    if not z_rows:
        raise RuntimeError(
            'No trials left after include_splits filter (or empty chunk list). '
            'Check include_splits and chunk contents.',
        )

    order = sorted(
        range(len(z_rows)),
        key=lambda i: (_split_order(split_labels[i]), eids[i], ntis[i]),
    )
    z_trials_time = np.stack([z_rows[i] for i in order], axis=0)
    trial_split_labels = [split_labels[i] for i in order]
    neural_trial_idx_rows = [ntis[i] for i in order]
    trial_interval_row = [intervals[i] for i in order]
    trial_session_ids = [eids[i] for i in order]
    per_trial_iv = np.stack(trial_interval_row, axis=0)

    aux_trials: dict[str, np.ndarray] | None = None
    if aux_chunk_ref:
        aux_trials = {
            ck: np.stack([aux_rows[ck][i] for i in order], axis=0)
            for ck in sorted(aux_chunk_ref)
        }

    train_intervals, val_intervals, test_intervals = _stack_split_intervals_from_rows(
        trial_split_labels, per_trial_iv,
    )
    neural_trial_idx = np.asarray(neural_trial_idx_rows, dtype=np.int64)

    stats = {
        'mode': 'inference_chunks',
        'n_bins': int(n_bins or 0),
        'num_chunk_files': len(chunk_files),
        'num_complete_trials': len(z_rows),
        'img_token_aux_keys': sorted(aux_chunk_ref) if aux_chunk_ref else [],
    }
    return (
        z_trials_time,
        trial_split_labels,
        stats,
        train_intervals,
        val_intervals,
        test_intervals,
        neural_trial_idx,
        trial_session_ids,
        per_trial_iv,
        aux_trials,
    )


def assemble_from_inference_batch_npz(
    *,
    batch_npz_files: list[tuple[int, Path]],
    include_splits: set[str],
    default_n_bins: int,
) -> tuple[
    np.ndarray,
    list[str],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    dict[str, np.ndarray] | None,
]:
    """Merge `<prefix>_batch*.npz` (per-step neural batch files) into `[N, T, V, D]` trials.

    Two-pass, pre-allocated implementation to minimise peak RAM:

    * **Pass 1** - reads only lightweight metadata arrays (no `z`); builds an index
      `(sid, split, trial_id, bin_id) -> (file_path, row_in_file)` and collects
      interval / split / session metadata. The large `z` arrays are never loaded.
    * **Pass 2** - determines `N` complete trials, allocates `z_trials_time` once as
      `[N, T, V, D]`, then re-reads each file once (grouped by path) and writes each
      bin directly into the pre-allocated array.

    Peak RSS is approximately 1x the final array plus the `z` array of one batch file at a time.
    """
    n_bins = int(default_n_bins)
    n_files_total = len(batch_npz_files)

    # ------------------------------------------------------------------ #
    # pass 1 - metadata only; no z arrays loaded                          #
    # ------------------------------------------------------------------ #
    # index[(sid, split, trial_id)][bin_id] = (path, row_in_file)
    index: dict[tuple[str, str, int], dict[int, tuple[Path, int]]] = {}
    trial_interval_sec: dict[tuple[str, str, int], np.ndarray] = {}
    rows_with_nan_interval = 0
    _vd_probe: tuple[Path, int] | None = None  # (path, row) used to read V,D later
    batch_aux_ref: frozenset[str] | None = None

    print(
        f'[batch_npz] Pass 1/2: reading metadata from {n_files_total} file(s)  '
        f'include_splits={sorted(include_splits)}  n_bins={n_bins}',
        flush=True,
    )
    _t1_start = time.perf_counter()
    _log_every = max(1, n_files_total // 10)  # ~10 progress lines for pass 1

    for _file_i, (_batch_idx, path) in enumerate(batch_npz_files):
        data = np.load(path, allow_pickle=True)
        if 'z' not in data.files:
            data.close()
            raise KeyError(f"{path}: expected 'z' array")
        # validate shape via the stored array header (cheap for npz).
        z_shape = data['z'].shape
        if len(z_shape) != 3:
            data.close()
            raise ValueError(f'{path}: z expected rank 3 [B,V,D], got {z_shape}')
        b = int(z_shape[0])

        nti = np.asarray(data['neural_trial_idx'], dtype=np.int64).reshape(-1)
        nbi = np.asarray(data['neural_bin_idx'], dtype=np.int64).reshape(-1)
        niv = np.asarray(data['neural_interval_sec'], dtype=np.float64)
        ts = np.asarray(data['trial_split'], dtype=object)
        sid = np.asarray(data['session_id'], dtype=object)

        aux_keys_f = frozenset(k for k in IMG_TOKEN_CAM_BATCH_KEYS if k in data.files)
        if batch_aux_ref is None:
            batch_aux_ref = aux_keys_f
        elif aux_keys_f != batch_aux_ref:
            data.close()
            raise ValueError(
                f'{path}: img_token camera keys {sorted(aux_keys_f)} != '
                f'{sorted(batch_aux_ref)} in other batch npz files',
            )

        data.close()

        if (
            nti.shape[0] != b
            or nbi.shape[0] != b
            or niv.shape != (b, 2)
            or ts.shape[0] != b
            or sid.shape[0] != b
        ):
            raise ValueError(f'{path}: metadata shape mismatch vs z shape {z_shape}')

        if (_file_i + 1) % _log_every == 0 or _file_i == n_files_total - 1:
            elapsed = time.perf_counter() - _t1_start
            print(
                f'[batch_npz] Pass 1/2: {_file_i + 1}/{n_files_total} files  '
                f'partial_trials={len(index)}  {elapsed:.1f}s',
                flush=True,
            )

        for i in range(b):
            sp = str(ts[i]).lower()
            if sp not in include_splits:
                continue
            sid_i = str(sid[i])
            tid = int(nti[i])
            bid = int(nbi[i])
            tk = (sid_i, sp, tid)
            iv_row = niv[i].reshape(2)

            if np.all(np.isnan(iv_row)):
                rows_with_nan_interval += 1

            bins_map = index.setdefault(tk, {})
            if bid in bins_map:
                raise ValueError(f'Duplicate neural_bin_idx={bid} for trial {tk} (file {path})')
            bins_map[bid] = (path, i)
            if _vd_probe is None:
                _vd_probe = (path, i)

            if tk not in trial_interval_sec:
                trial_interval_sec[tk] = iv_row.copy()
            else:
                prev = trial_interval_sec[tk]
                cur = iv_row
                if np.all(np.isnan(prev)) and not np.all(np.isnan(cur)):
                    trial_interval_sec[tk] = cur.copy()
                elif not np.all(np.isnan(prev)) and not np.all(np.isnan(cur)):
                    if not np.allclose(prev, cur, rtol=0.0, atol=0.0):
                        raise ValueError(
                            f'Inconsistent neural_interval_sec for trial {tk}: {prev} vs {cur}',
                        )

    _t1_elapsed = time.perf_counter() - _t1_start
    print(
        f'[batch_npz] Pass 1/2 done in {_t1_elapsed:.1f}s  '
        f'partial_or_complete_trials={len(index)}  '
        f'rows_with_nan_interval={rows_with_nan_interval}',
        flush=True,
    )
    if rows_with_nan_interval:
        print(
            f'[batch_npz] Warning: {rows_with_nan_interval} row(s) had all-nan '
            'neural_interval_sec (interval arrays may contain nan for affected trials).',
        )

    # ------------------------------------------------------------------ #
    # determine complete trials and their sorted order                    #
    # ------------------------------------------------------------------ #
    complete_tks: list[tuple[str, str, int]] = []
    incomplete = 0
    for tk in index:
        if set(index[tk].keys()) == set(range(n_bins)):
            complete_tks.append(tk)
        else:
            incomplete += 1

    complete_tks.sort(key=lambda k: (_split_order(k[1]), k[0], k[2]))
    n = len(complete_tks)
    print(f'[batch_npz] Trials: complete={n}  incomplete_skipped={incomplete}', flush=True)

    if n == 0:
        print(f'[diag] batch_npz: complete=0  incomplete={incomplete}  n_bins_required={n_bins}')
        raise RuntimeError(
            'No complete trials from inference batch npz files (all bins 0..n_bins-1 per trial). '
            'Check include_splits and saved batch files.',
        )

    # ------------------------------------------------------------------ #
    # probe one row to learn V, D                                         #
    # ------------------------------------------------------------------ #
    assert _vd_probe is not None
    _probe_path, _probe_row = _vd_probe
    _probe_data = np.load(_probe_path, allow_pickle=True)
    try:
        _probe_z = np.asarray(_probe_data['z'])
        v_dim, d_dim = int(_probe_z.shape[1]), int(_probe_z.shape[2])
        aux_keys_list = sorted(batch_aux_ref) if batch_aux_ref else []
        aux_trailing: dict[str, tuple[int, ...]] = {}
        for ak in aux_keys_list:
            if ak not in _probe_data.files:
                raise KeyError(f'{_probe_path}: missing camera key {ak!r} after pass 1 saw it')
            a = np.asarray(_probe_data[ak])
            if a.shape[0] != _probe_z.shape[0]:
                raise ValueError(
                    f'{_probe_path}: {ak} rows {a.shape[0]} != z rows {_probe_z.shape[0]}',
                )
            aux_trailing[ak] = tuple(int(x) for x in a.shape[1:])
    finally:
        _probe_data.close()
    del _probe_z

    _nbytes = n * n_bins * v_dim * d_dim * 4  # float32
    print(
        f'[batch_npz] Allocating z_trials_time ({n}, {n_bins}, {v_dim}, {d_dim}) float32  '
        f'= {_nbytes / 1024**3:.2f} GiB',
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # pass 2 - allocate output once, fill bin-by-bin per file             #
    # ------------------------------------------------------------------ #
    # group (trial_i, bin_id) lookups by file path to open each file once.
    # file_jobs[path] = list of (trial_i, bin_id, row_in_file)
    file_jobs: dict[Path, list[tuple[int, int, int]]] = {}
    trial_split_labels: list[str] = []
    trial_session_ids: list[str] = []
    neural_trial_idx_rows: list[int] = []
    trial_interval_row: list[np.ndarray] = []

    for trial_i, tk in enumerate(complete_tks):
        trial_split_labels.append(tk[1])
        trial_session_ids.append(tk[0])
        neural_trial_idx_rows.append(tk[2])
        trial_interval_row.append(
            trial_interval_sec[tk]
            if tk in trial_interval_sec
            else np.full(2, np.nan, dtype=np.float64),
        )
        for bid in range(n_bins):
            fpath, row = index[tk][bid]
            file_jobs.setdefault(fpath, []).append((trial_i, bid, row))

    # free the index now — no longer needed
    del index, trial_interval_sec

    z_trials_time = np.empty((n, n_bins, v_dim, d_dim), dtype=np.float32)
    aux_out: dict[str, np.ndarray] = {}
    for ak in aux_keys_list:
        aux_out[ak] = np.empty((n, n_bins, *aux_trailing[ak]), dtype=np.float32)

    n_files_pass2 = len(file_jobs)
    print(
        f'[batch_npz] Pass 2/2: filling z_trials_time from {n_files_pass2} unique file(s) …',
        flush=True,
    )
    _t2_start = time.perf_counter()
    _log_every2 = max(1, n_files_pass2 // 10)

    for _fi, (fpath, jobs) in enumerate(file_jobs.items()):
        data = np.load(fpath, allow_pickle=True)
        try:
            z_file = np.asarray(data['z'], dtype=np.float32)  # [B, V, D] for this file
            aux_files = {ak: np.asarray(data[ak], dtype=np.float32) for ak in aux_keys_list}
        finally:
            data.close()
        for trial_i, bid, row in jobs:
            z_trials_time[trial_i, bid] = z_file[row]
            for ak in aux_keys_list:
                aux_out[ak][trial_i, bid] = aux_files[ak][row]
        del z_file
        del aux_files

        if (_fi + 1) % _log_every2 == 0 or _fi == n_files_pass2 - 1:
            elapsed2 = time.perf_counter() - _t2_start
            print(
                f'[batch_npz] Pass 2/2: {_fi + 1}/{n_files_pass2} files  {elapsed2:.1f}s',
                flush=True,
            )

    print(
        f'[batch_npz] Pass 2/2 done in {time.perf_counter() - _t2_start:.1f}s  '
        f'z_trials_time shape={z_trials_time.shape}',
        flush=True,
    )

    per_trial_iv = np.stack(
        [np.asarray(x, dtype=np.float64).reshape(2) for x in trial_interval_row], axis=0,
    )

    train_intervals, val_intervals, test_intervals = _stack_split_intervals_from_rows(
        trial_split_labels, per_trial_iv,
    )
    neural_trial_idx = np.asarray(neural_trial_idx_rows, dtype=np.int64)

    stats: dict[str, Any] = {
        'mode': 'inference_batch_npz',
        'n_bins': int(n_bins),
        'num_batch_npz_files': len(batch_npz_files),
        'num_complete_trials': n,
        'trials_skipped_incomplete_bins': incomplete,
        'rows_with_all_nan_interval': int(rows_with_nan_interval),
        'img_token_aux_keys': aux_keys_list,
    }
    aux_trials = aux_out if aux_keys_list else None
    return (
        z_trials_time,
        trial_split_labels,
        stats,
        train_intervals,
        val_intervals,
        test_intervals,
        neural_trial_idx,
        trial_session_ids,
        per_trial_iv,
        aux_trials,
    )


def _any_split_subdir_has_batch_npz(
    input_dir: Path, include_splits: set[str], file_prefix: str,
) -> bool:
    """True if any included split subdirectory (`train`/`val`/`test`) has batch npz files."""
    for sp in _DEFAULT_SPLITS:
        if sp not in include_splits:
            continue
        if find_depth_fused_batch_npz_files_under(input_dir / sp, file_prefix):
            return True
    return False


def assemble_from_inference_batch_npz_split_roots(
    *,
    input_dir: Path,
    include_splits: set[str],
    default_n_bins: int,
    file_prefix: str,
) -> tuple[
    np.ndarray,
    list[str],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    dict[str, np.ndarray] | None,
]:
    """Merge neural batch `.npz` files **one split subdirectory at a time** (`input_dir/train`, …).

    Reduces peak RAM vs scanning `input_dir` once with `rglob`: each
    `assemble_from_inference_batch_npz` call only merges batches discovered under
    `input_dir/<split>/`.
    """
    include_order = [s for s in _DEFAULT_SPLITS if s in include_splits]
    assembled: dict[str, tuple[Any, ...]] = {}
    shape_tvd: tuple[int, int, int] | None = None

    merged_files = 0
    merged_complete = 0
    merged_incomplete = 0
    merged_nan_rows = 0

    for sp in include_order:
        root = input_dir / sp
        batch_files = find_depth_fused_batch_npz_files_under(root, file_prefix)
        if not batch_files:
            continue
        tup = assemble_from_inference_batch_npz(
            batch_npz_files=batch_files,
            include_splits={sp},
            default_n_bins=default_n_bins,
        )
        assembled[sp] = tup
        st = tup[2]
        merged_files += int(st['num_batch_npz_files'])
        merged_complete += int(st['num_complete_trials'])
        merged_incomplete += int(st['trials_skipped_incomplete_bins'])
        merged_nan_rows += int(st['rows_with_all_nan_interval'])
        z0 = tup[0]
        if shape_tvd is None:
            shape_tvd = (int(z0.shape[1]), int(z0.shape[2]), int(z0.shape[3]))

    if shape_tvd is None:
        raise RuntimeError(
            'split-folder batch assembly found no neural `<prefix>_batch*.npz` under '
            f'{input_dir}/{{train,val,test}}/ (file_prefix={file_prefix!r}).',
        )

    aux_key_order: list[str] | None = None
    aux_trail: dict[str, tuple[int, ...]] = {}
    for sp in include_order:
        if sp not in assembled:
            continue
        aux_probe = assembled[sp][9]
        if aux_probe:
            aux_key_order = sorted(aux_probe.keys())
            aux_trail = {k: tuple(int(x) for x in aux_probe[k].shape[2:]) for k in aux_key_order}
            break
    aux_blocks: dict[str, list[np.ndarray]] | None = (
        {k: [] for k in aux_key_order} if aux_key_order else None
    )

    z_blocks: list[np.ndarray] = []
    trial_split_labels_out: list[str] = []
    nt_parts: list[np.ndarray] = []
    iv_parts: list[np.ndarray] = []
    sid_flat: list[str] = []

    t_bins = int(shape_tvd[0])
    for sp in include_order:
        if sp not in assembled:
            z_blocks.append(np.empty((0, *shape_tvd), dtype=np.float32))
            nt_parts.append(np.empty((0,), dtype=np.int64))
            iv_parts.append(np.empty((0, 2), dtype=np.float64))
            if aux_blocks is not None:
                assert aux_key_order is not None
                for k in aux_key_order:
                    aux_blocks[k].append(np.empty((0, t_bins, *aux_trail[k]), dtype=np.float32))
            continue
        (
            z_tr,
            labs,
            _st,
            _ti_tr,
            _ti_va,
            _ti_te,
            nt,
            sid_list,
            piv,
            aux_part,
        ) = assembled[sp]
        if aux_blocks is not None:
            if not aux_part:
                raise ValueError(
                    f'split-folder merge: split {sp!r} has z trials but no img_token camera '
                    f'arrays; expected keys {aux_key_order}',
                )
            if sorted(aux_part.keys()) != aux_key_order:
                raise ValueError(
                    f'split-folder merge: split {sp!r} camera keys {sorted(aux_part.keys())} '
                    f'!= {aux_key_order}',
                )
            for k in aux_key_order:
                a = np.asarray(aux_part[k], dtype=np.float32)
                if tuple(int(x) for x in a.shape[2:]) != aux_trail[k]:
                    raise ValueError(
                        f'split-folder merge: {k!r} trailing dims {a.shape[2:]} != {aux_trail[k]}',
                    )
                aux_blocks[k].append(a)
        elif aux_part:
            raise ValueError(
                f'split-folder merge: split {sp!r} has camera arrays but no earlier split '
                'established a common camera key set (unexpected).',
            )
        z_blocks.append(np.asarray(z_tr, dtype=np.float32))
        trial_split_labels_out.extend(labs)
        nt_parts.append(np.asarray(nt, dtype=np.int64))
        iv_parts.append(np.asarray(piv, dtype=np.float64))
        sid_flat.extend(list(sid_list))

    z_trials_time = np.concatenate(z_blocks, axis=0)
    neural_trial_idx = np.concatenate(nt_parts, axis=0)
    per_trial_iv = np.concatenate(iv_parts, axis=0)

    train_iv, val_iv, test_iv = _stack_split_intervals_from_rows(
        trial_split_labels_out, per_trial_iv,
    )

    aux_merged: dict[str, np.ndarray] | None = None
    if aux_blocks is not None and aux_key_order is not None:
        aux_merged = {k: np.concatenate(aux_blocks[k], axis=0) for k in aux_key_order}

    stats_merged: dict[str, Any] = {
        'mode': 'inference_batch_npz_split_subdirs',
        'n_bins': int(default_n_bins),
        'num_batch_npz_files': merged_files,
        'num_complete_trials': merged_complete,
        'trials_skipped_incomplete_bins': merged_incomplete,
        'rows_with_all_nan_interval': merged_nan_rows,
        'img_token_aux_keys': list(aux_key_order) if aux_key_order else [],
    }

    return (
        z_trials_time,
        trial_split_labels_out,
        stats_merged,
        train_iv,
        val_iv,
        test_iv,
        neural_trial_idx,
        sid_flat,
        per_trial_iv,
        aux_merged,
    )


def expand_time_per_view(z_views: np.ndarray, n_time_bins: int) -> np.ndarray:
    """`[N, V, D]` -> `[N, T, V, D]` broadcast (same latent every bin)."""
    if z_views.ndim != 3:
        raise ValueError(f'expand_time_per_view expects [N, V, D], got {z_views.shape}')
    n, v, d = z_views.shape
    out = np.broadcast_to(z_views[:, np.newaxis, :, :], (n, n_time_bins, v, d))
    return np.asarray(out, dtype=np.float32)


def assemble_from_pair_metadata(
    *,
    pair_metadata_path: Path,
    latent_map: dict[tuple[str, int], np.ndarray],
    include_splits: set[str],
    session_id: str | None,
    n_bins: int,
) -> tuple[
    np.ndarray,
    list[str],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
]:
    """Build z_trials_time [N, T, V, D] and per-split interval arrays from neural_interval_sec."""
    raw = json.loads(pair_metadata_path.read_text(encoding='utf-8'))
    eid = str(raw.get('eid') or '')
    if session_id:
        eid = session_id
    neural_align = raw.get('neural_alignment') or {}
    n_bins = int(neural_align.get('neural_bins_per_trial', n_bins))

    print(f'[diag] pair_metadata eid={eid!r}  n_bins={n_bins}')
    print(f'[diag] latent_map has {len(latent_map)} entries')
    if latent_map:
        sample_keys = list(latent_map.keys())[:3]
        print(f'[diag] latent_map sample keys: {sample_keys}')
    else:
        print('[diag] latent_map is EMPTY — no depth_fused_z_<sid>_<pair>.npy files were found')

    pairs: list[dict[str, Any]] = raw.get('pairs') or []
    print(f'[diag] pair_metadata has {len(pairs)} pair rows total')
    trial_bins: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    trial_interval_sec: dict[tuple[str, int], np.ndarray] = {}
    vdim = ddim = None
    pair_rows_used = 0

    for row in pairs:
        split = str(row.get('split', '')).lower()
        if split not in include_splits:
            continue
        if 'neural_trial_idx' not in row or 'neural_bin_idx' not in row:
            continue
        pair_idx = int(row['pair_idx'])
        key = (eid, pair_idx)
        z = latent_map.get(key)
        if z is None:
            continue
        if vdim is None:
            vdim, ddim = z.shape
        elif z.shape != (vdim, ddim):
            raise ValueError(
                f'Inconsistent latent shape at pair_idx={pair_idx}: {z.shape} vs {(vdim, ddim)}',
            )
        tid = int(row['neural_trial_idx'])
        bid = int(row['neural_bin_idx'])
        tk = (split, tid)
        if tk not in trial_bins:
            trial_bins[tk] = {}
        if bid in trial_bins[tk]:
            raise ValueError(f'Duplicate neural_bin_idx={bid} for trial {tk}')
        trial_bins[tk][bid] = z
        pair_rows_used += 1
        iv_raw = row.get('neural_interval_sec')
        if iv_raw is not None:
            iv = np.asarray(iv_raw, dtype=np.float64).reshape(2)
            if tk in trial_interval_sec:
                if not np.allclose(trial_interval_sec[tk], iv):
                    raise ValueError(
                        f'Inconsistent neural_interval_sec for trial {tk}: '
                        f'{trial_interval_sec[tk]} vs {iv}',
                    )
            else:
                trial_interval_sec[tk] = iv

    pairs_eligible = sum(
        1 for row in pairs
        if str(row.get('split', '')).lower() in include_splits
        and 'neural_trial_idx' in row
        and 'neural_bin_idx' in row
    )
    print(f'[diag] pairs eligible (in split + has neural fields): {pairs_eligible}')
    print(f'[diag] pairs matched a latent (pair_rows_used): {pair_rows_used}')
    print(f'[diag] unique partial/complete trials in trial_bins: {len(trial_bins)}')

    trials_blocks: list[np.ndarray] = []
    trial_split_labels: list[str] = []
    trial_interval_row: list[np.ndarray] = []
    neural_trial_idx_rows: list[int] = []
    incomplete = 0
    for tk in sorted(trial_bins.keys(), key=lambda k: (_split_order(k[0]), k[1])):
        bins_map = trial_bins[tk]
        if set(bins_map.keys()) != set(range(n_bins)):
            incomplete += 1
            continue
        block = np.stack([bins_map[b] for b in range(n_bins)], axis=0)
        trials_blocks.append(block)
        trial_split_labels.append(tk[0])
        trial_interval_row.append(
            trial_interval_sec[tk]
            if tk in trial_interval_sec
            else np.full(2, np.nan, dtype=np.float64),
        )
        neural_trial_idx_rows.append(tk[1])

    if not trials_blocks:
        print(f'[diag] complete=0  incomplete={incomplete}  n_bins_required={n_bins}')
        if trial_bins:
            print(f'[diag] All {len(trial_bins)} trial(s) are INCOMPLETE — missing bins:')
            for tk, bins_map in list(trial_bins.items())[:5]:
                missing = sorted(set(range(n_bins)) - set(bins_map.keys()))
                print(
                    f'[diag]   trial {tk}: has bins {sorted(bins_map.keys())}, missing {missing}',
                )
        else:
            print('[diag] trial_bins is EMPTY — no pair row matched a latent.')
            print(f'[diag] Likely cause: eid in pair_metadata ({eid!r}) does not match')
            print('[diag] the session-id embedded in latent filenames (see sample keys above).')
        raise RuntimeError(
            'No complete trials (all bins present). Check latents, pair_metadata, and session id.',
        )

    z_trials_time = np.stack(trials_blocks, axis=0)
    per_trial_iv = np.stack(
        [np.asarray(x, dtype=np.float64).reshape(2) for x in trial_interval_row], axis=0,
    )
    train_intervals, val_intervals, test_intervals = _stack_split_intervals_from_rows(
        trial_split_labels, per_trial_iv,
    )
    trial_session_ids = [eid] * len(trial_split_labels)

    stats = {
        'mode': 'pair_metadata',
        'eid': eid,
        'n_bins': n_bins,
        'num_complete_trials': len(trials_blocks),
        'trials_skipped_incomplete_bins': incomplete,
        'num_pair_files_used': pair_rows_used,
    }
    neural_trial_idx = np.asarray(neural_trial_idx_rows, dtype=np.int64)
    return (
        z_trials_time,
        trial_split_labels,
        stats,
        train_intervals,
        val_intervals,
        test_intervals,
        neural_trial_idx,
        trial_session_ids,
        per_trial_iv,
    )


def _is_valid_trials_npz(path: Path) -> bool:
    """True if `path` is a readable trials npz containing a `z_trials_time`-style array."""
    try:
        d = np.load(path, allow_pickle=True)
        if 'z_trials_time' in d.files:
            return True
        return any(f'{s}_z_trials_time' in d.files for s in ('train', 'val', 'test'))
    except OSError:
        return False


def _latent_intermediates_present_local(latents_parent: Path, file_prefix: str) -> bool:
    """True if per-batch inference artifacts remain under `latents_parent`.

    E.g. an unmerged `depth_concat_z/` directory.
    """
    for pattern in (
        f'{file_prefix}_chunk*.npz',
        f'{file_prefix}_batch*.npz',
        f'{file_prefix}_batch*.npy',
    ):
        if any(latents_parent.rglob(pattern)):
            return True
    batch_prefix = f'{file_prefix}_batch'
    for p in latents_parent.rglob(f'{file_prefix}_*.npy'):
        if p.is_file() and not p.name.startswith(batch_prefix):
            return True
    return False


def maybe_check_against_neural(
    z: np.ndarray,
    neural_npz: Path,
    *,
    trial_splits: list[str] | None,
) -> None:
    """Sanity-check assembled latents against a neural `<eid>_aligned.npz` bundle.

    Compares `z_trials_time` per split to `train_spikes` / `val_spikes` / `test_spikes` in the
    bundle: trial counts and time-bin counts must match per split.

    Args:
        z: assembled latents, shape `[N, T, V, D]`.
        neural_npz: path to the `<eid>_aligned.npz` bundle.
        trial_splits: per-trial split labels aligned with `z`'s first axis, or `None` to skip the
            check (e.g. legacy combine mode with no per-trial split labels).

    Raises:
        ValueError: if `z` is not rank 4, or if trial/time-bin counts disagree with the neural
            bundle for any split.
        KeyError: if an expected `{split}_spikes` key is missing from `neural_npz`.
    """
    if trial_splits is None:
        print('[neural check] skipped: no per-trial split labels (legacy combine mode).')
        return
    if z.ndim != 4:
        raise ValueError(f'z_trials_time expected rank 4 [N,T,V,D], got {z.shape}')
    data = np.load(neural_npz)
    z_split = np.array([str(s).lower() for s in trial_splits])
    pairs = (('train', 'train_spikes'), ('val', 'val_spikes'), ('test', 'test_spikes'))
    for split, key in pairs:
        if key not in data.files:
            raise KeyError(f'{key} not in {neural_npz}; keys: {list(data.files)}')
        neu = np.asarray(data[key])
        if neu.ndim != 3:
            raise ValueError(f'{key} expected rank 3 [trials, T, units], got {neu.shape}')
        n_trial_neu, n_t_neu, n_units = neu.shape
        mask = z_split == split
        n_trial_z = int(mask.sum())
        if n_trial_z == 0:
            if n_trial_neu != 0:
                raise ValueError(
                    f'Split {split!r}: combined z has 0 trials but {key} has {n_trial_neu}',
                )
            print(f'[neural check] OK: {split} — 0 trials in z and {key} empty.')
            continue
        z_sel = z[mask]
        n_t_z, n_v, n_d = int(z_sel.shape[1]), int(z_sel.shape[2]), int(z_sel.shape[3])
        if n_trial_z != n_trial_neu:
            raise ValueError(
                f'Split {split!r}: trial count mismatch — '
                f'z has {n_trial_z}, {key} has {n_trial_neu}',
            )
        if n_t_z != n_t_neu:
            raise ValueError(
                f'Split {split!r}: time bin mismatch — z has {n_t_z}, {key} has {n_t_neu}',
            )
        print(
            f'[neural check] OK: {split} trials={n_trial_z} time_bins={n_t_z} '
            f'(neural n_units={n_units}; z views={n_v} num_latents={n_d})',
        )


def assemble_z_trials_time_from_inference_batches(
    *,
    input_dir: Path,
    pair_metadata: Path | None = None,
    session_id: str | None = None,
    include_splits: str = 'train,val,test',
    time_bins: int = 60,
    file_prefix: str = 'depth_fused_z',
    split_subdirs: bool | None = None,
) -> ZTrialsAssembly:
    """Merge inference intermediates under `input_dir` into a `ZTrialsAssembly`.

    Tries, in order: chunk files (`<file_prefix>_chunk*.npz`), split-subdirectory neural batch
    files, tree-wide neural batch files, pair-metadata-driven per-pair `.npy` files, and finally
    the legacy stacked-batch broadcast mode.

    Args:
        input_dir: inference run root to search.
        pair_metadata: session `pair_metadata.json`, used only when no batch/chunk files are found.
        session_id: override session id used when joining latents to `pair_metadata`.
        include_splits: comma-separated splits to include (e.g. `'train,val,test'`).
        time_bins: bins per trial; used for `default_n_bins` in batch assembly and for the legacy
            broadcast fallback.
        file_prefix: filename prefix matching inference output (e.g. `'depth_fused_z'`,
            `'img_tokens'`).
        split_subdirs: `True`/`False` to force split-subdirectory batch discovery on/off, or `None`
            for the auto heuristic (see `resolve_use_split_subdirs_batch`).

    Returns:
        A `ZTrialsAssembly` with the merged latents and associated per-trial metadata.

    Raises:
        NotADirectoryError: if `input_dir` does not exist.
    """
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    include_set = {s.strip().lower() for s in include_splits.split(',') if s.strip()}
    chunk_files = find_depth_fused_chunk_files(input_dir, file_prefix)

    use_split_roots = (
        not chunk_files
        and resolve_use_split_subdirs_batch(input_dir, file_prefix, split_subdirs)
        and _any_split_subdir_has_batch_npz(input_dir, include_set, file_prefix)
    )

    if chunk_files:
        (
            z_trials_time,
            trial_split_labels,
            chunk_stats,
            _train_iv,
            _val_iv,
            _test_iv,
            neural_trial_idx,
            trial_session_ids,
            per_trial_iv,
            aux_trials,
        ) = assemble_from_inference_chunks(
            chunk_files=chunk_files,
            include_splits=include_set,
        )
        meta = {
            'source_dir': str(input_dir),
            'mode': 'inference_chunks',
            'file_prefix': file_prefix,
            'z_trials_time_shape': list(z_trials_time.shape),
            'z_trials_time_layout': 'N_complete_trials, T_time_bins, V_input_views, num_latents',
            **chunk_stats,
        }
    elif use_split_roots:
        (
            z_trials_time,
            trial_split_labels,
            bn_stats,
            _train_iv_bn,
            _val_iv_bn,
            _test_iv_bn,
            neural_trial_idx,
            trial_session_ids,
            per_trial_iv,
            aux_trials,
        ) = assemble_from_inference_batch_npz_split_roots(
            input_dir=input_dir,
            include_splits=include_set,
            default_n_bins=time_bins,
            file_prefix=file_prefix,
        )
        meta = {
            'source_dir': str(input_dir),
            'mode': 'inference_batch_npz',
            'file_prefix': file_prefix,
            'z_trials_time_shape': list(z_trials_time.shape),
            'z_trials_time_layout': 'N_complete_trials, T_time_bins, V_input_views, num_latents',
            **bn_stats,
        }
    elif batch_npz_files := find_depth_fused_batch_npz_files(input_dir, file_prefix):
        if len(include_set) == 1:
            sole = next(iter(include_set))
            scoped = find_depth_fused_batch_npz_files_under(input_dir / sole, file_prefix)
            if scoped:
                batch_npz_files = scoped
        (
            z_trials_time,
            trial_split_labels,
            bn_stats,
            _train_iv_bn,
            _val_iv_bn,
            _test_iv_bn,
            neural_trial_idx,
            trial_session_ids,
            per_trial_iv,
            aux_trials,
        ) = assemble_from_inference_batch_npz(
            batch_npz_files=batch_npz_files,
            include_splits=include_set,
            default_n_bins=time_bins,
        )
        meta = {
            'source_dir': str(input_dir),
            'mode': 'inference_batch_npz',
            'file_prefix': file_prefix,
            'z_trials_time_shape': list(z_trials_time.shape),
            'z_trials_time_layout': 'N_complete_trials, T_time_bins, V_input_views, num_latents',
            **bn_stats,
        }
    elif pair_metadata is not None:
        pm = pair_metadata.resolve()
        if not pm.is_file():
            raise FileNotFoundError(pm)
        latent_map = load_latent_map_pair_files(input_dir, file_prefix)
        if not latent_map:
            raise FileNotFoundError(
                f'No {file_prefix}_<session>_<pair>.npy under {input_dir} '
                '(run inference on neural-aligned precache so each pair is saved).',
            )
        (
            z_trials_time,
            trial_split_labels,
            pm_stats,
            _train_iv_pm,
            _val_iv_pm,
            _test_iv_pm,
            neural_trial_idx,
            trial_session_ids,
            per_trial_iv,
        ) = assemble_from_pair_metadata(
            pair_metadata_path=pm,
            latent_map=latent_map,
            include_splits=include_set,
            session_id=session_id,
            n_bins=time_bins,
        )
        meta = {
            'source_dir': str(input_dir),
            'pair_metadata': str(pm),
            'mode': 'pair_metadata',
            'file_prefix': file_prefix,
            'z_trials_time_shape': list(z_trials_time.shape),
            'z_trials_time_layout': 'N_complete_trials, T_time_bins, V_input_views, num_latents',
            **pm_stats,
        }
        aux_trials = None
    else:
        z_batches = load_stack_batches(input_dir, file_prefix)
        z_trials_time = expand_time_per_view(z_batches, time_bins)
        trial_split_labels = None
        neural_trial_idx = None
        trial_session_ids = None
        per_trial_iv = None
        aux_trials = None
        meta = {
            'source_dir': str(input_dir),
            'mode': 'legacy_batch',
            'file_prefix': file_prefix,
            'z_trials_time_shape': list(z_trials_time.shape),
            'z_trials_time_layout': 'N_batches, T_bins (broadcast), V_input_views, num_latents',
            'time_bins': time_bins,
            'legacy_batch_stack_shape': list(z_batches.shape),
        }

    return ZTrialsAssembly(
        z_trials_time=z_trials_time,
        trial_split_labels=trial_split_labels,
        meta=meta,
        neural_trial_idx=neural_trial_idx,
        trial_session_ids=trial_session_ids,
        per_trial_iv=per_trial_iv,
        aux_trials=aux_trials,
    )


def _merge_aux_trials_into_save_kw(
    save_kw: dict[str, Any],
    *,
    aux_trials: dict[str, np.ndarray] | None,
    trial_split_labels: list[str] | None,
    include_splits: str,
    z_storage_uses_per_split_keys: bool,
) -> None:
    """Add `c2w_*_out` / … arrays; mirror `train_*_z_trials_time` layout when `split_kw` used."""
    if not aux_trials:
        return
    if trial_split_labels is None:
        for k, arr in aux_trials.items():
            save_kw[k] = np.asarray(arr, dtype=np.float32)
        return
    if z_storage_uses_per_split_keys:
        for k, arr in aux_trials.items():
            save_kw.update(_per_split_kw_for_aux(arr, trial_split_labels, include_splits, k))
    else:
        for k, arr in aux_trials.items():
            save_kw[k] = np.asarray(arr, dtype=np.float32)


def _write_camera_parameters_npz_sidecar(
    *,
    out_path_camera: Path,
    meta_cam: dict[str, Any],
    trial_split_labels: list[str] | None,
    include_splits: str,
    train_intervals: np.ndarray | None,
    val_intervals: np.ndarray | None,
    test_intervals: np.ndarray | None,
    neural_trial_idx: np.ndarray | None,
    aux_trials: dict[str, np.ndarray],
    z_storage_uses_per_split_keys: bool,
) -> None:
    """Write img_token camera arrays (same merge/sort as z) to a separate small `.npz`."""
    cam_kw: dict[str, Any] = {'meta_json': json.dumps(meta_cam)}
    if trial_split_labels is not None:
        cam_kw['trial_split'] = np.array(trial_split_labels, dtype=object)
    if train_intervals is not None and val_intervals is not None and test_intervals is not None:
        cam_kw['train_intervals'] = train_intervals
        cam_kw['val_intervals'] = val_intervals
        cam_kw['test_intervals'] = test_intervals
        cam_kw['neural_trial_idx'] = neural_trial_idx
    _merge_aux_trials_into_save_kw(
        cam_kw,
        aux_trials=aux_trials,
        trial_split_labels=trial_split_labels,
        include_splits=include_splits,
        z_storage_uses_per_split_keys=z_storage_uses_per_split_keys,
    )
    out_path_camera.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path_camera, **cam_kw)
    print(f'Wrote cameras (same sort/layout as trials z) → {out_path_camera}')


def _split_camera_sidecar_basename(split_name: str) -> str:
    """Sidecar filename for `img_tokens` when writing one trials NPZ per split (OOM-safe path)."""
    return f'combined_camera_parameters_{split_name}.npz'


def _write_flat_trials_npz_from_assembly(
    *,
    output: Path,
    trial_fname_for_meta: str,
    z_trials_time: np.ndarray,
    trial_split_labels: list[str] | None,
    meta: dict[str, Any],
    neural_trial_idx: np.ndarray | None,
    per_trial_iv: np.ndarray | None,
    aux_trials: dict[str, np.ndarray] | None,
    include_splits: str,
    file_prefix: str,
    img_tokens_camera_sidecar_basename: str | None = None,
) -> tuple[list[Path], np.ndarray]:
    """Write a single flat trials `.npz` (no session subdirs) plus an optional camera sidecar."""
    train_intervals = val_intervals = test_intervals = None
    if (
        trial_split_labels is not None
        and per_trial_iv is not None
        and neural_trial_idx is not None
    ):
        train_intervals, val_intervals, test_intervals = _stack_split_intervals_from_rows(
            trial_split_labels, per_trial_iv,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    split_kw: dict[str, np.ndarray] = {}
    if trial_split_labels is not None:
        split_kw = _per_split_z_trials_kw(z_trials_time, trial_split_labels, include_splits)
    sidecar = _img_tokens_sidecar_cameras(file_prefix=file_prefix, aux_trials=aux_trials)
    meta_latent = dict(meta)
    cam_basename = img_tokens_camera_sidecar_basename or _IMG_TOKEN_CAMERA_COMBINED_FILENAME
    if sidecar:
        meta_latent['camera_parameters_npz_filename'] = cam_basename
        meta_latent['camera_npz_note'] = (
            f'Camera arrays are in {cam_basename} (same trial order and '
            'train_/val_/test_* layout as z).'
        )
    save_kw: dict[str, Any] = {'meta_json': json.dumps(meta_latent)}
    if split_kw:
        save_kw.update(split_kw)
    else:
        save_kw['z_trials_time'] = z_trials_time
    if trial_split_labels is not None:
        save_kw['trial_split'] = np.array(trial_split_labels, dtype=object)
    if train_intervals is not None:
        save_kw['train_intervals'] = train_intervals
        save_kw['val_intervals'] = val_intervals
        save_kw['test_intervals'] = test_intervals
        save_kw['neural_trial_idx'] = neural_trial_idx
    if not sidecar:
        _merge_aux_trials_into_save_kw(
            save_kw,
            aux_trials=aux_trials,
            trial_split_labels=trial_split_labels,
            include_splits=include_splits,
            z_storage_uses_per_split_keys=bool(split_kw),
        )
    try:
        approx = sum(int(np.asarray(v).nbytes) for v in save_kw.values())
    except (TypeError, ValueError, AttributeError):
        approx = 0
    print(
        '[combine] np.savez_compressed starting (zlib — large payloads can take many minutes '
        f'with no further messages until finished). → {output}'
        + (f'  ~{approx / (1024**3):.2f} GiB raw array bytes in payload' if approx else ''),
        flush=True,
    )
    np.savez_compressed(output, **save_kw)
    written: list[Path] = [output.resolve()]
    if sidecar and aux_trials is not None:
        meta_cam = {
            **meta,
            'camera_parameters_file': cam_basename,
            'companion_trials_npz_filename': trial_fname_for_meta,
        }
        cam_out = output.parent / cam_basename
        _write_camera_parameters_npz_sidecar(
            out_path_camera=cam_out,
            meta_cam=meta_cam,
            trial_split_labels=trial_split_labels,
            include_splits=include_splits,
            train_intervals=train_intervals,
            val_intervals=val_intervals,
            test_intervals=test_intervals,
            neural_trial_idx=neural_trial_idx,
            aux_trials=aux_trials,
            z_storage_uses_per_split_keys=bool(split_kw),
        )
        written.append(cam_out.resolve())
    print(f'Wrote {output}')
    if split_kw:
        for k in sorted(split_kw.keys()):
            print(f'  {k}  {split_kw[k].shape}')
    else:
        print(f'  z_trials_time  {z_trials_time.shape}')
    return written, z_trials_time
