"""Load neural decoding outputs on PCA-compressed img tokens, unproject to full-D tokens.

Uses ``compressed_pred_from_decoder_result`` + ``pca_unproject`` (see
``docs/ibl_encoding_decoding.md`` step 5). Writes per-trial batch npz files under
``--out-root/<eid>/<split>/`` (one directory per requested split, e.g. ``val``, ``test``)
matching inference-style batching (one trial per file).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from beast.logging import log_step
from beast.sable_encoding_decoding.img_token.neural_decoder import (
    compressed_pred_from_decoder_result,
    resolve_compressed_trials_npz_path,
)
from beast.sable_encoding_decoding.img_token.pca_compression import pca_unproject

_VALID_SPLITS = ('val', 'test')


def _load_decoding_payload(path: Path, eid: str, split: str) -> dict:
    """Load the per-eid, per-split CNN decoding payload from a neural-decoder output `.npy`.

    Supports both the unified format (`inner['cnn'][split]`) and the pre-existing flat legacy
    format (`inner['cnn']` holding `'pred'` directly), where legacy payloads are only valid for
    `split == 'test'`.

    Args:
        path: path to the `.npy` file (an object array of a dict, or a dict directly).
        eid: session id key to extract from the loaded mapping.
        split: which split's predictions to extract (`'val'` or `'test'`).

    Returns:
        The inner `'cnn'` payload dict for `eid`/`split`.

    Raises:
        KeyError: if `eid` is not present, or `split` predictions are unavailable.
    """
    arr = np.load(path, allow_pickle=True)
    data = arr.item() if getattr(arr, 'ndim', 1) == 0 else arr
    if eid not in data:
        available = list(data.keys()) if hasattr(data, 'keys') else type(data)
        raise KeyError(f'{path}: missing key {eid!r}; got {available}')
    inner = data[eid]
    if not (isinstance(inner, dict) and 'cnn' in inner):
        return inner
    cnn = inner['cnn']
    if isinstance(cnn, dict) and split in cnn:
        return cnn[split]
    if split == 'test':
        # legacy flat format: 'cnn' holds test predictions directly, no per-split nesting
        return cnn
    raise KeyError(
        f"{path}: no {split!r} predictions under 'cnn' for eid={eid!r}; "
        f"got keys {sorted(cnn.keys()) if isinstance(cnn, dict) else type(cnn)}",
    )


def _load_pca_bundle(path: Path) -> tuple:
    """Load the pickled sklearn PCA model and session normalization stats.

    Args:
        path: path to the `img_tokens_pca_joint.npz` bundle.

    Returns:
        Tuple `(pca, z_avg, z_std)`.
    """
    with np.load(path, allow_pickle=True) as d:
        pca = pickle.loads(d['pca_sklearn_pickle'][0])
        z_avg = np.asarray(d['x_session_avg'], dtype=np.float64)
        z_std = np.asarray(d['x_session_std'], dtype=np.float64)
    return pca, z_avg, z_std


def parse_neural_trial_index_csv(s: str | None) -> list[int] | None:
    """Parse a comma-separated list of neural trial ids from `--neural-trial-index`.

    Args:
        s: comma-separated ints, e.g. `'0,4,10'`, or `None`/empty.

    Returns:
        List of parsed ints, or `None` if `s` is `None`/blank.

    Raises:
        ValueError: if `s` is blank between commas, or contains duplicate ids.
    """
    if s is None or not str(s).strip():
        return None
    parts = [p.strip() for p in str(s).split(',')]
    if not parts or any(p == '' for p in parts):
        raise ValueError('--neural-trial-index must be comma-separated ints, e.g. 0,4,10')
    out = [int(p) for p in parts]
    if len(set(out)) != len(out):
        raise ValueError(f'--neural-trial-index has duplicate neural ids: {out!r}')
    return out


def parse_include_splits_csv(s: str) -> list[str]:
    """Parse a comma-separated `--include-splits` argument into a validated split list.

    Args:
        s: comma-separated splits, e.g. `'val,test'`.

    Returns:
        Lowercased, whitespace-trimmed split names, in the order given.

    Raises:
        argparse.ArgumentTypeError: if `s` is blank or contains an invalid split name.
    """
    splits = [x.strip().lower() for x in s.split(',') if x.strip()]
    if not splits:
        raise argparse.ArgumentTypeError('--include-splits must contain at least one split')
    invalid = [x for x in splits if x not in _VALID_SPLITS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f'Invalid split(s) {invalid}; expected comma-separated val,test',
        )
    return splits


def indices_for_neural_trials(neural_idx_test: np.ndarray, requested: list[int]) -> list[int]:
    """Map each neural trial id in `requested` to its row index in `neural_idx_test`.

    Args:
        neural_idx_test: array of neural trial ids, one per split row.
        requested: neural trial ids to look up, order preserved in the output.

    Returns:
        List of row indices into `neural_idx_test`, in the same order as `requested`.

    Raises:
        ValueError: if `neural_idx_test` has duplicate ids, or any `requested` id is missing.
    """
    row_by_neural: dict[int, int] = {}
    for i in range(len(neural_idx_test)):
        n = int(neural_idx_test[i])
        if n in row_by_neural:
            prev = row_by_neural[n]
            raise ValueError(
                f'neural_idx_test rows {prev} and {i} both have neural_trial_idx {n}; '
                'ambiguous for --neural-trial-index',
            )
        row_by_neural[n] = i
    rows: list[int] = []
    missing: list[int] = []
    for r in requested:
        if r not in row_by_neural:
            missing.append(r)
        else:
            rows.append(row_by_neural[r])
    if missing:
        raise ValueError(
            f'--neural-trial-index ids not present in this split: {sorted(missing)!r}; '
            f'split has neural trial ids from {sorted(row_by_neural.keys())}',
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the img-token unprojection entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eid', type=str, required=True)
    ap.add_argument(
        '--decoding-npy',
        type=Path,
        required=True,
        help='decoding_results_img_tokens_compressed.npy from the neural decoder test step.',
    )
    ap.add_argument(
        '--pca-npz', type=Path, required=True, help='img_tokens_pca_joint.npz from PCA step',
    )
    ap.add_argument(
        '--compressed-trials-npz',
        type=Path,
        required=True,
        help='img_tokens_compressed_trials.npz (for L, k_comp and neural trial metadata)',
    )
    ap.add_argument(
        '--out-root',
        type=Path,
        required=True,
        help='e.g. $MODEL_ROOT/img_tokens_compressed_estimated',
    )
    ap.add_argument(
        '--include-splits',
        type=parse_include_splits_csv,
        default=['test'],
        metavar='SPLITS',
        help=(
            'Comma-separated splits to unproject and write, e.g. val,test (default: test). '
            'Each split is written to --out-root/<eid>/<split>/.'
        ),
    )
    ap.add_argument(
        '--neural-trial-index',
        type=str,
        default=None,
        help=(
            'Optional comma-separated global neural trial indices, applied independently within '
            'each requested split: only those rows are unprojected; writes '
            'img_tokens_estimated_neuraltrialXXXX.npz (one file per requested id).'
        ),
    )
    return ap.parse_args(argv)


def _unproject_one_split(
    split: str,
    *,
    eid: str,
    decoding_npy: Path,
    trials_path: Path,
    pca,
    z_avg: np.ndarray,
    z_std: np.ndarray,
    out_root: Path,
    neural_filter: list[int] | None,
) -> None:
    """Unproject one split's neural-decoded compressed tokens and write per-trial `.npz` files.

    Args:
        split: split name (`'val'` or `'test'`).
        eid: session id.
        decoding_npy: path to the neural decoder's `decoding_results_img_tokens_compressed.npy`.
        trials_path: resolved path to `img_tokens_compressed_trials.npz`.
        pca: fitted sklearn PCA model.
        z_avg: session mean used to denormalize (broadcastable to the full token shape).
        z_std: session std used to denormalize.
        out_root: root directory; output is written under `out_root/eid/split/`.
        neural_filter: optional list of neural trial ids to restrict output to.

    Raises:
        KeyError: if `trials_path` lacks `{split}_z_trials_time`.
        ValueError: if shapes disagree between the decoder's `pred` and the split's tokens.
    """
    t0 = time.perf_counter()
    cnn = _load_decoding_payload(decoding_npy, eid, split)
    pred = np.asarray(cnn['pred'], dtype=np.float64)
    with np.load(trials_path, allow_pickle=True) as d:
        keys = set(d.files)
        split_z_key = f'{split}_z_trials_time'
        if split_z_key not in keys:
            raise KeyError(f'{trials_path}: need {split_z_key}; got {sorted(keys)}')
        split_z = np.asarray(d[split_z_key], dtype=np.float32)
        trial_split_all = np.asarray(d['trial_split'], dtype=object)
        neural_idx_all = np.asarray(d['neural_trial_idx'], dtype=np.int64)
        meta_json = None
        if 'meta_json' in keys:
            raw = d['meta_json']
            meta_json = raw.item() if hasattr(raw, 'item') else raw
        split_iv_key = f'{split}_intervals'
        split_iv = (
            np.asarray(d[split_iv_key], dtype=np.float64) if split_iv_key in keys else None
        )
    log_step(
        f'[split={split!r}] Loaded compressed trial metadata in '
        f'{time.perf_counter() - t0:.2f}s  {split_z_key}_shape={split_z.shape}',
        level='info',
    )

    split_mask = np.array([str(t).lower() == split for t in trial_split_all], dtype=bool)
    neural_idx_split = neural_idx_all[split_mask]

    k, t_bins, le = pred.shape
    _, _, l_tok, k_comp = split_z.shape
    if le != l_tok * k_comp:
        raise ValueError(
            f"[split={split!r}] pred last dim {le} != L*k_comp ({l_tok}*{k_comp}); "
            'check PCA n_feat_keep matches decoding.',
        )

    if k != split_z.shape[0]:
        raise ValueError(
            f"[split={split!r}] pred trials K={k} != {split_z_key} rows {split_z.shape[0]}",
        )
    if len(neural_idx_split) != k:
        raise ValueError(
            f"[split={split!r}] neural_idx_split len {len(neural_idx_split)} != pred K={k}",
        )

    row_orig: list[int] | None = None
    z_pred_k = compressed_pred_from_decoder_result(
        {eid: cnn}, eid, n_tokens=l_tok, n_comp=k_comp,
    )
    if neural_filter is not None:
        t0 = time.perf_counter()
        row_orig = indices_for_neural_trials(neural_idx_split, neural_filter)
        ix = np.asarray(row_orig, dtype=np.int64)
        z_pred_k = np.asarray(z_pred_k[ix], dtype=np.float64)
        neural_idx_split = neural_idx_split[ix]
        if split_iv is not None:
            split_iv = split_iv[ix]
        log_step(
            f"[split={split!r}] Applied neural trial filter in "
            f'{time.perf_counter() - t0:.2f}s  selected={len(neural_filter)}',
            level='info',
        )

    out_dir = out_root / eid / split
    out_dir.mkdir(parents=True, exist_ok=True)

    k_out = int(z_pred_k.shape[0])
    t0 = time.perf_counter()
    log_step(
        f"[split={split!r}] Unprojecting and writing {k_out} trial(s) one at a time to keep "
        'peak memory low',
        level='info',
    )
    for j in range(k_out):
        slab = pca_unproject(pca, z_pred_k[j : j + 1], z_avg, z_std)
        slab = np.asarray(slab, dtype=np.float32)
        row_in_full = int(row_orig[j]) if row_orig is not None else j
        ni = int(neural_idx_split[j])
        kw = {
            'z': slab,
            'neural_trial_idx': np.int64(ni),
            'trial_split': np.array([split], dtype=object),
            'meta_json': json.dumps(
                {
                    'source': 'unproject.py',
                    'decoding_npy': str(decoding_npy.resolve()),
                    'pca_npz': str(trials_path.resolve()),
                    'compressed_trials_npz': str(trials_path.resolve()),
                    'trial_row_in_split': row_in_full,
                    'layout': 'z shape (1, T, L, D_full)',
                    'split': split,
                    **(
                        {'neural_trial_index_filter': neural_filter}
                        if neural_filter is not None
                        else {}
                    ),
                },
            ),
        }
        if meta_json is not None:
            kw['compressed_trials_meta_json'] = np.asarray(meta_json)
        if split_iv is not None and split_iv.shape[0] > j:
            kw['trial_interval'] = split_iv[j]

        batch_path = out_dir / f'img_tokens_estimated_neuraltrial{ni:04d}.npz'
        np.savez_compressed(batch_path, **kw)
        log_step(
            f'[split={split!r}] Wrote {j + 1}/{k_out}: {batch_path.name}  z_shape={slab.shape}',
            level='info',
        )
        del slab, kw

    label = k_out if neural_filter is None else len(neural_filter)
    log_step(
        f"[split={split!r}] Wrote {label} neural-trial npz files under {out_dir} in "
        f'{time.perf_counter() - t0:.2f}s',
        level='info',
    )


def main(argv: list[str] | None = None) -> None:
    """Run the img-token unprojection pipeline end to end (CLI entry point)."""
    t_total = time.perf_counter()
    args = parse_args(argv)

    log_step(
        f'Starting img-token unprojection for eid={args.eid} splits={args.include_splits}',
        level='info',
    )

    t0 = time.perf_counter()
    trials_path = resolve_compressed_trials_npz_path(
        args.compressed_trials_npz, session_id=args.eid,
    )
    log_step(
        f'Resolved compressed trials path in {time.perf_counter() - t0:.2f}s: {trials_path}',
        level='info',
    )

    t0 = time.perf_counter()
    pca_npz_path = resolve_compressed_trials_npz_path(args.pca_npz, session_id=args.eid)
    pca, z_avg, z_std = _load_pca_bundle(pca_npz_path)
    log_step(
        f'Loaded PCA bundle in {time.perf_counter() - t0:.2f}s  '
        f'z_avg_shape={z_avg.shape}  z_std_shape={z_std.shape}',
        level='info',
    )

    neural_filter = parse_neural_trial_index_csv(args.neural_trial_index)

    for split in args.include_splits:
        _unproject_one_split(
            split,
            eid=args.eid,
            decoding_npy=args.decoding_npy,
            trials_path=trials_path,
            pca=pca,
            z_avg=z_avg,
            z_std=z_std,
            out_root=args.out_root,
            neural_filter=neural_filter,
        )

    log_step(
        f'Finished img-token unprojection in {time.perf_counter() - t_total:.2f}s',
        level='info',
    )


if __name__ == '__main__':
    main()
