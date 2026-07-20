"""Load neural decoding outputs on PCA-compressed img tokens, unproject to full-D tokens.

Uses ``compressed_pred_from_decoder_result`` + ``pca_unproject`` (see
``docs/ibl_encoding_decoding.md`` step 5). Writes per-trial batch npz files under
``--out-root/<eid>/test/`` matching inference-style batching (one trial per file).
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


def _load_decoding_payload(path: Path, eid: str) -> dict:
    """Load the per-eid CNN decoding payload from a `decoding_results_img_tokens_compressed.npy`.

    Args:
        path: path to the `.npy` file (an object array of a dict, or a dict directly).
        eid: session id key to extract from the loaded mapping.

    Returns:
        The inner `'cnn'` payload dict for `eid` when nested, else the raw per-eid value.

    Raises:
        KeyError: if `eid` is not present in the loaded mapping.
    """
    arr = np.load(path, allow_pickle=True)
    data = arr.item() if getattr(arr, 'ndim', 1) == 0 else arr
    if eid not in data:
        available = list(data.keys()) if hasattr(data, 'keys') else type(data)
        raise KeyError(f'{path}: missing key {eid!r}; got {available}')
    inner = data[eid]
    if isinstance(inner, dict) and 'cnn' in inner:
        return inner['cnn']
    return inner


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


def indices_for_neural_trials(neural_idx_test: np.ndarray, requested: list[int]) -> list[int]:
    """Map each neural trial id in `requested` to its row index in `neural_idx_test`.

    Args:
        neural_idx_test: array of neural trial ids, one per test-split row.
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
            f'--neural-trial-index ids not present in test split: {sorted(missing)!r}; '
            f'test has neural trial ids from {sorted(row_by_neural.keys())}',
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
        '--neural-trial-index',
        type=str,
        default=None,
        help=(
            'Optional comma-separated global neural trial indices (test split only): only those '
            'rows are unprojected; writes img_tokens_estimated_neuraltrialXXXX.npz (one file per '
            'requested id).'
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the img-token unprojection pipeline end to end (CLI entry point)."""
    t_total = time.perf_counter()
    args = parse_args(argv)

    log_step(f'Starting img-token unprojection for eid={args.eid}', level='info')

    t0 = time.perf_counter()
    trials_path = resolve_compressed_trials_npz_path(
        args.compressed_trials_npz, session_id=args.eid,
    )
    log_step(
        f'Resolved compressed trials path in {time.perf_counter() - t0:.2f}s: {trials_path}',
        level='info',
    )

    t0 = time.perf_counter()
    cnn = _load_decoding_payload(args.decoding_npy, args.eid)
    pred = np.asarray(cnn['pred'], dtype=np.float64)
    with np.load(trials_path, allow_pickle=True) as d:
        keys = set(d.files)
        if 'test_z_trials_time' not in keys:
            raise KeyError(f'{trials_path}: need test_z_trials_time; got {sorted(keys)}')
        test_z = np.asarray(d['test_z_trials_time'], dtype=np.float32)
        trial_split_all = np.asarray(d['trial_split'], dtype=object)
        neural_idx_all = np.asarray(d['neural_trial_idx'], dtype=np.int64)
        meta_json = None
        if 'meta_json' in keys:
            raw = d['meta_json']
            meta_json = raw.item() if hasattr(raw, 'item') else raw
        te_iv = (
            np.asarray(d['test_intervals'], dtype=np.float64) if 'test_intervals' in keys else None
        )
    log_step(
        f'Loaded compressed trial metadata in {time.perf_counter() - t0:.2f}s  '
        f'test_z_shape={test_z.shape}',
        level='info',
    )

    test_mask = np.array([str(t).lower() == 'test' for t in trial_split_all], dtype=bool)
    neural_idx_test = neural_idx_all[test_mask]

    k, t_bins, le = pred.shape
    _, _, l_tok, k_comp = test_z.shape
    if le != l_tok * k_comp:
        raise ValueError(
            f'pred last dim {le} != L*k_comp ({l_tok}*{k_comp}); '
            'check PCA n_feat_keep matches decoding.',
        )

    if k != test_z.shape[0]:
        raise ValueError(
            f'pred trials K={k} != test_z_trials_time rows {test_z.shape[0]} '
            '(decoder evaluates on held-out test only).',
        )
    if len(neural_idx_test) != k:
        raise ValueError(f'neural_idx_test len {len(neural_idx_test)} != pred K={k}')

    neural_filter = parse_neural_trial_index_csv(args.neural_trial_index)
    row_orig: list[int] | None = None
    z_pred_k = compressed_pred_from_decoder_result(
        {args.eid: cnn}, args.eid, n_tokens=l_tok, n_comp=k_comp,
    )
    if neural_filter is not None:
        t0 = time.perf_counter()
        row_orig = indices_for_neural_trials(neural_idx_test, neural_filter)
        ix = np.asarray(row_orig, dtype=np.int64)
        z_pred_k = np.asarray(z_pred_k[ix], dtype=np.float64)
        neural_idx_test = neural_idx_test[ix]
        if te_iv is not None:
            te_iv = te_iv[ix]
        log_step(
            f'Applied neural trial filter in {time.perf_counter() - t0:.2f}s  '
            f'selected={len(neural_filter)}',
            level='info',
        )

    t0 = time.perf_counter()
    pca, z_avg, z_std = _load_pca_bundle(args.pca_npz)
    log_step(
        f'Loaded PCA bundle in {time.perf_counter() - t0:.2f}s  '
        f'z_avg_shape={z_avg.shape}  z_std_shape={z_std.shape}',
        level='info',
    )

    out_dir = args.out_root / args.eid / 'test'
    out_dir.mkdir(parents=True, exist_ok=True)

    k_out = int(z_pred_k.shape[0])
    t0 = time.perf_counter()
    log_step(
        f'Unprojecting and writing {k_out} trial(s) one at a time to keep peak memory low',
        level='info',
    )
    for j in range(k_out):
        slab = pca_unproject(pca, z_pred_k[j : j + 1], z_avg, z_std)
        slab = np.asarray(slab, dtype=np.float32)
        row_in_full = int(row_orig[j]) if row_orig is not None else j
        ni = int(neural_idx_test[j])
        kw = {
            'z': slab,
            'neural_trial_idx': np.int64(ni),
            'trial_split': np.array(['test'], dtype=object),
            'meta_json': json.dumps(
                {
                    'source': 'unproject.py',
                    'decoding_npy': str(args.decoding_npy.resolve()),
                    'pca_npz': str(args.pca_npz.resolve()),
                    'compressed_trials_npz': str(trials_path.resolve()),
                    'trial_row_in_test_split': row_in_full,
                    'layout': 'z shape (1, T, L, D_full)',
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
        if te_iv is not None and te_iv.shape[0] > j:
            kw['trial_interval'] = te_iv[j]

        batch_path = out_dir / f'img_tokens_estimated_neuraltrial{ni:04d}.npz'
        np.savez_compressed(batch_path, **kw)
        log_step(
            f'Wrote {j + 1}/{k_out}: {batch_path.name}  z_shape={slab.shape}',
            level='info',
        )
        del slab, kw

    label = k_out if neural_filter is None else len(neural_filter)
    log_step(
        f'Wrote {label} neural-trial npz files under {out_dir} in {time.perf_counter() - t0:.2f}s',
        level='info',
    )
    log_step(
        f'Finished img-token unprojection in {time.perf_counter() - t_total:.2f}s',
        level='info',
    )


if __name__ == '__main__':
    main()
