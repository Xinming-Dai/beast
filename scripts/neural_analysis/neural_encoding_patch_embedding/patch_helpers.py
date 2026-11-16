#!/usr/bin/env python3
"""Loaders and summaries for the patch-embedding encoding experiment (ridge, later TCN).

Features are PCA-reduced patch embeddings (K components) of Full SABLE, NoGeom SABLE and DINO.
Ridge results are `fullfit_<model>_f<fold>_k<K>.npz` files per session (5-fold CV), TCN results
are pickled `patch_tcn_k<K>_fold<F>.npy` files. Colors and fonts are not defined here, use
`scripts/neural_analysis/plot_helpers.py` through :func:`model_color`.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.neural_analysis.plot_helpers import ENCODING_BASE_COLORS, bps_per_neuron

SESSION_EIDS = {
    '78': '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
    '72': '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
    '4b': '4b00df29-3769-43be-bb40-128b1cba6d35',
}
MODELS = ('full', 'nogeom', 'dino')
MODEL_LABELS = {'full': 'Full SABLE', 'nogeom': 'SABLE NoGeom', 'dino': 'DINO'}
MODEL_COLOR_KEYS = {'full': 'sable_full', 'nogeom': 'sable_nogeom', 'dino': 'dino'}
PCA_KS = (6, 12, 24, 32, 64, 128)
Y_LABEL = 'Avg BPS'

RIDGE_TIME_SLICE = (6, 54)  # ridge predicts 48 of the 60 spike bins, this window reproduces it
RIDGE_DURATION_S = 0.8  # 48 bins of 1/60 s
PRED_FLOOR = 1e-3  # lower bound for ridge rates, which can be negative

RIDGE_BASE = Path('/work/hdd/bezq/jzhang107/sable/cv5_step20851')
RIDGE_ROOTS = {
    '78': RIDGE_BASE / 'patch_ridge_cv5',
    '72': RIDGE_BASE / 'patch_ridge_cv5_sessions' / 'session_72',
    '4b': RIDGE_BASE / 'patch_ridge_cv5_sessions' / 'session_4b',
}
CKPT_ROOT = Path('/work/hdd/bfsr/xdai3/project3d/twoview3d_ckpts')
TCN_ROOTS = {
    'full': CKPT_ROOT / 'beast_sable/ibl_pretrain_restricted_sessions/21047248/latents/patch_TCN',
    'nogeom': (
        CKPT_ROOT
        / 'beast_sable/ibl_pretrain_restricted_sessions/22359987_disable_geom/latents/patch_TCN'
    ),
    'dino': CKPT_ROOT / 'dino_baseline/latents/patch_TCN',
}
NEURAL_DIR = Path('/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data')
RESULTS_DIR = Path(
    CKPT_ROOT / 'beast_sable/ibl_pretrain_restricted_sessions/SABLE_encoding_patch_embedding',
)

RIDGE_PATTERN = re.compile(r'fullfit_(?P<model>\w+)_f(?P<fold>\d+)_k(?P<k>\d+)\.npz')
TCN_PATTERN = re.compile(r'patch_tcn_k(?P<k>\d+)_fold(?P<fold>\d+)\.npy')


@dataclass
class Record:
    """Per-neuron held-out BPS of one (method, session, model, K, fold) run.

    Attributes:
        bps: per-neuron BPS, shape (neurons,), NaN for neurons without test spikes.
        trial_ids: sorted row indices of the test trials of this fold.
    """

    bps: np.ndarray
    trial_ids: np.ndarray


RecordKey = tuple[str, str, str, int, int]  # method, session, model, k, fold


def load_session_spikes(eid: str, neural_dir: Path) -> np.ndarray:
    """Stack train, val and test spikes in the original trial order.

    Args:
        eid: session eid.
        neural_dir: directory holding `<eid>/<eid>_aligned.npz`.

    Returns:
        Array of shape (trials, time, neurons).
    """
    with np.load(neural_dir / eid / f'{eid}_aligned.npz') as data:
        return np.concatenate(
            [data['train_spikes'], data['val_spikes'], data['test_spikes']],
            axis=0,
        )


def load_ridge_records(
    session: str,
    root: Path,
    spikes: np.ndarray,
    models: tuple[str, ...] = MODELS,
) -> dict[RecordKey, Record]:
    """Load the ridge per-neuron BPS of every (model, K, fold) file under `root`.

    Args:
        session: session name used in the record keys.
        root: directory with `fullfit_<model>_f<fold>_k<K>.npz` files.
        spikes: session spikes of shape (trials, time, neurons), used to mask neurons.
        models: models to load.

    Returns:
        Records keyed by (`'ridge'`, session, model, K, fold).

    Raises:
        FileNotFoundError: if `root` holds no matching ridge file.
    """
    records: dict[RecordKey, Record] = {}
    spikes_per_trial = spikes.sum(axis=1)
    for path in sorted(root.glob('fullfit_*.npz')):
        match = RIDGE_PATTERN.fullmatch(path.name)
        if match is None or match['model'] not in models:
            continue
        with np.load(path) as data:
            trial_ids = np.sort(data['trial_ids'])
            has_spikes = spikes_per_trial[data['trial_ids']].sum(axis=0) > 0
            bps = np.where(has_spikes, data['fold_bps'].astype(np.float64), np.nan)
        key = ('ridge', session, match['model'], int(match['k']), int(match['fold']))
        records[key] = Record(bps=bps, trial_ids=trial_ids)
    if not records:
        raise FileNotFoundError(f'no ridge files found in {root}')
    return records


def load_tcn_records(
    session: str,
    eid: str,
    roots: dict[str, Path],
    time_slice: tuple[int, int] | None = None,
) -> dict[RecordKey, Record]:
    """Load the TCN per-neuron BPS of every (model, K, fold) file found for a session.

    Per-neuron BPS is recomputed from the stored `gt` and `pred`, which already gives NaN for
    neurons without spikes in the test trials. Models with no output directory are skipped.

    Args:
        session: session name used in the record keys.
        eid: session eid, the subfolder of each model root.
        roots: model name to directory holding `<eid>/patch_tcn_k<K>_fold<F>.npy`.
        time_slice: optional (start, stop) time bins to score, default all bins.

    Returns:
        Records keyed by (`'tcn'`, session, model, K, fold).
    """
    records: dict[RecordKey, Record] = {}
    for model, root in roots.items():
        for path in sorted((root / eid).glob('patch_tcn_k*_fold*.npy')):
            match = TCN_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            payload = np.load(path, allow_pickle=True).item()
            result = payload[eid]['cnn']
            gt, pred = result['gt'], result['pred']
            if time_slice is not None:
                gt, pred = gt[:, slice(*time_slice)], pred[:, slice(*time_slice)]
            key = ('tcn', session, model, int(match['k']), int(match['fold']))
            records[key] = Record(
                bps=bps_per_neuron(gt, pred),
                trial_ids=np.sort(payload['meta']['idx_test']),
            )
    return records


def check_matching_trials(records: dict[RecordKey, Record]) -> None:
    """Check that ridge and TCN runs of the same session and fold share the same test trials.

    Args:
        records: records of both methods.

    Raises:
        ValueError: if the test trials of a ridge and a TCN run differ.
    """
    for (method, session, _model, _k, fold), record in records.items():
        if method != 'tcn':
            continue
        for (other_method, other_session, _m, _kk, other_fold), other in records.items():
            if (other_method, other_session, other_fold) != ('ridge', session, fold):
                continue
            if not np.array_equal(record.trial_ids, other.trial_ids):
                raise ValueError(
                    f'ridge and TCN test trials differ: session {session} fold {fold}',
                )
            break


def _vectors_by_fold(
    records: dict[RecordKey, Record],
    method: str,
    model: str,
    k: int,
    sessions: list[str],
) -> dict[int, np.ndarray]:
    """Per-neuron BPS vectors by fold, concatenated over `sessions` (only shared folds)."""
    per_session = [
        {key[4]: rec.bps for key, rec in records.items() if key[:4] == (method, s, model, k)}
        for s in sessions
    ]
    folds = set.intersection(*(set(d) for d in per_session)) if per_session else set()
    return {f: np.concatenate([d[f] for d in per_session]) for f in sorted(folds)}


def summarize(
    records: dict[RecordKey, Record],
    *,
    pooled: bool,
    error: str,
    neuron_folds: list[int] | None = None,
) -> list[dict]:
    """Mean and error of BPS for every (method, session, model, K).

    Args:
        records: records of both methods and all sessions.
        pooled: pool all sessions into one unit named `'all'`, else one unit per session.
        error: `'fold'` for mean over neurons then SD (ddof=1) across folds, `'neuron'` for
            per-neuron BPS averaged over folds then SEM across neurons.
        neuron_folds: folds used when `error='neuron'`, default all folds on disk.

    Returns:
        One row per unit with keys method, session, model, k, mean, err, n_folds, n_neurons.

    Raises:
        ValueError: if `error` is not `'fold'` or `'neuron'`.
    """
    if error not in {'fold', 'neuron'}:
        raise ValueError(f'unknown error type: {error!r}')
    sessions = sorted({key[1] for key in records})
    units = [('all', sessions)] if pooled else [(s, [s]) for s in sessions]
    rows: list[dict] = []
    for method, model, k in sorted({(key[0], key[2], key[3]) for key in records}):
        for unit, unit_sessions in units:
            by_fold = _vectors_by_fold(records, method, model, k, unit_sessions)
            if error == 'neuron' and neuron_folds is not None:
                by_fold = {f: v for f, v in by_fold.items() if f in neuron_folds}
            if not by_fold:
                continue
            if error == 'fold':
                fold_means = np.array([np.nanmean(v) for v in by_fold.values()])
                mean = float(fold_means.mean())
                err = float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0
                n_neurons = int(np.isfinite(next(iter(by_fold.values()))).sum())
            else:
                # neurons without spikes in every selected fold are all-NaN, which is expected
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    per_neuron = np.nanmean(np.stack(list(by_fold.values())), axis=0)
                per_neuron = per_neuron[np.isfinite(per_neuron)]
                mean = float(per_neuron.mean())
                err = float(per_neuron.std(ddof=1) / np.sqrt(len(per_neuron)))
                n_neurons = len(per_neuron)
            rows.append(
                {
                    'method': method,
                    'session': unit,
                    'model': model,
                    'k': k,
                    'mean': mean,
                    'err': err,
                    'n_folds': len(by_fold),
                    'n_neurons': n_neurons,
                },
            )
    return rows


def model_color(model: str) -> str:
    """Color of a model, taken from `ENCODING_BASE_COLORS` in `plot_helpers.py`."""
    return ENCODING_BASE_COLORS[MODEL_COLOR_KEYS[model]]


def load_ridge_predictions(
    session: str,
    model: str,
    k: int,
    fold: int,
    spikes: np.ndarray,
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the held-out ridge predictions of one fold together with the matching spikes.

    Ridge predicts `RIDGE_TIME_SLICE` of the 60 spike bins. Predictions can be slightly negative,
    so they are floored at `PRED_FLOOR` to make them valid Poisson rates.

    Args:
        session: session name, a key of `RIDGE_ROOTS`.
        model: model name, one of `MODELS`.
        k: number of PCA components.
        fold: held-out fold.
        spikes: session spikes of shape (trials, time, neurons), see `load_session_spikes`.
        root: ridge directory, default `RIDGE_ROOTS[session]`.

    Returns:
        trial_ids of shape (trials,), gt and pred, both of shape (trials, 48, neurons).
    """
    path = (root or RIDGE_ROOTS[session]) / f'fullfit_{model}_f{fold}_k{k}.npz'
    with np.load(path) as data:
        trial_ids = np.asarray(data['trial_ids'])
        pred = np.clip(data['pred'].astype(np.float64), PRED_FLOOR, None)
    gt = spikes[trial_ids][:, slice(*RIDGE_TIME_SLICE)].astype(np.float64)
    return trial_ids, gt, pred


def neuron_bps(
    records: dict[RecordKey, Record],
    method: str,
    session: str,
    model: str,
    k: int,
    folds: list[int] | None = None,
) -> np.ndarray:
    """Per-neuron BPS of one (method, session, model, K), averaged over `folds`.

    Args:
        records: records of all runs.
        method: `'ridge'` or `'tcn'`.
        session: session name.
        model: model name.
        k: number of PCA components.
        folds: folds to average, default all folds on disk. Neurons without test spikes in all
            selected folds are NaN.

    Returns:
        Array of shape (neurons,).

    Raises:
        KeyError: if no selected fold exists for this run.
    """
    by_fold = _vectors_by_fold(records, method, model, k, [session])
    if folds is not None:
        by_fold = {fold: vec for fold, vec in by_fold.items() if fold in folds}
    if not by_fold:
        raise KeyError(f'no {method} record for session {session} model {model} k {k} {folds}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(np.stack(list(by_fold.values())), axis=0)


def restrict_records(
    records: dict[RecordKey, Record],
    idx_by_session: dict[str, list[int]],
) -> dict[RecordKey, Record]:
    """Keep only the given neurons of every session, all other neurons become NaN.

    `summarize` drops non-finite neurons, so the restricted records can be summarized as usual.

    Args:
        records: records of all runs.
        idx_by_session: session name to the indices of the neurons to keep.

    Returns:
        New records with the same keys, NaN BPS for the neurons that are not kept.
    """
    restricted: dict[RecordKey, Record] = {}
    for key, record in records.items():
        keep = np.zeros(record.bps.shape, dtype=bool)
        keep[idx_by_session[key[1]]] = True
        restricted[key] = Record(
            bps=np.where(keep, record.bps, np.nan),
            trial_ids=record.trial_ids,
        )
    return restricted


def load_ridge_session_data(
    sessions: list[str],
    neural_dir: Path = NEURAL_DIR,
) -> tuple[dict[RecordKey, Record], dict[str, np.ndarray]]:
    """Load the ridge records and the spikes of several sessions.

    Args:
        sessions: session names, keys of `SESSION_EIDS`.
        neural_dir: directory holding `<eid>/<eid>_aligned.npz`.

    Returns:
        Ridge records of all sessions and the spikes per session, shape (trials, time, neurons).
    """
    records: dict[RecordKey, Record] = {}
    spikes_by_session: dict[str, np.ndarray] = {}
    for session in sessions:
        spikes = load_session_spikes(SESSION_EIDS[session], neural_dir)
        spikes_by_session[session] = spikes
        records.update(load_ridge_records(session, RIDGE_ROOTS[session], spikes))
    return records, spikes_by_session
