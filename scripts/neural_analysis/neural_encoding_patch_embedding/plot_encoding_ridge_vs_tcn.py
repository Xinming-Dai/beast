#!/usr/bin/env python3
"""Held-out BPS vs. number of PCA components for ridge and TCN encoders.

Reproduces the per-session figure of the ridge analysis (Full SABLE, NoGeom SABLE, DINO,
5-fold CV) and adds the TCN encoder (dashed lines, square markers). Also draws a pooled figure
over all sessions. Every layout is drawn with two error-bar types:

- ``fold``: mean over neurons per fold, then mean +/- SD across folds (SD is 0 with one fold).
- ``neuron``: per-neuron BPS averaged over the selected folds, then mean +/- SEM across neurons.

Only neurons with at least one spike in a fold's test trials are used. Writes four figures
(``per_session_fold_sd``, ``per_session_neuron_sem``, ``pooled_fold_sd``,
``pooled_neuron_sem``) and a CSV with every plotted number.

TCN files are pickled dicts, so run this inside the ``beast`` conda env.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/plot_encoding_ridge_vs_tcn.py \
  --no-show
```
"""

import argparse
import csv
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.plot_helpers import (  # noqa: E402
    _default_output_path_helper,
    bps_per_neuron,
)

SESSION_EIDS = {
    '78': '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
    '72': '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
    '4b': '4b00df29-3769-43be-bb40-128b1cba6d35',
}
MODELS = ('full', 'nogeom', 'dino')
MODEL_LABELS = {'full': 'Full SABLE', 'nogeom': 'NoGeom SABLE', 'dino': 'DINO'}
MODEL_COLORS = {'full': 'tab:blue', 'nogeom': 'tab:orange', 'dino': 'tab:green'}
KS = (6, 12, 24, 32, 64, 128)
METHOD_STYLES = {
    'ridge': {'label': 'ridge', 'linestyle': '-', 'marker': 'o', 'offset': 0.0},
    'tcn': {'label': 'TCN', 'linestyle': '--', 'marker': 's', 'offset': 0.12},
}
Y_LABEL = 'Held-out BPS (bits/spike)'

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
    for path in sorted(root.glob('fullfit_*.npz')):
        match = RIDGE_PATTERN.fullmatch(path.name)
        if match is None or match['model'] not in models:
            continue
        with np.load(path) as data:
            trial_ids = np.sort(data['trial_ids'])
            has_spikes = spikes[data['trial_ids']].sum(axis=(0, 1)) > 0
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


def plot_encoding(
    rows: list[dict],
    *,
    pooled: bool,
    error: str,
    n_neurons: dict[str, int],
    footnote: str,
    save_path: Path,
    show: bool = False,
) -> plt.Figure:
    """Draw held-out BPS vs. K for every model and method, then save the figure.

    Args:
        rows: output of `summarize`.
        pooled: draw one pooled panel instead of one panel per session.
        error: `'fold'` or `'neuron'`, only used for the title and legend.
        n_neurons: session name (or `'all'`) to total number of neurons, for panel titles.
        footnote: text under the panels, describing folds, masking and time bins.
        save_path: output image path.
        show: open an interactive window.

    Returns:
        The matplotlib figure.
    """
    sessions_present = {r['session'] for r in rows}
    units = ['all'] if pooled else [s for s in SESSION_EIDS if s in sessions_present]
    fig, axes = plt.subplots(
        1,
        len(units),
        figsize=(6.4, 5.6) if pooled else (15.9, 5.4),
        squeeze=False,
        dpi=150,
    )
    for ax, unit in zip(axes[0], units):
        for method, style in METHOD_STYLES.items():
            for model in MODELS:
                sub = sorted(
                    (
                        r
                        for r in rows
                        if (r['method'], r['session'], r['model']) == (method, unit, model)
                    ),
                    key=lambda r: r['k'],
                )
                if not sub:
                    continue
                x = np.array([KS.index(r['k']) for r in sub]) + style['offset']
                ax.errorbar(
                    x,
                    [r['mean'] for r in sub],
                    yerr=[r['err'] for r in sub],
                    color=MODEL_COLORS[model],
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    label=f'{MODEL_LABELS[model]} ({style["label"]})',
                )
        ax.set_xticks(range(len(KS)), [str(k) for k in KS])
        ax.set_xlabel('Number of PCA components (K)')
        ax.set_ylabel(Y_LABEL)
        title = 'All sessions' if pooled else f'Session {unit}'
        ax.set_title(f'{title} (n={n_neurons[unit]})')
        ax.grid(axis='y', alpha=0.3)

    spread = 'SD across held-out folds' if error == 'fold' else 'SEM across neurons'
    fig.suptitle(
        f'Neural encoding performance, ridge vs. TCN\nmean ± {spread}',
        fontsize=14,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.89),
        ncol=3 if pooled else 6,
        frameon=False,
        fontsize=9,
    )
    fig.text(0.5, 0.005, footnote, ha='center', va='bottom', fontsize=7, color='0.35')
    # set margins by hand: tight_layout would also reserve space for the suptitle and legend
    if pooled:
        fig.subplots_adjust(left=0.13, right=0.97, top=0.70, bottom=0.15)
    else:
        fig.subplots_adjust(left=0.05, right=0.99, top=0.79, bottom=0.15, wspace=0.25)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _write_csv(rows: list[dict], path: Path) -> None:
    """Write summary rows to a CSV file."""
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Load ridge and TCN results, then write the four figures and the summary CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument(
        '--neuron-folds',
        type=int,
        nargs='+',
        default=None,
        help='folds for the neuron-SEM plots (default: the folds the TCN has, else all)',
    )
    parser.add_argument(
        '--tcn-time-slice',
        type=int,
        nargs=2,
        default=None,
        metavar=('START', 'STOP'),
        help='time bins used to score the TCN (ridge predictions have 48 bins, TCN has 60)',
    )
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    records: dict[RecordKey, Record] = {}
    for session in args.sessions:
        eid = SESSION_EIDS[session]
        spikes = load_session_spikes(eid, args.neural_dir)
        records.update(load_ridge_records(session, RIDGE_ROOTS[session], spikes))
        records.update(
            load_tcn_records(
                session,
                eid,
                TCN_ROOTS,
                time_slice=tuple(args.tcn_time_slice) if args.tcn_time_slice else None,
            ),
        )
    check_matching_trials(records)

    tcn_folds = sorted({key[4] for key in records if key[0] == 'tcn'})
    neuron_folds = args.neuron_folds if args.neuron_folds is not None else tcn_folds or None
    n_neurons = {
        s: next(rec.bps.size for key, rec in records.items() if key[1] == s)
        for s in args.sessions
    }
    n_neurons['all'] = sum(n_neurons.values())
    folds_note = f'neuron plots use fold(s) {neuron_folds if neuron_folds else "all"}'
    time_note = (
        f'TCN scored on time bins {tuple(args.tcn_time_slice)}'
        if args.tcn_time_slice
        else 'TCN scored on all 60 time bins (ridge: 48)'
    )
    footnote = (
        f'neurons with test spikes only; fold SD is 0 for a single fold; {folds_note}; {time_note}'
    )

    save_dir = (
        args.output_dir
        or _default_output_path_helper(RESULTS_DIR, 'encoding_ridge_vs_tcn', 'summary.csv').parent
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for pooled in (False, True):
        for error, tag in (('fold', 'fold_sd'), ('neuron', 'neuron_sem')):
            rows = summarize(records, pooled=pooled, error=error, neuron_folds=neuron_folds)
            layout = 'pooled' if pooled else 'per_session'
            path = save_dir / f'{layout}_{tag}.png'
            plot_encoding(
                rows,
                pooled=pooled,
                error=error,
                n_neurons=n_neurons,
                footnote=footnote,
                save_path=path,
                show=not args.no_show,
            )
            all_rows += [{'layout': layout, 'error': error, **row} for row in rows]
            print(path)
    _write_csv(all_rows, save_dir / 'summary.csv')
    print(save_dir / 'summary.csv')


if __name__ == '__main__':
    main()
