#!/usr/bin/env python3
"""Rank all neurons of all sessions for the patch-embedding ridge encoders and save the rankings.

Every ``fullfit_<model>_f<fold>_k<k>.npz`` of the ridge results directories is scored per neuron on
its held-out trials: BPS, the correlation of the predicted and ground-truth PSTH, and the modulation
of the predicted PSTH over time. One CSV per (session, model, K, fold) is written to
``RANKING_DIR/<session>/``, so the plotting scripts of ``neural_encoding_patch_embedding`` can look
it up with ``--ranking``. By default everything is ranked, the flags restrict the runs.

```bash
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_corr_patch.py
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_corr_patch.py \
  --session 72 --model dino full --k 6 --fold 0
```
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (
    MODELS,
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    RIDGE_PATTERN,
    RIDGE_ROOTS,
    SESSION_EIDS,
    load_ridge_predictions,
    load_session_spikes,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (
    DEFAULT_MIN_MODULATION,
    DEFAULT_MIN_PSTH_CORR,
    pass_filters,
    rank_neurons,
    save_ranking,
)

RANKING_DIR = RESULTS_DIR / 'neuron_ranking'


def ranking_path(session: str, model: str, k: int, fold: int, root: Path = RANKING_DIR) -> Path:
    """Path of the ranking CSV of one (session, model, K, fold).

    Args:
        session: session name.
        model: model name.
        k: number of PCA components.
        fold: held-out fold.
        root: folder of the rankings.

    Returns:
        `<root>/<session>/ranking_<model>_k<k>_fold<fold>.csv`.
    """
    return root / session / f'ranking_{model}_k{k}_fold{fold}.csv'


def find_ridge_runs(
    root: Path,
    models: list[str],
    ks: list[int],
    folds: list[int] | None,
) -> list[tuple[str, int, int]]:
    """The (model, K, fold) runs found in a ridge results directory.

    Args:
        root: directory with `fullfit_<model>_f<fold>_k<K>.npz` files.
        models: models to keep.
        ks: numbers of PCA components to keep.
        folds: folds to keep, None for all.

    Returns:
        The runs, sorted by model, K and fold.
    """
    runs = []
    for path in root.glob('fullfit_*.npz'):
        match = RIDGE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        model, k, fold = match['model'], int(match['k']), int(match['fold'])
        if model in models and k in ks and (folds is None or fold in folds):
            runs.append((model, k, fold))
    return sorted(runs)


def main() -> None:
    """Write one ranking CSV per (session, model, K, fold) found in the ridge results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument('--model', nargs='+', default=list(MODELS), choices=MODELS)
    parser.add_argument('--k', nargs='+', type=int, default=list(PCA_KS), choices=PCA_KS)
    parser.add_argument('--fold', nargs='+', type=int, default=None, help='default: all folds')
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('--output-dir', type=Path, default=RANKING_DIR)
    args = parser.parse_args()

    n_written = 0
    for session in args.session:
        runs = find_ridge_runs(RIDGE_ROOTS[session], args.model, args.k, args.fold)
        if not runs:
            print(f'session {session}: no ridge files found in {RIDGE_ROOTS[session]}')
            continue
        spikes = load_session_spikes(SESSION_EIDS[session], args.neural_dir)
        for model, k, fold in runs:
            _, gt, pred = load_ridge_predictions(session, model, k, fold, spikes)
            df = rank_neurons(gt, pred)
            path = ranking_path(session, model, k, fold, args.output_dir)
            save_ranking(df, path, {'session': session, 'model': model, 'k': k, 'fold': fold})
            n_pass = pass_filters(
                df,
                min_psth_corr=DEFAULT_MIN_PSTH_CORR,
                min_modulation=DEFAULT_MIN_MODULATION,
            ).sum()
            print(f'{path}: {len(df)} neurons, {n_pass} pass the default PSTH filters')
            n_written += 1
    print(f'wrote {n_written} rankings to {args.output_dir}')


if __name__ == '__main__':
    main()
