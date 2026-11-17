#!/usr/bin/env python3
"""Rank the neurons of every session by the held-out BPS of Full SABLE and save the rankings.

The ranking is the basis of the 3D / 2D neuron split of
``neural_encoding_patch_embedding/bar_plot_encoding_patch.py --split-frac``: neurons that rank high
for Full SABLE (3D) are compared with the rest (2D). It uses the same per-neuron ridge BPS as the
bar plot. For every session and K, one CSV is written per fold and one for the BPS averaged over
all folds (`fold<F>` and `foldall` in the file name), so the plotting script only reads a CSV and
applies the fraction of neurons without recomputing the ranking.

```bash
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_for_bar_plot.py
```
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    SESSION_EIDS,
    load_ridge_session_data,
    neuron_bps,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (
    rank_by_bps,
    save_ranking,
)

logger = logging.getLogger(__name__)

RANKING_DIR = RESULTS_DIR / 'neuron_ranking'


def full_bps_ranking_path(
    session: str,
    k: int,
    folds: list[int] | None,
    root: Path = RANKING_DIR,
) -> Path:
    """Path of the Full SABLE BPS ranking CSV of one (session, K) and fold choice.

    Args:
        session: session name.
        k: number of PCA components.
        folds: a single fold, or None for the BPS averaged over all folds.
        root: folder of the rankings.

    Returns:
        `<root>/ranking_full_bps_<session>_k<k>_fold<fold>.csv`, with `fold` being `all` for None.
    """
    fold = 'all' if folds is None else str(folds[0])
    return root / f'ranking_full_bps_{session}_k{k}_fold{fold}.csv'


def main() -> None:
    """Write the per-fold and all-fold Full SABLE BPS rankings of every session and K."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument('--k', type=int, nargs='+', default=list(PCA_KS), choices=PCA_KS)
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('--output-dir', type=Path, default=RANKING_DIR)
    args = parser.parse_args()

    records, _ = load_ridge_session_data(args.sessions, args.neural_dir, ks=tuple(args.k))
    n_written = 0
    for session in args.sessions:
        for k in args.k:
            folds = sorted(
                {key[4] for key in records if key[1:4] == (session, 'full', k)},
            )
            if not folds:
                logger.warning(f'no full SABLE ridge files for session {session} K={k}, skipped')
                continue
            # one ranking per fold, then one for the BPS averaged over all folds on disk
            for fold_choice in [*([fold] for fold in folds), None]:
                bps = neuron_bps(records, 'ridge', session, 'full', k, fold_choice)
                meta_folds = ','.join(map(str, folds if fold_choice is None else fold_choice))
                path = full_bps_ranking_path(session, k, fold_choice, args.output_dir)
                save_ranking(
                    rank_by_bps(bps),
                    path,
                    {'session': session, 'model': 'full', 'k': k, 'folds': meta_folds},
                )
                n_written += 1
            n_valid = int(np.isfinite(bps).sum())
            logger.info(f'session {session} K={k} folds {folds}: {n_valid} neurons with BPS')
    print(f'wrote {n_written} rankings to {args.output_dir}')


if __name__ == '__main__':
    main()
