#!/usr/bin/env python3
"""PSTH stacks of the top-BPS neurons of the selected and of the unselected group.

Group counterpart of `psth_stack_patch.py`, drawn with the same :func:`plot_psth_stack`. The two
groups are the neurons flagged `selected == 1` and the other neurons in the per-neuron csv files of
`neuron_selection_handoff/`, read with `load_selection_records`. Within each group, the neurons
stacked are chosen in three steps:

1. keep the neurons with `wins_vs_dino >= --min-wins` (the csv column counts the five folds where
   Full SABLE beats DINO, so 5 means in every fold),
2. of those, keep the `--pool-size` neurons whose predicted PSTH correlates best with the ground
   truth (`psth_corr`, read from the ranking saved by
   `neuron_ranking_and_filtering/rank_neurons_bps_corr_patch.py` for this session, last model, K
   and fold),
3. of those, stack the `--num-neurons` neurons with the highest csv BPS of the last `--model`
   (`<model>_mean_bps`, the mean over the five folds).

`--min-wins 0` skips step 1 and `--pool-size 0` skips step 2 and the ranking file, so both zero
means the plain top-BPS neurons. The PSTHs and `psth_corr` are those of one fold, while the BPS and
`wins_vs_dino` are averaged over folds.

It writes `psth_stack_<session>_<models>_k<K>_fold<F>_selected.png` and `..._unselected.png`, each
with a `.txt` next to it that lists the BPS of the stacked neurons, as `psth_stack_patch.py` does.

```bash
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_corr_patch.py
python scripts/neural_analysis/neural_encoding_patch_embedding/psth_stack_patch_selected_vs_unselected.py \
  --session 72 --model dino full --k 6 --fold 0 --num-neurons 30 --no-show \
  --min-wins 5 --pool-size 20 --line-scale 0.3
```
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (
    MODELS,
    NEURAL_DIR,
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
    load_ridge_predictions,
    load_selection_records,
    load_session_spikes,
    model_color,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.psth_stack_patch import (
    format_bps_text,
    plot_psth_stack,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.top_bps_psth_compare_models_patch import (
    parse_trials_idx_arg,
    trials_idx_to_bounds,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.rank_neurons_bps_corr_patch import (
    RANKING_DIR,
    ranking_path,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (
    load_ranking,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper

SELECTION_KS = (6, 32)  # numbers of PCA components of the handoff csv files
GROUPS = ('selected', 'unselected')
N_FOLDS = 5  # `wins_vs_dino` counts the folds where Full SABLE beats DINO, so it is at most this
DEFAULT_POOL_SIZE = 20
LINE_SCALE = 1.0  # factor on the widths below, which are those of `neural_encoding_cls_token`
GT_WIDTH = 1.1  # ground truth
DINO_WIDTH = 3.0  # drawn at the back
WIDTH = 2.5  # the other model, drawn on top of DINO
HEIGHT_PER_NEURON = 0.4  # inches, 4 neurons give the 1.6 in height of `psth_stack_patch.py`


def load_wins_vs_dino(session: str, k: int, directory: Path) -> np.ndarray:
    """Load the `wins_vs_dino` column of the handoff csv of one session and K.

    Args:
        session: session name used in the csv file name.
        k: number of PCA components.
        directory: folder with the `session_<session>_k<K>_all_neurons.csv` files.

    Returns:
        Number of folds where Full SABLE beats DINO for every neuron, shape (neurons,), ordered
        by `neuron_index`.

    Raises:
        FileNotFoundError: if the csv of this session and K does not exist.
        ValueError: if the csv has no `wins_vs_dino` column.
    """
    path = directory / f'session_{session}_k{k}_all_neurons.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found')
    table = pd.read_csv(path)
    if 'wins_vs_dino' not in table.columns:
        raise ValueError(f'{path} has no column wins_vs_dino')
    return table.sort_values('neuron_index')['wins_vs_dino'].to_numpy(np.float64)


def split_neurons_by_selection(idx_selected: list[int], n_neurons: int) -> dict[str, list[int]]:
    """Split the neurons of a session into the selected and the unselected group.

    Args:
        idx_selected: indices of the neurons flagged `selected`.
        n_neurons: number of neurons of the session.

    Returns:
        The sorted neuron indices of each group in `GROUPS`.

    Raises:
        ValueError: if a selected index is outside `range(n_neurons)`.
    """
    if any(not 0 <= idx < n_neurons for idx in idx_selected):
        raise ValueError(f'selected neuron indices must be within 0..{n_neurons - 1}')
    set_selected = set(idx_selected)
    return {
        'selected': sorted(set_selected),
        'unselected': [idx for idx in range(n_neurons) if idx not in set_selected],
    }


def rank_by_selection_bps(
    bps: np.ndarray,
    wins_vs_dino: np.ndarray,
    psth_corr: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sort the neurons by the csv BPS and attach the columns the selection filters on.

    Args:
        bps: csv BPS of every neuron, shape (neurons,), NaN where `valid_bps == 0`.
        wins_vs_dino: folds where Full SABLE beats DINO, shape (neurons,).
        psth_corr: per-neuron PSTH correlation, shape (neurons,). NaN for every neuron if None.

    Returns:
        One row per neuron, by decreasing BPS with NaN last, with columns `neuron_idx`, `bps`,
        `wins_vs_dino` and `psth_corr`.
    """
    bps = np.asarray(bps, dtype=np.float64)
    table = pd.DataFrame(
        {
            'neuron_idx': np.arange(bps.size),
            'bps': bps,
            'wins_vs_dino': np.asarray(wins_vs_dino, dtype=np.float64),
            'psth_corr': np.full(bps.shape, np.nan) if psth_corr is None else psth_corr,
        },
    )
    return table.sort_values('bps', ascending=False, na_position='last', kind='stable')


def top_neurons_in_pool(
    table: pd.DataFrame,
    idx_group: list[int],
    num_neurons: int,
    *,
    min_wins: int,
    pool_size: int,
) -> tuple[list[int], int]:
    """Top-BPS neurons of one group among those with many wins and a well predicted PSTH.

    Args:
        table: one row per neuron, by decreasing BPS, see `rank_by_selection_bps`.
        idx_group: neuron indices of the group.
        num_neurons: number of neurons to return.
        min_wins: minimum `wins_vs_dino`, 0 to skip the filter.
        pool_size: number of neurons with the highest `psth_corr` kept after the wins filter,
            0 to skip the pool. Neurons with a NaN `psth_corr` are not kept.

    Returns:
        Neuron indices by decreasing BPS, shorter than `num_neurons` if the pool is small, and the
        number of neurons of the group that passed the wins filter and have a BPS.
    """
    passed = table[
        table['neuron_idx'].isin(idx_group)
        & table['bps'].notna()
        & (table['wins_vs_dino'] >= min_wins)
    ]
    pool = passed
    if pool_size:
        # nlargest drops the BPS order, so the pool is sorted again
        pool = passed.nlargest(pool_size, 'psth_corr').sort_values(
            'bps',
            ascending=False,
            kind='stable',
        )
    return [int(idx) for idx in pool['neuron_idx'].iloc[:num_neurons]], len(passed)


def plot_session(args: argparse.Namespace, session: str) -> None:
    """Write the selected and the unselected PSTH stack of one session.

    Args:
        args: parsed arguments of the script.
        session: session name, a key of `SESSION_EIDS`.

    Raises:
        ValueError: if the csv and the data have a different number of neurons.
    """
    spikes = load_session_spikes(SESSION_EIDS[session], args.neural_dir)
    # the held-out trials, and so the ground truth, are the same for every model
    gt, preds = None, []
    for model in args.model:
        _, gt, pred = load_ridge_predictions(session, model, args.k, args.fold, spikes)
        preds.append(pred)
    t0, t1 = trials_idx_to_bounds(gt.shape[0], args.trial_idx)
    gt, preds = gt[t0:t1], [pred[t0:t1] for pred in preds]

    records, idx_selected = load_selection_records(session, args.k, args.selection_dir)
    groups = split_neurons_by_selection(idx_selected, gt.shape[2])
    # csv BPS of each model, averaged over the folds, NaN for the neurons without a valid BPS
    bps_by_model = [records[('ridge', session, model, args.k, 0)].bps for model in args.model]
    wins_vs_dino = load_wins_vs_dino(session, args.k, args.selection_dir)
    if wins_vs_dino.size != gt.shape[2]:
        raise ValueError(
            f'session {session}: the csv has {wins_vs_dino.size} neurons, the data {gt.shape[2]}'
        )
    psth_corr, modulation = None, None
    if args.pool_size:
        path = ranking_path(session, args.model[-1], args.k, args.fold, args.ranking_dir)
        saved = load_ranking(
            path,
            {'session': session, 'model': args.model[-1], 'k': args.k, 'fold': args.fold},
            gt.shape[2],
        ).set_index('neuron_idx')
        psth_corr = saved['psth_corr'].sort_index().to_numpy()
        modulation = saved['modulation'].sort_index().to_numpy()
    table = rank_by_selection_bps(bps_by_model[-1], wins_vs_dino, psth_corr)

    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_psth_stack_selected',
        'psth_stack.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f'psth_stack_{session}_{"_".join(args.model)}_k{args.k}_fold{args.fold}'
    for group, idx_group in groups.items():
        neuron_indices, n_passed = top_neurons_in_pool(
            table,
            idx_group,
            args.num_neurons,
            min_wins=args.min_wins,
            pool_size=args.pool_size,
        )
        print(
            f'session {session}, {group} group: {len(idx_group)} neurons, {n_passed} with '
            f'wins_vs_dino >= {args.min_wins} and a BPS, '
            f'pool of {min(args.pool_size or n_passed, n_passed)}'
        )
        if not neuron_indices:
            print(f'{group}: no neuron left, skipped (lower --min-wins or use --pool-size 0)')
            continue
        print(f'top neurons of the {group} group: {neuron_indices}')
        save_path = save_dir / f'{stem}_{group}.png'
        bps_text = format_bps_text(
            neuron_indices,
            bps_by_model,
            args.model,
            session=session,
            k=args.k,
            fold=args.fold,
            psth_corr=psth_corr,
            modulation=modulation,
        )
        # the BPS is the csv mean over the folds, only the PSTH metrics belong to the fold
        bps_text = bps_text.replace(
            f', fold {args.fold}\n',
            f', {group} group, mean BPS over folds, PSTHs of fold {args.fold}\n',
            1,
        )
        # one line per neuron follows the two header lines, in the order of `neuron_indices`
        lines = bps_text.splitlines()
        lines[2:] = [
            f'{line} | wins_vs_dino={int(wins_vs_dino[idx])}'
            for line, idx in zip(lines[2:], neuron_indices, strict=True)
        ]
        bps_text = '\n'.join(lines) + '\n'
        save_path.with_suffix('.txt').write_text(bps_text)
        print(bps_text)
        # DINO goes first so it is drawn at the back, whatever the order of --model
        order = sorted(range(len(args.model)), key=lambda idx: args.model[idx] != 'dino')
        plot_psth_stack(
            gt,
            [preds[idx] for idx in order],
            neuron_indices,
            colors=[model_color(args.model[idx]) for idx in order],
            normalize=not args.no_normalize,
            figsize=(args.width, args.height_per_neuron * len(neuron_indices)),
            line_scale=args.line_scale,
            widths=[DINO_WIDTH if args.model[idx] == 'dino' else WIDTH for idx in order],
            gt_width=GT_WIDTH,
            label_neurons=True,
            save_path=save_path,
            also_save_pdf=args.save_pdf,
            show=not args.no_show,
        )
        print(save_path)


def main() -> None:
    """Write one PSTH stack of the top-BPS neurons per group and session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--session',
        nargs='+',
        default=['78'],
        choices=SESSION_EIDS,
        help='one or more sessions, e.g. --session 78 72 4b',
    )
    parser.add_argument(
        '--model',
        nargs='+',
        default=['full'],
        choices=MODELS,
        help='one or two models, e.g. --model dino full; neurons are ranked by the last one',
    )
    parser.add_argument('--k', type=int, default=6, choices=SELECTION_KS)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--num-neurons', type=int, default=4, help='top-BPS neurons per group')
    parser.add_argument(
        '--trial-idx',
        type=parse_trials_idx_arg,
        default=None,
        metavar='N | K,N',
        help='0-based trials: N alone = first N trials; two values = indices K..N inclusive',
    )
    parser.add_argument('--selection-dir', type=Path, default=SELECTION_DIR)
    parser.add_argument('--ranking-dir', type=Path, default=RANKING_DIR)
    parser.add_argument(
        '--min-wins',
        type=int,
        default=N_FOLDS,
        choices=range(N_FOLDS + 1),
        help='minimum number of folds where Full SABLE beats DINO (wins_vs_dino), 0 to skip',
    )
    parser.add_argument(
        '--pool-size',
        type=int,
        default=DEFAULT_POOL_SIZE,
        help=(
            'neurons with the highest PSTH correlation kept per group before the top BPS are '
            'taken, 0 to skip and not read the ranking file'
        ),
    )
    parser.add_argument('--no-normalize', action='store_true', help='plot raw PSTH values')
    parser.add_argument('--width', type=float, default=1.0, help='figure width in inches')
    parser.add_argument(
        '--height-per-neuron',
        type=float,
        default=HEIGHT_PER_NEURON,
        help='figure height in inches of every stacked neuron, so more neurons extend the figure',
    )
    parser.add_argument(
        '--line-scale',
        type=float,
        default=LINE_SCALE,
        help='factor on the line widths, which match `neural_encoding_cls_token` at 1',
    )
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if len(args.model) > 2:
        parser.error('--model takes one or two models')

    for session in dict.fromkeys(args.session):
        plot_session(args, session)


if __name__ == '__main__':
    main()
