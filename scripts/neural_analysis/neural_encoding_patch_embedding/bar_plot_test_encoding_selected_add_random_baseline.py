#!/usr/bin/env python3
"""Held-out BPS bar plot of the selected neurons with the frame-permutation random baseline.

Same bar plot as `bar_selected_k<K>.png` of `bar_plot_test_encoding_selected.py`, restricted to the
neurons flagged `selected` in the per-neuron csv files of `neuron_selection_handoff/`
(`session_<session>_k<K>_all_neurons.csv`), plus the K=6 frame-permutation random baseline as the
leftmost bar. The plot has no session markers, lines or legend. The baseline does not exist for
the other K, whose plots have no random bar. It is read per fold from the random baseline folder,
averaged over the five folds like the csv BPS and restricted to the selected neurons with a valid
BPS.

The selection favors neurons where Full SABLE beats DINO and NoGeom SABLE and was made on the same
cross-validation folds the BPS comes from, while the random baseline was not part of it. The bars
describe the selected neurons and are not an unbiased estimate of the Full SABLE advantage.

Per K it writes `bar_selected_k<K>.png` and `bar_selected_k<K>.txt`, the markdown table with the
mean and SEM of every session and of the pooled sessions behind the bars.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_test_encoding_selected_add_random_baseline.py \
  --k 6 --no-show
```
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.bar_plot_encoding_figure3 import (  # noqa: E402, E501
    plot_encoding_bars,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_encoding_patch import (  # noqa: E402, E501
    BASELINE_MODEL,
    bar_models,
    build_markdown_table,
    save_figure,
    summarize_group,
    unit_bars,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_test_encoding_selected import (  # noqa: E402, E501
    FOLDS_IN_CSV,
    GROUP,
    SELECTION_KS,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    NEURAL_DIR,
    RANDOM_BASELINE_K,
    RANDOM_BASELINE_ROOTS,
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
    Record,
    RecordKey,
    load_random_baseline_records,
    load_selection_records,
    load_session_spikes,
    model_color,
    restrict_records,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402


def mask_random_records(
    records_random: dict[RecordKey, Record],
    records_selection: dict[RecordKey, Record],
) -> dict[RecordKey, Record]:
    """Make the random baseline NaN where the csv BPS of the same session is invalid.

    The csv marks neurons without a valid BPS with `valid_bps == 0`, which `load_selection_records`
    turns into NaN. The baseline gets the same neurons so that all bars of a session describe the
    same set of neurons.

    Args:
        records_random: output of `load_random_baseline_records`, one record per fold.
        records_selection: output of `load_selection_records`, whose NaN neurons are invalid.

    Returns:
        New baseline records with the same keys.

    Raises:
        ValueError: if the baseline and the csv of a session have different numbers of neurons.
    """
    masked: dict[RecordKey, Record] = {}
    for key, record in records_random.items():
        key_reference = (key[0], key[1], BASELINE_MODEL, key[3], 0)
        valid = np.isfinite(records_selection[key_reference].bps)
        if record.bps.shape != valid.shape:
            raise ValueError(
                f'session {key[1]} has {record.bps.shape[0]} neurons in the random baseline but '
                f'{valid.shape[0]} in the selection csv',
            )
        masked[key] = Record(bps=np.where(valid, record.bps, np.nan), trial_ids=record.trial_ids)
    return masked


def plot_bars(
    rows: list[dict],
    args: argparse.Namespace,
    save_dir: Path,
    stem: str,
) -> None:
    """Write the pooled bar plot, without session markers, and the markdown table.

    Args:
        rows: output of `summarize_group` of the selected neurons.
        args: parsed arguments of the script, with `k`, `sessions`, `folds`, `pool` and `no_show`.
        save_dir: output folder.
        stem: file name without extension.
    """
    models = bar_models(rows, args.k)
    means, errs, _ = unit_bars(rows, 'all', args.k, models)
    fig, ax = plt.subplots(figsize=(2.6, 2.2), dpi=200)
    plot_encoding_bars(
        [MODEL_LABELS[model] for model in models],
        np.full_like(means, np.nan),
        means,
        np.full_like(errs, np.nan),
        errs,
        [(model_color(model), model_color(model)) for model in models],
        plot_encoders='cnn',
        xtick_color='black',
        ax=ax,
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_dir / f'{stem}.png', args.no_show)
    table = build_markdown_table(
        rows,
        [*args.sessions, 'all'],
        args.k,
        args.folds,
        GROUP,
        args.pool,
    )
    (save_dir / f'{stem}.txt').write_text(table)
    print(save_dir / f'{stem}.png')
    print(table)


def plot_selected_random_k(args: argparse.Namespace, k: int, save_dir: Path) -> None:
    """Write the plot and table of the selected neurons at one K, with the random baseline.

    Args:
        args: parsed arguments of the script.
        k: number of PCA components.
        save_dir: output folder.
    """
    records: dict[RecordKey, Record] = {}
    idx_selected: dict[str, list[int]] = {}
    for session in args.sessions:
        records_session, idx_selected[session] = load_selection_records(
            session,
            k,
            args.selection_dir,
        )
        records.update(records_session)
        if k == RANDOM_BASELINE_K and not args.no_random_baseline:
            spikes = load_session_spikes(SESSION_EIDS[session], args.neural_dir)
            records_random = load_random_baseline_records(
                session,
                RANDOM_BASELINE_ROOTS[session],
                spikes,
            )
            records.update(mask_random_records(records_random, records_session))
    records = restrict_records(records, idx_selected)
    # the summary reads k, folds and pool from the arguments
    args_k = argparse.Namespace(**{**vars(args), 'k': k, 'folds': FOLDS_IN_CSV})
    plot_bars(summarize_group(records, args_k), args_k, save_dir, f'bar_selected_k{k}')


def main() -> None:
    """Write the bar plot and markdown table of the selected neurons per K."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument(
        '--k',
        type=int,
        nargs='+',
        default=list(SELECTION_KS),
        choices=SELECTION_KS,
    )
    parser.add_argument('--selection-dir', type=Path, default=SELECTION_DIR)
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument(
        '--pool',
        choices=('session', 'neuron'),
        default='session',
        help=(
            'how the sessions are pooled: "session" averages the session means and takes the SEM '
            'across sessions, "neuron" pools all neurons so larger sessions weigh more'
        ),
    )
    parser.add_argument(
        '--no-random-baseline',
        action='store_true',
        help=f'leave out the random baseline bar, which only exists for K={RANDOM_BASELINE_K}',
    )
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_barplot_selected_random',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        plot_selected_random_k(args, k, save_dir)


if __name__ == '__main__':
    main()
