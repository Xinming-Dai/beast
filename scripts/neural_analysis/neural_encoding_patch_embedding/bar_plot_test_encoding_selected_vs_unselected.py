#!/usr/bin/env python3
"""Held-out BPS bar plots of the selected and unselected neurons of the patch-embedding encoders.

Extends `bar_plot_encoding_selected.py`, which only draws the neurons flagged `selected` in the
per-neuron csv files of `neuron_selection_handoff/` (`session_<session>_k<K>_all_neurons.csv`),
with the complementary group: the neurons that are not selected but have a valid BPS. Neurons with
`valid_bps == 0` belong to neither group.

Per K and group (`selected`, `unselected`) it writes `bar_<group>_k<K>.png` and `.txt` (absolute
BPS) and `bar_<group>_k<K>_delta.png` and `.txt` (per-neuron BPS minus DINO). It also writes
`bar_selected_vs_unselected_k<K>_sessions.png`, a figure with a panel per session and one row per
group, and `bar_selected_vs_unselected_k<K>_delta.png`, the two delta bar plots side by side on a
shared y axis.

The selection favors neurons where Full SABLE beats DINO and NoGeom SABLE and was made on the same
cross-validation folds the BPS comes from. The selected neurons therefore overstate the Full SABLE
advantage and the unselected neurons understate it, so neither group is an unbiased estimate.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_test_encoding_selected_vs_unselected.py \
  --k 6 32 --no-show
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
    SESSION_MARKERS,
    bar_models,
    plot_group,
    plot_sessions_grid,
    save_figure,
    summarize_group,
    unit_bars,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_test_encoding_selected import (  # noqa: E402, E501
    FOLDS_IN_CSV,
    SELECTION_KS,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
    Record,
    RecordKey,
    load_selection_records,
    model_color,
    restrict_records,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

GROUPS = {
    'selected': 'selected neurons of neuron_selection_handoff',
    'unselected': 'unselected neurons of neuron_selection_handoff',
}
GROUP_ROW_LABELS = {'unselected': 'Unselected neurons', 'selected': 'Selected neurons'}


def unselected_indices(
    records: dict[RecordKey, Record],
    idx_selected: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Indices of the neurons of every session that are not selected.

    Neurons with an invalid BPS are NaN in the records, so `restrict_records` and `summarize`
    drop them from the unselected group without further filtering.

    Args:
        records: records of one K, as returned by `load_selection_records`.
        idx_selected: session name to the indices of the selected neurons.

    Returns:
        Session name to the sorted indices of the neurons that are not selected.
    """
    n_neurons = {key[1]: record.bps.shape[0] for key, record in records.items()}
    return {
        session: sorted(set(range(n)) - set(idx_selected[session]))
        for session, n in n_neurons.items()
    }


def plot_delta_side_by_side(
    rows_by_group: dict[str, list[dict]],
    args: argparse.Namespace,
    save_path: Path,
) -> None:
    """Write the pooled delta bar plots of the groups side by side on a shared y axis.

    Args:
        rows_by_group: group name to the output of `summarize_group` with `BASELINE_MODEL`.
        args: parsed arguments of the script, with `k`, `sessions` and `no_show`.
        save_path: output image path.
    """
    fig, axes = plt.subplots(
        1,
        len(rows_by_group),
        figsize=(3.2 * len(rows_by_group), 2.4),
        dpi=200,
        sharey=True,
        squeeze=False,
    )
    for ax, (group, rows) in zip(axes[0], rows_by_group.items()):
        models = bar_models(rows, args.k, BASELINE_MODEL)
        means, errs, n_neurons = unit_bars(rows, 'all', args.k, models)
        plot_encoding_bars(
            [MODEL_LABELS[model] for model in models],
            np.full_like(means, np.nan),
            means,
            np.full_like(errs, np.nan),
            errs,
            [(model_color(model), model_color(model)) for model in models],
            plot_encoders='cnn',
            y_label=f'Δ BPS vs {MODEL_LABELS[BASELINE_MODEL]}',
            xtick_color='black',
            ax=ax,
        )
        # one marker per session, joined across models, so the session offset is visible
        x_bars = [patch.get_x() + patch.get_width() / 2 for patch in ax.patches]
        for session, marker in zip(args.sessions, SESSION_MARKERS):
            means_session, _, _ = unit_bars(rows, session, args.k, models)
            ax.plot(
                x_bars,
                means_session,
                marker=marker,
                color='black',
                markerfacecolor='white',
                markersize=3.5,
                linewidth=0.6,
                alpha=0.8,
                label=f'Session {session}',
            )
        ax.set_title(GROUP_ROW_LABELS[group], fontsize=10, fontweight='medium')
    for ax in axes[0, 1:]:
        ax.set_ylabel('')
    axes[0, 0].legend(
        loc='upper left',
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=7,
        handlelength=1.5,
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_path, args.no_show)
    print(save_path)


def plot_selected_vs_unselected_k(args: argparse.Namespace, k: int, save_dir: Path) -> None:
    """Write the plots and tables of the selected and the unselected neurons at one K.

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
    idx_by_group = {
        'selected': idx_selected,
        'unselected': unselected_indices(records, idx_selected),
    }
    # the reused plotting functions read k, sessions, folds, pool and no_show from the arguments
    args_k = argparse.Namespace(**{**vars(args), 'k': k, 'folds': FOLDS_IN_CSV})
    rows_by_group: dict[str, list[dict]] = {}
    rows_delta_by_group: dict[str, list[dict]] = {}
    for group, idx_by_session in idx_by_group.items():
        records_group = restrict_records(records, idx_by_session)
        stem = f'bar_{group}_k{k}'
        rows_by_group[group] = summarize_group(records_group, args_k)
        rows_delta_by_group[group] = summarize_group(records_group, args_k, BASELINE_MODEL)
        plot_group(rows_by_group[group], args_k, save_dir, stem, GROUPS[group])
        plot_group(
            rows_delta_by_group[group],
            args_k,
            save_dir,
            f'{stem}_delta',
            GROUPS[group],
            BASELINE_MODEL,
        )
    stem_both = f'bar_selected_vs_unselected_k{k}'
    plot_sessions_grid(
        {GROUP_ROW_LABELS[group]: rows_by_group[group] for group in GROUP_ROW_LABELS},
        args_k,
        save_dir / f'{stem_both}_sessions.png',
    )
    plot_delta_side_by_side(
        {group: rows_delta_by_group[group] for group in GROUP_ROW_LABELS},
        args_k,
        save_dir / f'{stem_both}_delta.png',
    )


def main() -> None:
    """Write the bar plots and markdown tables of the selected and unselected neurons per K."""
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
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_barplot_selected_vs_unselected',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        plot_selected_vs_unselected_k(args, k, save_dir)


if __name__ == '__main__':
    main()
