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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (
    MODEL_LABELS,
    MODELS,
    NEURAL_DIR,
    PCA_KS,
    RIDGE_ROOTS,
    RESULTS_DIR,
    SESSION_EIDS,
    TCN_ROOTS,
    Y_LABEL,
    Record,
    RecordKey,
    check_matching_trials,
    load_ridge_records,
    load_session_spikes,
    load_tcn_records,
    model_color,
    summarize,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper

METHOD_STYLES = {
    'ridge': {'label': 'ridge', 'linestyle': '-', 'marker': 'o', 'offset': 0.0},
    'tcn': {'label': 'TCN', 'linestyle': '--', 'marker': 's', 'offset': 0.12},
}


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
                x = np.array([PCA_KS.index(r['k']) for r in sub]) + style['offset']
                ax.errorbar(
                    x,
                    [r['mean'] for r in sub],
                    yerr=[r['err'] for r in sub],
                    color=model_color(model),
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    linewidth=2,
                    markersize=5,
                    capsize=3,
                    label=f'{MODEL_LABELS[model]} ({style["label"]})',
                )
        ax.set_xticks(range(len(PCA_KS)), [str(k) for k in PCA_KS])
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
