#!/usr/bin/env python3
"""Held-out BPS bar plot of the patch-embedding ridge encoders at one K.

Patch-embedding counterpart of ``neural_encoding_cls_token/bar_plot_encoding_figure3.py``: one bar
per model (DINO, NoGeom SABLE, Full SABLE), with each session's mean overlaid as a marker.

BPS differs between sessions, so the pooling is selectable with ``--pool``:
- ``session`` (default): bar is the mean of the session means, error bar is the SEM across
  sessions, so every session counts once.
- ``neuron``: bar is the mean over all neurons, error bar is the SEM over neurons, so sessions
  with more neurons weigh more.

Also writes a second figure (``*_delta.png``) of each model's per-neuron BPS minus DINO's, which
removes the session offset, and a markdown table next to each figure.

```bash
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_for_bar_plot.py
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_encoding_patch.py \
  --k 6 --folds 0 --no-show
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_encoding_patch.py \
  --k 6 --folds 0 --no-show --split-frac 0.5
```
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.bar_plot_encoding_figure3 import (  # noqa: E402
    plot_encoding_bars,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    SESSION_EIDS,
    load_ridge_session_data,
    model_color,
    restrict_records,
    summarize,
    summarize_session_mean,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.rank_neurons_bps_for_bar_plot import (  # noqa: E402, E501
    RANKING_DIR,
    full_bps_ranking_path,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (  # noqa: E402
    load_ranking,
    split_by_rank,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

BAR_MODELS = ('dino', 'nogeom', 'full')  # bar order, left to right
BASELINE_MODEL = 'dino'  # model subtracted in the difference plot
SESSION_MARKERS = ('o', 's', '^', 'D', 'v')


def bar_models(baseline: str | None = None) -> tuple[str, ...]:
    """Models of the bars in order, without the `baseline` model that is subtracted from them."""
    return tuple(model for model in BAR_MODELS if model != baseline)


def build_markdown_table(
    rows: list[dict],
    units: list[str],
    k: int,
    folds: list[int],
    group: str | None = None,
    pool: str = 'neuron',
    baseline: str | None = None,
) -> str:
    """Markdown table of mean +/- SEM, one row per session plus the pooled row.

    Args:
        rows: output of `summarize_group`, per session and pooled.
        units: session names followed by `'all'`.
        k: number of PCA components.
        folds: folds the neurons were averaged over.
        group: description of the neuron group, appended to the title, None for all neurons.
        pool: how the `'all'` row was pooled, `'neuron'` or `'session'`.
        baseline: model subtracted from the others, None for the absolute BPS.

    Returns:
        The markdown text.
    """
    models = bar_models(baseline)
    header = ['Session', 'Neurons', *(MODEL_LABELS[model] for model in models)]
    what = 'Held-out BPS' if baseline is None else f'Held-out BPS minus {MODEL_LABELS[baseline]}'
    pooled_error = 'across sessions' if pool == 'session' else 'over neurons'
    title = (
        f'{what} (bits/spike), ridge, K={k}, fold(s) {folds}, mean +/- SEM over neurons per '
        f'session, pooled row: mean +/- SEM {pooled_error}'
    )
    lines = [
        title if group is None else f'{title}, {group}',
        '',
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(['---'] * len(header)) + ' |',
    ]
    for unit in units:
        by_model = {r['model']: r for r in rows if r['session'] == unit and r['k'] == k}
        n_neurons = by_model[models[0]]['n_neurons']
        cells = [f'{by_model[m]["mean"]:.4f} +/- {by_model[m]["err"]:.4f}' for m in models]
        name = 'All sessions' if unit == 'all' else unit
        lines.append('| ' + ' | '.join([name, str(n_neurons), *cells]) + ' |')
    return '\n'.join(lines) + '\n'


def split_neurons_by_ranking(
    records: dict,
    sessions: list[str],
    spikes_by_session: dict[str, np.ndarray],
    k: int,
    folds: list[int],
    frac: float,
    ranking_dir: Path,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Split the neurons of every session into the top `frac` by Full SABLE BPS and the rest.

    A single fold reads the ranking of that fold, several folds must be all folds on disk and read
    the ranking of the BPS averaged over them.

    Args:
        records: ridge records of all sessions.
        sessions: session names.
        spikes_by_session: spikes per session, shape (trials, time, neurons).
        k: number of PCA components.
        folds: folds the neurons are averaged over.
        frac: fraction of the neurons per session that goes to the 3D group.
        ranking_dir: folder written by `rank_neurons_bps_for_bar_plot.py`.

    Returns:
        Neuron indices per session of the 3D group and of the 2D group.

    Raises:
        ValueError: if several `folds` are not all the folds on disk of a session.
        FileNotFoundError: if the ranking of a session does not exist.
    """
    idx_3d: dict[str, list[int]] = {}
    idx_2d: dict[str, list[int]] = {}
    for session in sessions:
        folds_on_disk = sorted({key[4] for key in records if key[1:4] == (session, 'full', k)})
        if len(folds) == 1:
            folds_ranked = folds
        elif sorted(folds) == folds_on_disk:
            folds_ranked = None
        else:
            raise ValueError(
                f'rankings exist for a single fold or all folds {folds_on_disk}, got {folds}',
            )
        path = full_bps_ranking_path(session, k, folds_ranked, ranking_dir)
        if not path.exists():
            raise FileNotFoundError(f'{path} not found, run rank_neurons_bps_for_bar_plot.py first')
        ranking = load_ranking(
            path,
            {
                'session': session,
                'model': 'full',
                'k': k,
                'folds': ','.join(map(str, folds_ranked or folds_on_disk)),
            },
            n_neurons=spikes_by_session[session].shape[2],
        )
        idx_3d[session], idx_2d[session] = split_by_rank(ranking, frac)
    return idx_3d, idx_2d


def summarize_group(
    records: dict,
    args: argparse.Namespace,
    baseline: str | None = None,
) -> list[dict]:
    """Mean and SEM of every session and of the pooled sessions.

    Args:
        records: ridge records, restricted to one neuron group or not.
        args: parsed arguments of the script.
        baseline: model subtracted per neuron from the others, None for the absolute BPS. The
            difference is always pooled over sessions because it exists to cancel the session
            offset.

    Returns:
        Rows of `summarize` or `summarize_session_mean`, per session followed by the pooled
        `'all'` rows.
    """
    if args.pool == 'session' or baseline is not None:
        return summarize_session_mean(records, neuron_folds=args.folds, baseline=baseline)
    rows = []
    for pooled in (False, True):
        rows += summarize(records, pooled=pooled, error='neuron', neuron_folds=args.folds)
    return rows


def unit_bars(
    rows: list[dict],
    unit: str,
    k: int,
    models: tuple[str, ...] = BAR_MODELS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean, SEM and neuron count of one session (or `'all'`) for every bar model.

    Args:
        rows: output of `summarize_group`.
        unit: session name or `'all'`.
        k: number of PCA components.
        models: models of the bars in order.

    Returns:
        Means and SEMs ordered as `models`, and the number of neurons.
    """
    by_model = {r['model']: r for r in rows if r['session'] == unit and r['k'] == k}
    means = np.array([by_model[model]['mean'] for model in models])
    errs = np.array([by_model[model]['err'] for model in models])
    return means, errs, by_model[models[0]]['n_neurons']


def save_figure(fig: plt.Figure, save_path: Path, no_show: bool) -> None:
    """Save a figure and either show or close it.

    Args:
        fig: figure to save.
        save_path: output image path.
        no_show: close the figure instead of showing it.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.05, facecolor='white')
    if no_show:
        plt.close(fig)
    else:
        plt.show()


def plot_group(
    rows: list[dict],
    args: argparse.Namespace,
    save_dir: Path,
    stem: str,
    group: str | None = None,
    baseline: str | None = None,
) -> None:
    """Write the pooled bar plot with a marker per session and the markdown table.

    Args:
        rows: output of `summarize_group` of one neuron group, or of all neurons.
        args: parsed arguments of the script.
        save_dir: output folder.
        stem: file name without extension.
        group: description of the neuron group for the table title, None for all neurons.
        baseline: model subtracted from the others in `rows`, None for the absolute BPS.
    """
    models = bar_models(baseline)
    means, errs, _ = unit_bars(rows, 'all', args.k, models)
    fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=200)
    plot_encoding_bars(
        [MODEL_LABELS[model] for model in models],
        np.full_like(means, np.nan),
        means,
        np.full_like(errs, np.nan),
        errs,
        [(model_color(model), model_color(model)) for model in models],
        plot_encoders='cnn',
        y_label=None if baseline is None else f'Δ BPS vs {MODEL_LABELS[baseline]}',
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
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.0, 1.22),
        frameon=False,
        ncol=len(args.sessions),
        fontsize=7,
        handlelength=1.5,
        columnspacing=0.8,
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_dir / f'{stem}.png', args.no_show)
    table = build_markdown_table(
        rows,
        [*args.sessions, 'all'],
        args.k,
        args.folds,
        group,
        'session' if baseline is not None else args.pool,
        baseline,
    )
    (save_dir / f'{stem}.txt').write_text(table)
    print(save_dir / f'{stem}.png')
    print(table)


def plot_sessions_grid(
    rows_by_row: dict[str, list[dict]],
    args: argparse.Namespace,
    save_path: Path,
) -> None:
    """Write one figure with a bar panel per session, one row per neuron group.

    Args:
        rows_by_row: row label to the output of `summarize_group` of that neuron group, in the
            order the rows are drawn.
        args: parsed arguments of the script.
        save_path: output image path.
    """
    n_rows, n_cols = len(rows_by_row), len(args.sessions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        dpi=200,
        sharex=True,
        squeeze=False,
    )
    for idx_row, (row_label, rows) in enumerate(rows_by_row.items()):
        for idx_col, session in enumerate(args.sessions):
            ax = axes[idx_row, idx_col]
            means, errs, n_neurons = unit_bars(rows, session, args.k)
            plot_encoding_bars(
                [MODEL_LABELS[model] for model in BAR_MODELS],
                np.full_like(means, np.nan),
                means,
                np.full_like(errs, np.nan),
                errs,
                [(model_color(model), model_color(model)) for model in BAR_MODELS],
                plot_encoders='cnn',
                xtick_color='black',
                ax=ax,
            )
            if idx_row == 0:
                ax.set_title(f'Session {session}', fontsize=12, fontweight='medium')
            if idx_col == 0:
                ax.set_ylabel(f'{row_label}\nAvg BPS', fontsize=12, fontweight='medium')
            else:
                ax.set_ylabel('')
            ax.text(
                0.98,
                0.98,
                f'n={n_neurons}',
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=9,
            )
        # sessions of the same neuron group share the y range so they can be compared
        y_lims = [axes[idx_row, idx_col].get_ylim() for idx_col in range(n_cols)]
        for ax in axes[idx_row]:
            ax.set_ylim(min(lo for lo, _ in y_lims), max(hi for _, hi in y_lims))
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_path, args.no_show)
    print(save_path)


def main() -> None:
    """Write the pooled bar plot and the markdown table, for all neurons or per 3D / 2D group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument('--k', type=int, default=6, choices=PCA_KS)
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument(
        '--split-frac',
        type=float,
        default=None,
        help=(
            'fraction of the neurons of each session, ranked by Full SABLE BPS, that goes to the '
            '3D group, the rest is the 2D group; writes one plot and table per group and one '
            'figure with a panel per session (2D row, 3D row)'
        ),
    )
    parser.add_argument(
        '--ranking-dir',
        type=Path,
        default=RANKING_DIR,
        help='with --split-frac: folder written by rank_neurons_bps_for_bar_plot.py',
    )
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

    # only this K is read from disk, all folds are kept because the ranking split needs them
    records, spikes_by_session = load_ridge_session_data(
        args.sessions,
        args.neural_dir,
        ks=(args.k,),
    )
    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_barplot',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f'bar_k{args.k}_fold{"".join(map(str, args.folds))}'

    if args.split_frac is None:
        plot_group(summarize_group(records, args), args, save_dir, stem)
        plot_group(
            summarize_group(records, args, BASELINE_MODEL),
            args,
            save_dir,
            f'{stem}_delta',
            baseline=BASELINE_MODEL,
        )
        return
    idx_3d, idx_2d = split_neurons_by_ranking(
        records,
        args.sessions,
        spikes_by_session,
        args.k,
        args.folds,
        args.split_frac,
        args.ranking_dir,
    )
    pct_3d = round(100 * args.split_frac)
    groups = {
        '3d': (idx_3d, f'3D neurons (top {pct_3d}% by Full SABLE BPS)'),
        '2d': (idx_2d, f'2D neurons (remaining {100 - pct_3d}% by Full SABLE BPS)'),
    }
    # each group is summarized once, the rows feed the pooled plot, the table and the grid
    rows_by_suffix = {
        suffix: summarize_group(restrict_records(records, idx_by_session), args)
        for suffix, (idx_by_session, _) in groups.items()
    }
    for suffix, (idx_by_session, group) in groups.items():
        plot_group(rows_by_suffix[suffix], args, save_dir, f'{stem}_{suffix}', group)
        plot_group(
            summarize_group(restrict_records(records, idx_by_session), args, BASELINE_MODEL),
            args,
            save_dir,
            f'{stem}_{suffix}_delta',
            group,
            BASELINE_MODEL,
        )
    plot_sessions_grid(
        {'2D neurons': rows_by_suffix['2d'], '3D neurons': rows_by_suffix['3d']},
        args,
        save_dir / f'{stem}_sessions.png',
    )


if __name__ == '__main__':
    main()
