#!/usr/bin/env python3
"""Paper figure of the 2D tuned and 3D tuned neurons' BPS gain of the patch-embedding encoders.

Paper version of the `bar_selected_vs_unselected_k<K>_delta.png` figure of
`bar_plot_test_encoding_selected_vs_unselected.py`: the held-out BPS of each model minus DINO's,
per neuron, for the unselected neurons (2D tuned) and the selected neurons (3D tuned) of the
per-neuron csv files of `neuron_selection_handoff/`. Both groups share one panel with a pair of
bars per model, 2D tuned dashed and 3D tuned solid. The session means are not drawn.

For the given K it writes `bar_2d_vs_3d_tuned_k<K>.png` and, in the same style, one figure per
group, `bar_2d_tuned_k<K>.png` and `bar_3d_tuned_k<K>.png`. The group is only named in the file
name, the single-group figures have no title and no legend. Next to the group figures are
`bar_2d_tuned_k<K>.txt` and `bar_3d_tuned_k<K>.txt`, the markdown tables with the mean and SEM of
every session and of the pooled sessions behind the figures.

The selection favors neurons where Full SABLE beats DINO and NoGeom SABLE and was made on the same
cross-validation folds the BPS comes from. The 3D tuned neurons therefore overstate the Full SABLE
advantage and the 2D tuned neurons understate it, so neither group is an unbiased estimate.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_paper_encoding_selected_vs_unselected.py \
  --k 6 --no-show
```
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.bar_plot_encoding_figure3 import (  # noqa: E402, E501
    BAR_FONT_SIZE,
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
    SELECTION_KS,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_test_encoding_selected_vs_unselected import (  # noqa: E402, E501
    unselected_indices,
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

# left to right in the figure: group key, title, file stem and table description
GROUPS = {
    '2d': ('2D tuned', 'bar_2d_tuned', '2D tuned neurons (not selected in the handoff csv)'),
    '3d': ('3D tuned', 'bar_3d_tuned', '3D tuned neurons (selected in the handoff csv)'),
}


def plot_tuned_delta(
    rows_by_group: dict[str, list[dict]],
    args: argparse.Namespace,
    save_path: Path,
) -> None:
    """Write the pooled delta bar plot of one or both groups of neurons.

    Each model gets a bar per group, 2D tuned on the left with a dashed outline and a light fill,
    3D tuned on the right and solid, both in the color of the model. With a single group there is
    one wider bar per model and no legend, the group is named in the file name.

    Args:
        rows_by_group: group key of `GROUPS` to the output of `summarize_group` with
            `BASELINE_MODEL`, one or both groups in the order they are drawn.
        args: parsed arguments of the script, with `k` and `no_show`.
        save_path: output image path.
    """
    n_groups = len(rows_by_group)
    models = bar_models(next(iter(rows_by_group.values())), args.k, BASELINE_MODEL)
    width = 0.34 if n_groups == 2 else 0.5
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
    tops, bottoms = [0.0], [0.0]
    for idx_group, (group, rows) in enumerate(rows_by_group.items()):
        means, errs, _ = unit_bars(rows, 'all', args.k, models)
        x_bars = np.arange(len(models)) + (idx_group - (n_groups - 1) / 2) * (width + 0.04)
        colors = [model_color(model) for model in models]
        is_2d = group == '2d'
        ax.bar(
            x_bars,
            means,
            width,
            yerr=errs,
            color=colors if not is_2d else [to_rgba(color, 0.35) for color in colors],
            edgecolor=colors if is_2d else 'none',
            linestyle='--' if is_2d else '-',
            linewidth=1.0,
            capsize=2,
            error_kw={'elinewidth': 0.8},
        )
        for x_bar, mean, err in zip(x_bars, means, errs):
            # the offset is in points so the label hugs the error bar at any y scale
            ax.annotate(
                f'{mean:.3f}',
                (x_bar, mean + err if mean >= 0 else mean - err),
                xytext=(0, 2 if mean >= 0 else -2),
                textcoords='offset points',
                ha='center',
                va='bottom' if mean >= 0 else 'top',
                fontsize=BAR_FONT_SIZE,
            )
        tops += list(means + errs)
        bottoms += list(means - errs)
    ax.set_xticks(np.arange(len(models)))
    # the end of each label sits under its bar center
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in models],
        rotation=45,
        ha='right',
        rotation_mode='anchor',
        fontsize=BAR_FONT_SIZE,
    )
    ax.set_ylabel(f'Δ BPS vs {MODEL_LABELS[BASELINE_MODEL]}', fontsize=BAR_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=BAR_FONT_SIZE, width=1.5, length=5, direction='out')
    ax.tick_params(axis='x', length=0, pad=2)
    # headroom for the value labels above the tallest bar and below a negative one
    span = max(tops) - min(bottoms)
    ax.set_ylim(min(bottoms) - 0.12 * span if min(bottoms) < 0 else 0, max(tops) + 0.22 * span)
    ax.set_xlim(-0.6, len(models) - 0.4)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.axhline(0, color='black', linewidth=1.5, zorder=0.5)
    if n_groups == 2:
        ax.legend(
            handles=[
                Patch(
                    facecolor=to_rgba('gray', 0.35),
                    edgecolor='gray',
                    linestyle='--',
                    linewidth=1.0,
                    label=GROUPS['2d'][0],
                ),
                Patch(facecolor='gray', edgecolor='none', label=GROUPS['3d'][0]),
            ],
            loc='upper left',
            frameon=False,
            fontsize=8,
            handlelength=1.2,
        )
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_path, args.no_show)
    print(save_path)


def plot_paper_k(args: argparse.Namespace, save_dir: Path) -> None:
    """Write the combined figure, one figure per group and the tables at `args.k`.

    Args:
        args: parsed arguments of the script.
        save_dir: output folder.
    """
    records: dict[RecordKey, Record] = {}
    idx_selected: dict[str, list[int]] = {}
    for session in args.sessions:
        records_session, idx_selected[session] = load_selection_records(
            session,
            args.k,
            args.selection_dir,
        )
        records.update(records_session)
    idx_by_group = {'2d': unselected_indices(records, idx_selected), '3d': idx_selected}
    # the summary reads folds and pool from the arguments, the delta is always pooled over sessions
    args_k = argparse.Namespace(**{**vars(args), 'folds': FOLDS_IN_CSV, 'pool': 'session'})
    rows_by_group: dict[str, list[dict]] = {}
    for group, idx_by_session in idx_by_group.items():
        rows_by_group[group] = summarize_group(
            restrict_records(records, idx_by_session),
            args_k,
            BASELINE_MODEL,
        )
        table = build_markdown_table(
            rows_by_group[group],
            [*args.sessions, 'all'],
            args.k,
            FOLDS_IN_CSV,
            GROUPS[group][2],
            'session',
            BASELINE_MODEL,
        )
        (save_dir / f'{GROUPS[group][1]}_k{args.k}.txt').write_text(table)
        print(table)
    plot_tuned_delta(rows_by_group, args_k, save_dir / f'bar_2d_vs_3d_tuned_k{args.k}.png')
    for group, rows in rows_by_group.items():
        plot_tuned_delta({group: rows}, args_k, save_dir / f'{GROUPS[group][1]}_k{args.k}.png')


def main() -> None:
    """Write the paper figure and markdown tables of the 2D and 3D tuned neurons at one K."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument('--k', type=int, required=True, choices=SELECTION_KS)
    parser.add_argument('--selection-dir', type=Path, default=SELECTION_DIR)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_barplot_paper',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_paper_k(args, save_dir)


if __name__ == '__main__':
    main()
