#!/usr/bin/env python3
"""Held-out BPS of the whisker pad and paws ROIs for the 3D tuned neurons and the rest.

Grouped bar plot with one group of bars per ROI, whisker pad and paws, and in each group a bar for
the 3D tuned (selected) neurons and one for The Rest (the neurons not selected in the per-neuron
csv files of `neuron_selection_handoff/`). A bar is the mean held-out BPS over the neurons of the
group, averaged over the five folds, with the SEM across neurons as error bar. The ROI results
are the body-part ridge runs, where Lightning Pose keypoints select the SABLE patch tokens around
the whisker pad (the `head` ROI folder) and the paws (`paws_concat` by default). Neurons without a
valid BPS in the csv are left out of both groups. The bar style follows
`bar_plot_test_encoding_selected_add_random_baseline.py`.

It writes `Roi_bar_plot_s<session>_k<K>.png` and, next to it, `Roi_bar_plot_s<session>_k<K>.txt`
with two markdown tables: the mean, SEM and number of neurons behind the bars, and the test of
3D tuned against The Rest for every ROI. The test is a two-sided Welch t-test (unequal variances)
on the per-neuron BPS, with a two-sided Mann-Whitney U test as a check against the skewed BPS
distribution. Both p-values are Holm-corrected over the two ROIs.

The selection favors neurons where Full SABLE beats DINO and NoGeom SABLE and was made on the same
cross-validation folds the BPS comes from, so the gap between the two groups is partly built in and
the p-values are optimistic.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/Roi_bar_plot.py --k 6 --no-show
```
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.bar_plot_encoding_figure3 import (  # noqa: E402, E501
    BAR_FONT_SIZE,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_encoding_patch import (  # noqa: E402, E501
    save_figure,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_test_encoding_selected import (  # noqa: E402, E501
    SELECTION_KS,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
    Y_LABEL,
    model_color,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.ROI_preference_composition import (  # noqa: E402, E501
    DEFAULT_PAWS_ROI,
    GROUP_LABELS,
    PARTS,
    load_roi_groups,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

# left to right in the figure
ROIS = ('head', 'paws')
ROI_LABELS = dict(PARTS)
# the two bars of every ROI, left to right
BAR_GROUPS = ('selected', 'unselected')
# 3D tuned neurons in the Full SABLE color, The Rest in the DINO color
GROUP_COLORS = {'selected': model_color('full'), 'unselected': model_color('dino')}
BAR_WIDTH = 0.3
ROI_SPACING = 1.0  # distance between the centers of two ROI groups


def _valid_group_bps(
    bps_by_part: dict[str, np.ndarray],
    idx_neurons: list[int] | np.ndarray,
) -> np.ndarray:
    """Per-neuron BPS of the ROIs and the whole frame for the neurons of a group with a valid BPS.

    Args:
        bps_by_part: per-neuron BPS of shape (neurons,) under the keys of `ROIS` and `'whole'`,
            NaN for neurons without a valid BPS.
        idx_neurons: indices of the neurons of the group.

    Returns:
        Array of shape (len(ROIS) + 1, neurons kept), one row per ROI in the order of `ROIS`
        followed by the whole frame. Neurons with a NaN BPS in any row are dropped from every row,
        so all bars and tests of a group describe the same neurons.

    Raises:
        ValueError: if the group has no neuron with a valid BPS.
    """
    stacked = np.stack([bps_by_part[part] for part in (*ROIS, 'whole')])[:, idx_neurons]
    stacked = stacked[:, np.isfinite(stacked).all(axis=0)]
    if stacked.shape[1] == 0:
        raise ValueError('the group has no neuron with a valid BPS')
    return stacked


def roi_stats(
    bps_by_part: dict[str, np.ndarray],
    idx_neurons: list[int] | np.ndarray,
) -> dict[str, float]:
    """Mean BPS and SEM across neurons of every ROI for one group of neurons.

    Args:
        bps_by_part: per-neuron BPS of shape (neurons,) under the keys of `ROIS` and `'whole'`,
            NaN for neurons without a valid BPS.
        idx_neurons: indices of the neurons of the group.

    Returns:
        `<roi>_mean` and `<roi>_sem` for every ROI in `ROIS` and `'n_neurons'`. Neurons with a NaN
        BPS in any ROI or in the whole frame are dropped from every ROI, so all bars of a group
        describe the same neurons. The SEM uses ddof=1 and is 0 for a single neuron.

    Raises:
        ValueError: if the group has no neuron with a valid BPS.
    """
    stacked = _valid_group_bps(bps_by_part, idx_neurons)
    n_neurons = stacked.shape[1]
    stats: dict[str, float] = {'n_neurons': n_neurons}
    for roi, bps in zip(ROIS, stacked):
        stats[f'{roi}_mean'] = float(bps.mean())
        stats[f'{roi}_sem'] = float(bps.std(ddof=1) / np.sqrt(n_neurons)) if n_neurons > 1 else 0.0
    return stats


def plot_roi_bars(
    stats_by_group: dict[str, dict[str, float]],
    save_path: Path,
    no_show: bool,
) -> None:
    """Write the grouped bar plot of the mean BPS of every ROI and neuron group.

    Args:
        stats_by_group: group key of `BAR_GROUPS` to the output of `roi_stats`.
        save_path: output image path.
        no_show: close the figure instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(2.6, 2.2), dpi=200)
    x_rois = np.arange(len(ROIS)) * ROI_SPACING
    tops = []
    for idx_group, group in enumerate(BAR_GROUPS):
        stats = stats_by_group[group]
        means = np.array([stats[f'{roi}_mean'] for roi in ROIS])
        errs = np.array([stats[f'{roi}_sem'] for roi in ROIS])
        x_bars = x_rois + (idx_group - (len(BAR_GROUPS) - 1) / 2) * (BAR_WIDTH + 0.03)
        ax.bar(
            x_bars,
            means,
            BAR_WIDTH,
            yerr=errs,
            color=GROUP_COLORS[group],
            edgecolor='none',
            capsize=2,
            error_kw={'elinewidth': 0.8},
            label=GROUP_LABELS[group],
        )
        for x_bar, mean, err in zip(x_bars, means, errs):
            # the offset is in points so the label hugs the error bar at any y scale
            ax.annotate(
                f'{mean:.3f}',
                (x_bar, mean + err),
                xytext=(0, 2),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=BAR_FONT_SIZE - 1,
            )
        tops += list(means + errs)
    ax.set_xticks(x_rois)
    ax.set_xticklabels([ROI_LABELS[roi] for roi in ROIS], fontsize=BAR_FONT_SIZE)
    ax.set_ylabel(Y_LABEL, fontsize=BAR_FONT_SIZE)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis='y', labelsize=BAR_FONT_SIZE, width=1.5, length=7, direction='out')
    ax.tick_params(axis='x', length=0, pad=2)
    # headroom for the value labels and the legend above the tallest bar
    ax.set_ylim(0, max(tops) * 1.4)
    ax.set_xlim(x_rois.min() - 0.6, x_rois.max() + 0.6)
    ax.grid(False)
    for side in ('top', 'right', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.axhline(0, color='black', linewidth=1.5, zorder=0.5)
    ax.legend(
        loc='upper right',
        frameon=False,
        ncol=1,
        fontsize=8,
        handlelength=1.2,
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_path, no_show)
    print(save_path)


def build_markdown_table(
    stats_by_group: dict[str, dict[str, float]],
    session: str,
    k: int,
    paws_roi: str,
) -> str:
    """Markdown table of the mean BPS and SEM behind the bars.

    Args:
        stats_by_group: group key of `BAR_GROUPS` to the output of `roi_stats`.
        session: session name.
        k: number of PCA components.
        paws_roi: name of the ROI used for Paws.

    Returns:
        The table as text, one row per group.
    """
    lines = [
        f'Mean held-out BPS +/- SEM across neurons, session {session}, K={k}, paws = {paws_roi}',
        '',
        '| group | neurons | whisker pad BPS | paws BPS |',
        '|---|---|---|---|',
    ]
    for group in BAR_GROUPS:
        stats = stats_by_group[group]
        lines.append(
            f'| {GROUP_LABELS[group]} | {stats["n_neurons"]} '
            f'| {stats["head_mean"]:.3f} +/- {stats["head_sem"]:.3f} '
            f'| {stats["paws_mean"]:.3f} +/- {stats["paws_sem"]:.3f} |',
        )
    return '\n'.join(lines) + '\n'


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values.

    Args:
        pvals: raw p-values of the family of comparisons, in any order.

    Returns:
        The adjusted p-values in the order of `pvals`. The smallest p-value is multiplied by the
        number of comparisons, the next by one less and so on, the running maximum keeps the
        adjusted values in the order of the raw ones, and values are capped at 1.
    """
    n_tests = len(pvals)
    adjusted = [0.0] * n_tests
    running_max = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        running_max = max(running_max, (n_tests - rank) * pvals[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def compare_groups(
    bps_by_part: dict[str, np.ndarray],
    idx_groups: dict[str, list[int]],
) -> dict[str, dict[str, float]]:
    """Test the per-neuron BPS of the 3D tuned neurons against The Rest for every ROI.

    Runs a two-sided Welch t-test (unequal variances) and a two-sided Mann-Whitney U test per ROI,
    on the same complete-case neurons as `roi_stats`. Each of the two p-values is Holm-corrected
    over the ROIs.

    Args:
        bps_by_part: per-neuron BPS of shape (neurons,) under the keys of `ROIS` and `'whole'`,
            NaN for neurons without a valid BPS.
        idx_groups: neuron indices under the keys `'selected'` and `'unselected'`.

    Returns:
        For every ROI in `ROIS`: `n_selected`, `n_rest`, `difference` (mean BPS of the selected
        minus The Rest), `t` and `p_welch` of the t-test, `u` and `p_mwu` of the Mann-Whitney test,
        and the Holm-adjusted `p_welch_holm` and `p_mwu_holm`.

    Raises:
        ValueError: if a group has no neuron with a valid BPS.
    """
    bps_selected = _valid_group_bps(bps_by_part, idx_groups['selected'])
    bps_rest = _valid_group_bps(bps_by_part, idx_groups['unselected'])
    results: dict[str, dict[str, float]] = {}
    for idx_roi, roi in enumerate(ROIS):
        selected, rest = bps_selected[idx_roi], bps_rest[idx_roi]
        welch = ttest_ind(selected, rest, equal_var=False, alternative='two-sided')
        mwu = mannwhitneyu(selected, rest, alternative='two-sided')
        results[roi] = {
            'n_selected': len(selected),
            'n_rest': len(rest),
            'difference': float(selected.mean() - rest.mean()),
            't': float(welch.statistic),
            'p_welch': float(welch.pvalue),
            'u': float(mwu.statistic),
            'p_mwu': float(mwu.pvalue),
        }
    for key in ('welch', 'mwu'):
        adjusted = holm_adjust([results[roi][f'p_{key}'] for roi in ROIS])
        for roi, p_adjusted in zip(ROIS, adjusted):
            results[roi][f'p_{key}_holm'] = p_adjusted
    return results


def build_test_table(
    results: dict[str, dict[str, float]],
    session: str,
    k: int,
    paws_roi: str,
) -> str:
    """Markdown table of the tests of 3D tuned against The Rest, with a note on how to read it.

    Args:
        results: output of `compare_groups`.
        session: session name.
        k: number of PCA components.
        paws_roi: name of the ROI used for Paws.

    Returns:
        The table and its note as text, one row per ROI.
    """
    lines = [
        f'3D-tuned vs The Rest, per-neuron held-out BPS, session {session}, K={k}, '
        f'paws = {paws_roi}',
        '',
        '| ROI | n 3D-tuned | n The Rest | difference | t | Welch p | Welch p (Holm) '
        '| U | Mann-Whitney p | Mann-Whitney p (Holm) |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for roi in ROIS:
        res = results[roi]
        lines.append(
            f'| {ROI_LABELS[roi]} | {res["n_selected"]} | {res["n_rest"]} '
            f'| {res["difference"]:.4f} | {res["t"]:.3f} | {res["p_welch"]:.4g} '
            f'| {res["p_welch_holm"]:.4g} | {res["u"]:.0f} | {res["p_mwu"]:.4g} '
            f'| {res["p_mwu_holm"]:.4g} |',
        )
    lines += [
        '',
        'difference = mean BPS of 3D-tuned minus The Rest. Welch: two-sided t-test with unequal '
        'variances. Mann-Whitney: two-sided U test, a check against the skewed BPS distribution. '
        'Holm: correction over the two ROIs.',
        'The neurons are the samples of a single session, and the selection of the 3D-tuned '
        'neurons was made on the same folds as the BPS, so the p-values are optimistic.',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the grouped bar plot and the markdown tables of one session and K."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='78', choices=SESSION_EIDS)
    parser.add_argument('--k', type=int, default=6, choices=SELECTION_KS)
    parser.add_argument('--roi-dir', type=Path, default=None)
    parser.add_argument('--paws-roi', default=DEFAULT_PAWS_ROI)
    parser.add_argument('--selection-dir', type=Path, default=SELECTION_DIR)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    bps_by_part, idx_groups = load_roi_groups(
        args.session,
        args.k,
        args.roi_dir,
        args.paws_roi,
        args.selection_dir,
    )
    stats_by_group = {group: roi_stats(bps_by_part, idx_groups[group]) for group in BAR_GROUPS}

    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'Roi_bar_plot',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f'Roi_bar_plot_s{args.session}_k{args.k}'
    test_results = compare_groups(bps_by_part, idx_groups)
    table = '\n'.join(
        [
            build_markdown_table(stats_by_group, args.session, args.k, args.paws_roi),
            build_test_table(test_results, args.session, args.k, args.paws_roi),
        ],
    )
    (save_dir / f'{stem}.txt').write_text(table)
    print(table)
    plot_roi_bars(stats_by_group, save_dir / f'{stem}.png', args.no_show)


if __name__ == '__main__':
    main()
