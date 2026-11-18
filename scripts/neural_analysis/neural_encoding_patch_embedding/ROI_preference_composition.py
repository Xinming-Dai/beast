#!/usr/bin/env python3
"""Percentage of reliably encoded neurons assigned to each ROI preference.

Stacked bar plot of the whisker pad, paws and other share of the encoding of 3D tuned neurons.

Reads the body-part ROI ridge results (Lightning Pose keypoints select the SABLE patch tokens
around the whisker pad and the paws, `bodypart_roi/session_<session>/fullfit/<roi>/`) and the
whole-frame Full SABLE ridge BPS of the per-neuron csv files of `neuron_selection_handoff/`. For
the groups All, 3D tuned (selected) and The Rest (unselected) neurons it draws one horizontal
stacked bar split into Whisker pad, Paws and Other.

The shares are computed from the mean held-out BPS of each group, averaged over the five folds:
Whisker pad is the mean BPS of the whisker pad patches (the `head` ROI folder), Paws that of the
paws ROI (`paws_concat` by default) and Other is the part of the whole-frame mean BPS that neither
explains, `whole - whisker pad - paws`. The whisker pad and paws patches can carry overlapping
information, so Other is clipped at 0 when they together exceed the whole frame. The three values
are then normalized to percentages.

It writes `ROI_preference_composition_s<session>_k<K>.png` and, next to it, the markdown table
with the mean BPS and the percentages behind the figure.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/ROI_preference_composition.py \
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
    BAR_FONT_SIZE,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_encoding_patch import (  # noqa: E402, E501
    save_figure,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_test_encoding_selected import (  # noqa: E402, E501
    SELECTION_KS,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_test_encoding_selected_vs_unselected import (  # noqa: E402, E501
    unselected_indices,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    RESULTS_DIR,
    RIDGE_BASE,
    SELECTION_DIR,
    SESSION_EIDS,
    load_selection_records,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

BODYPART_ROI_ROOT = RIDGE_BASE / 'bodypart_roi'
N_FOLDS = 5
HEAD_ROI = 'head'
DEFAULT_PAWS_ROI = 'paws_concat'
# left to right in the stacked bar, key of `group_shares` and legend label
PARTS = (('head', 'Whisker pad'), ('paws', 'Paws'), ('other', 'Other'))
PART_COLORS = {'head': '#4285F4', 'paws': '#E69F3C', 'other': '#92929C'}
# top to bottom in the figure
GROUP_LABELS = {'all': 'All', 'selected': '3D-tuned', 'unselected': 'The Rest'}
MIN_LABEL_PERCENT = 7.0  # narrower segments get no label, it would not fit


def load_roi_bps(roi_dir: Path, roi: str, k: int) -> np.ndarray:
    """Load the per-neuron held-out BPS of a body-part ROI averaged over the folds.

    Args:
        roi_dir: directory with one subfolder per ROI, e.g. `.../session_78/fullfit`.
        roi: ROI subfolder name, e.g. `'head'`.
        k: number of PCA components.

    Returns:
        Array of shape (neurons,), the mean of `fold_bps` over the folds found on disk.

    Raises:
        FileNotFoundError: if no `fullfit_full_f<fold>_k<K>.npz` file exists for this ROI.
    """
    paths = sorted((roi_dir / roi).glob(f'fullfit_full_f*_k{k}.npz'))
    if not paths:
        raise FileNotFoundError(f'no ridge files for roi {roi!r} and K={k} in {roi_dir / roi}')
    folds = []
    for path in paths:
        with np.load(path) as data:
            folds.append(data['fold_bps'].astype(np.float64))
    return np.mean(folds, axis=0)


def group_shares(
    bps_by_part: dict[str, np.ndarray],
    idx_neurons: list[int] | np.ndarray,
) -> dict[str, float]:
    """Mean BPS and percentage share of head, paws and other for one group of neurons.

    Args:
        bps_by_part: per-neuron BPS of shape (neurons,) under the keys `'head'`, `'paws'` and
            `'whole'`, NaN for neurons without a valid BPS.
        idx_neurons: indices of the neurons of the group.

    Returns:
        Mean BPS under `'head'`, `'paws'`, `'whole'` and `'other'`, where `'other'` is
        `whole - head - paws` clipped at 0, and the percentages under `'head_pct'`, `'paws_pct'`
        and `'other_pct'`, which add up to 100. Neurons with a NaN BPS in any of the three ROIs
        are dropped, `'n_neurons'` is the number kept.

    Raises:
        ValueError: if the group has no neuron with a valid BPS or no positive BPS to share.
    """
    stacked = np.stack([bps_by_part[part] for part in ('head', 'paws', 'whole')])[:, idx_neurons]
    stacked = stacked[:, np.isfinite(stacked).all(axis=0)]
    if stacked.shape[1] == 0:
        raise ValueError('the group has no neuron with a valid BPS')
    head, paws, whole = stacked.mean(axis=1)
    other = max(whole - head - paws, 0.0)
    total = head + paws + other
    if total <= 0:
        raise ValueError(f'the group has no positive BPS to share: head={head}, paws={paws}')
    return {
        'head': float(head),
        'paws': float(paws),
        'whole': float(whole),
        'other': float(other),
        'head_pct': float(100 * head / total),
        'paws_pct': float(100 * paws / total),
        'other_pct': float(100 * other / total),
        'n_neurons': int(stacked.shape[1]),
    }


def plot_stacked_share(
    shares_by_group: dict[str, dict[str, float]],
    save_path: Path,
    no_show: bool,
) -> None:
    """Write the horizontal stacked bar plot of the whisker pad, paws and other share.

    Args:
        shares_by_group: group key of `GROUP_LABELS` to the output of `group_shares`, in the order
            of the bars from top to bottom.
        save_path: output image path.
        no_show: close the figure instead of showing it.
    """
    fig, ax = plt.subplots(figsize=(4.1, 2.85), dpi=200)
    y_bars = np.arange(len(shares_by_group))[::-1]
    for y_bar, shares in zip(y_bars, shares_by_group.values()):
        left = 0.0
        for part, _label in PARTS:
            pct = shares[f'{part}_pct']
            ax.barh(y_bar, pct, left=left, height=0.38, color=PART_COLORS[part])
            if pct >= MIN_LABEL_PERCENT:
                ax.text(
                    left + pct / 2,
                    y_bar,
                    f'{pct:.0f}%',
                    ha='center',
                    va='center',
                    color='white',
                    fontsize=BAR_FONT_SIZE,
                    fontweight='bold',
                )
            left += pct
    ax.set_yticks(y_bars)
    ax.set_yticklabels(
        [GROUP_LABELS[group] for group in shares_by_group],
        fontsize=BAR_FONT_SIZE + 1,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(shares_by_group) - 0.4)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.set_xticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=PART_COLORS[part], edgecolor='none')
        for part, _label in PARTS
    ]
    ax.legend(
        handles,
        [label for _part, label in PARTS],
        loc='upper center',
        bbox_to_anchor=(0.45, -0.02),
        ncol=len(PARTS),
        frameon=False,
        fontsize=BAR_FONT_SIZE,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=2.5,
    )
    fig.tight_layout(pad=0.4)
    save_figure(fig, save_path, no_show)
    print(save_path)


def build_markdown_table(
    shares_by_group: dict[str, dict[str, float]],
    session: str,
    k: int,
    paws_roi: str,
) -> str:
    """Markdown table of the mean BPS and the percentage shares behind the figure.

    Args:
        shares_by_group: group key of `GROUP_LABELS` to the output of `group_shares`.
        session: session name.
        k: number of PCA components.
        paws_roi: name of the ROI used for Paws.

    Returns:
        The table as text, one row per group.
    """
    lines = [
        f'Mean held-out BPS and share of whisker pad, paws and other, session {session}, K={k}, '
        f'paws = {paws_roi}',
        '',
        '| group | neurons | whisker pad BPS | paws BPS | whole BPS | other BPS '
        '| whisker pad % | paws % | other % |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for group, shares in shares_by_group.items():
        lines.append(
            f'| {GROUP_LABELS[group]} | {shares["n_neurons"]} | {shares["head"]:.3f} '
            f'| {shares["paws"]:.3f} | {shares["whole"]:.3f} | {shares["other"]:.3f} '
            f'| {shares["head_pct"]:.1f} | {shares["paws_pct"]:.1f} | {shares["other_pct"]:.1f} |',
        )
    return '\n'.join(lines) + '\n'


def load_roi_groups(
    session: str,
    k: int,
    roi_dir: Path | None = None,
    paws_roi: str = DEFAULT_PAWS_ROI,
    selection_dir: Path = SELECTION_DIR,
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Load the per-neuron BPS of the ROIs and the whole frame, and the neuron groups.

    Args:
        session: session name, a key of `SESSION_EIDS`.
        k: number of PCA components.
        roi_dir: directory with one subfolder per ROI, default the `fullfit` folder of the session
            under `BODYPART_ROI_ROOT`.
        paws_roi: name of the ROI subfolder used for Paws.
        selection_dir: folder with the `session_<session>_k<K>_all_neurons.csv` files.

    Returns:
        The per-neuron BPS of shape (neurons,) under `'head'`, `'paws'` and `'whole'`, and the
        neuron indices of the groups `'all'`, `'selected'` and `'unselected'`.

    Raises:
        ValueError: if the ROI results and the selection csv have different numbers of neurons.
    """
    roi_dir = roi_dir or BODYPART_ROI_ROOT / f'session_{session}' / 'fullfit'
    records, idx_selected = load_selection_records(session, k, selection_dir)
    bps_by_part = {
        'head': load_roi_bps(roi_dir, HEAD_ROI, k),
        'paws': load_roi_bps(roi_dir, paws_roi, k),
        'whole': records[('ridge', session, 'full', k, 0)].bps,
    }
    n_neurons = len(bps_by_part['whole'])
    if any(len(bps) != n_neurons for bps in bps_by_part.values()):
        raise ValueError(
            f'neuron counts differ: {({part: len(bps) for part, bps in bps_by_part.items()})}',
        )
    idx_groups = {
        'all': list(range(n_neurons)),
        'selected': idx_selected,
        'unselected': unselected_indices(records, {session: idx_selected})[session],
    }
    return bps_by_part, idx_groups


def main() -> None:
    """Write the stacked bar plot and the markdown table of one session and K."""
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
    shares_by_group = {
        group: group_shares(bps_by_part, idx) for group, idx in idx_groups.items()
    }

    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'ROI_preference_composition',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f'ROI_preference_composition_s{args.session}_k{args.k}'
    table = build_markdown_table(shares_by_group, args.session, args.k, args.paws_roi)
    (save_dir / f'{stem}.txt').write_text(table)
    print(table)
    plot_stacked_share(shares_by_group, save_dir / f'{stem}.png', args.no_show)


if __name__ == '__main__':
    main()
