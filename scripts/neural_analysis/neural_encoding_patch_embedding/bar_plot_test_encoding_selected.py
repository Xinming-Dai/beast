#!/usr/bin/env python3
"""Held-out BPS bar plot of the selected neurons of the patch-embedding ridge encoders.

Same figures as `bar_plot_encoding_patch.py`, restricted to the neurons flagged `selected` in the
per-neuron csv files of `neuron_selection_handoff/` (`session_<session>_k<K>_all_neurons.csv`):
one bar per model (DINO, NoGeom SABLE, Full SABLE) with each session's mean overlaid as a marker.
The csv BPS is already averaged over the five folds, and the frame-permutation random baseline is
not part of it, so it is not drawn.

The selection favors neurons where Full SABLE beats DINO and NoGeom SABLE (`wins_vs_dino`,
`wins_vs_nog`) and was made on the same cross-validation folds the BPS comes from. The bars
describe the selected neurons and are not an unbiased estimate of the Full SABLE advantage.

Per K it writes `bar_selected_k<K>.png` and `.txt` (absolute BPS), `bar_selected_k<K>_delta.png`
and `.txt` (per-neuron BPS minus DINO) and `bar_selected_k<K>_sessions.png` (a panel per session).

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_test_encoding_selected.py \
  --k 6 32 --no-show
```
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_patch_embedding.bar_plot_encoding_patch import (  # noqa: E402, E501
    BASELINE_MODEL,
    plot_group,
    plot_sessions_grid,
    summarize_group,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
    load_selection_records,
    restrict_records,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

SELECTION_KS = (6, 32)  # numbers of PCA components of the handoff csv files
FOLDS_IN_CSV = [0, 1, 2, 3, 4]  # the csv BPS is the mean over these folds
GROUP = 'selected neurons of neuron_selection_handoff'


def plot_selected_k(args: argparse.Namespace, k: int, save_dir: Path) -> None:
    """Write the plots and tables of the selected neurons at one K.

    Args:
        args: parsed arguments of the script.
        k: number of PCA components.
        save_dir: output folder.
    """
    records = {}
    idx_selected: dict[str, list[int]] = {}
    for session in args.sessions:
        records_session, idx_selected[session] = load_selection_records(
            session,
            k,
            args.selection_dir,
        )
        records.update(records_session)
    records = restrict_records(records, idx_selected)
    # the reused plotting functions read k, sessions, folds, pool and no_show from the arguments
    args_k = argparse.Namespace(**{**vars(args), 'k': k, 'folds': FOLDS_IN_CSV})
    stem = f'bar_selected_k{k}'
    rows = summarize_group(records, args_k)
    plot_group(rows, args_k, save_dir, stem, GROUP)
    plot_group(
        summarize_group(records, args_k, BASELINE_MODEL),
        args_k,
        save_dir,
        f'{stem}_delta',
        GROUP,
        BASELINE_MODEL,
    )
    plot_sessions_grid(
        {'Selected neurons': rows},
        args_k,
        save_dir / f'{stem}_sessions.png',
    )


def main() -> None:
    """Write the bar plots and markdown tables of the selected neurons for every K."""
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
        'encoding_barplot_selected',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    for k in args.k:
        plot_selected_k(args, k, save_dir)


if __name__ == '__main__':
    main()
