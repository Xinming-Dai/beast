#!/usr/bin/env python3
"""Per-neuron BPS scatter from the per-neuron csv files of `neuron_selection_handoff/`.

Handoff counterpart of `scatter_plot_encoding_patch.py`, drawn with the same
:func:`plot_bps_scatter`, but reading `session_<session>_k<K>_all_neurons.csv` instead of the
ridge predictions. Each point is one neuron with a valid BPS (`valid_bps == 1`). The two csv
columns to compare are passed with `--columns`, for example `full_mean_bps` against
`dino_mean_bps`, or any other numeric column such as `full_bps_f0`. One plot is written per
session, so several sessions can be passed at once.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/scatter_plot_encoding_selecte_vs_unselected.py \
  --sessions 78 72 4b --columns full_mean_bps dino_mean_bps --k 6 --no-show --neurons selected
```
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.scatter_plot_figure3 import (  # noqa: E402
    plot_bps_scatter,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    RESULTS_DIR,
    SELECTION_DIR,
    SESSION_EIDS,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

SELECTION_KS = (6, 32)  # numbers of PCA components of the handoff csv files
NEURON_GROUPS = ('all', 'selected', 'unselected')
DOT_SIZE = 8.0  # marker area, smaller than the default of `plot_bps_scatter` (18)


def column_label(column: str) -> str:
    """Axis label of a csv column, the model name if the column starts with a model key.

    Args:
        column: csv column name, e.g. `full_mean_bps`.

    Returns:
        The label of the model (e.g. `Full SABLE`), or the column name for other columns.
    """
    return MODEL_LABELS.get(column.split('_')[0], column)


def load_selection_bps(
    session: str,
    k: int,
    directory: Path,
    columns: tuple[str, str],
    group: str = 'all',
) -> tuple[np.ndarray, np.ndarray]:
    """Load two columns of the handoff csv of one session and K for the neurons with a valid BPS.

    Args:
        session: session name used in the csv file name.
        k: number of PCA components.
        directory: folder with the `session_<session>_k<K>_all_neurons.csv` files.
        columns: csv columns for the y and the x axis.
        group: `'all'` neurons, only the `'selected'` ones or only the `'unselected'` ones.

    Returns:
        The values of the first and of the second column, one entry per neuron.

    Raises:
        FileNotFoundError: if the csv of this session and K does not exist.
        ValueError: if a needed column is missing.
    """
    path = directory / f'session_{session}_k{k}_all_neurons.csv'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found')
    table = pd.read_csv(path)
    missing = [col for col in ('valid_bps', 'selected', *columns) if col not in table.columns]
    if missing:
        raise ValueError(f'{path} has no column(s) {missing}')
    keep = table['valid_bps'] == 1
    if group != 'all':
        keep &= table['selected'] == (1 if group == 'selected' else 0)
    return tuple(table.loc[keep, col].to_numpy(np.float64) for col in columns)


def main() -> None:
    """Write one scatter plot per session, the first column on the y axis, the second on x."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=['78'], choices=SESSION_EIDS)
    parser.add_argument(
        '--columns',
        nargs=2,
        metavar=('COLUMN_Y', 'COLUMN_X'),
        default=('full_mean_bps', 'dino_mean_bps'),
        help='csv columns, the first on the y axis, the second on the x axis',
    )
    parser.add_argument('--k', type=int, default=6, choices=SELECTION_KS)
    parser.add_argument(
        '--neurons',
        choices=NEURON_GROUPS,
        default='all',
        help='draw all neurons, only the selected ones or only the unselected ones',
    )
    parser.add_argument('--selection-dir', type=Path, default=SELECTION_DIR)
    parser.add_argument('--panel-label', default=None)
    parser.add_argument('--show-bps', action='store_true')
    parser.add_argument('--no-axis-labels', action='store_true')
    parser.add_argument('--dot-size', type=float, default=DOT_SIZE, help='marker area in points^2')
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    column_y, column_x = args.columns
    suffix = '' if args.neurons == 'all' else f'_{args.neurons}'
    for session in args.sessions:
        bps_y, bps_x = load_selection_bps(
            session,
            args.k,
            args.selection_dir,
            (column_y, column_x),
            args.neurons,
        )
        name = f'scatter_{session}_{column_y}_vs_{column_x}_k{args.k}{suffix}.png'
        if args.output_dir is None:
            save_path = _default_output_path_helper(RESULTS_DIR, 'encoding_scatter_handoff', name)
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            save_path = args.output_dir / name
        plot_bps_scatter(
            bps_x,
            bps_y,
            x_label=column_label(column_x),
            y_label=column_label(column_y),
            panel_label=args.panel_label,
            show_bps=args.show_bps,
            show_axis_labels=not args.no_axis_labels,
            dot_size=args.dot_size,
            save_path=save_path,
            also_save_pdf=args.save_pdf,
            show=not args.no_show,
        )
        print(save_path)


if __name__ == '__main__':
    main()
