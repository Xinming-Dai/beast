#!/usr/bin/env python3
"""LaTeX table of encoding BPS and decoding PSNR/SSIM for the model-size ablation.

One results root holds an encoding and a decoding tree with the same ``<method>/<eid>/``
layout:

```text
model_size_ablation/
├── BEAST_RESNET_model_size_ablation_encoding/<method>/<eid>/encoding_results*.npy
└── BEAST_RESNET_model_size_ablation_decoding/<method>/<eid>/psnr_ssim_metrics*.npz
```

The method folder names differ from ``DEFAULT_METHODS`` in ``plot_helpers.py`` (and the
SABLE folder is named differently in the two trees), so the method lists and display labels
are defined at the top of this module. The table style mirrors
:mod:`latex_table_encoding_figure3`. By default methods are rows and the three metrics
(decoding PSNR, decoding SSIM, encoding BPS) are columns, which keeps the table narrow;
``--layout metrics-as-rows`` transposes it so methods run across the header:

```bash
python scripts/neural_analysis/latex_table_model_size_ablation.py \
    --results-dir /path/to/model_size_ablation 
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.neural_analysis.bar_plot_encoding_figure3 import (
    _allowed_eid_set,
    collect_method_stats as collect_encoding_stats,
)
from scripts.neural_analysis.decoding_metrics_figure2 import (
    collect_method_stats as collect_decoding_stats,
)
from scripts.neural_analysis.latex_table_encoding_figure3 import format_cells
from scripts.neural_analysis.plot_helpers import EID_SET, _default_output_path_helper

RESULTS_DIR = Path('/projects/bfsr/xdai3/project3d/iclr_plotting/ibl/model_size_ablation')
ENCODING_SUBDIR = 'BEAST_RESNET_model_size_ablation_encoding'
DECODING_SUBDIR = 'BEAST_RESNET_model_size_ablation_decoding'
DEFAULT_METHODS_ENCODING: list[str] = [
    'resnet_18',
    'resnet_152',
    'beast_vit_base',
    'beast_vit_large',
    'sable_dino',
]
DEFAULT_METHODS_DECODING: list[str] = [
    'resnet_18',
    'resnet_152',
    'beast_vit_base',
    'beast_vit_large',
    'sable',
]
DEFAULT_METHOD_LABELS: list[str] = [
    'ResNet-AE-18',
    'ResNet-AE-152',
    'BEAST (ViT Base)',
    'BEAST (ViT Large)',
    'SABLE',
]


def build_latex_table(
    header: list[str],
    rows: list[tuple[str, list[str]]],
    *,
    caption: str,
    label: str,
) -> str:
    """Assemble a multi-row LaTeX table in the :mod:`latex_table_encoding_figure3` style.

    Args:
        header: Header cells, starting with the label of the row-label column.
        rows: ``(row_label, cells)`` pairs; each ``cells`` list has one entry per data column.
        caption: Table caption.
        label: LaTeX ``\\label{}`` key.

    Returns:
        The full ``table`` environment as a string.

    Raises:
        ValueError: If any row's cell count does not match the number of data columns.
    """
    n = len(header) - 1
    for row_label, cells in rows:
        if len(cells) != n:
            raise ValueError(f'Row {row_label!r} has {len(cells)} cells; expected {n}')
    header_row = ' & '.join(header)
    data_rows = ''.join(f'{" & ".join([row_label, *cells])} \\\\\n' for row_label, cells in rows)
    column_spec = f'l*{{{n}}}{{c}}'
    return (
        '\\begin{table}[ht!]\n'
        '\\centering\n'
        '\\small\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{{column_spec}}}\n'
        '\\hline\n'
        f'{header_row} \\\\ \\hline\n'
        f'{data_rows}'
        '\\hline\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )


def main() -> None:
    """Parse CLI arguments, aggregate encoding/decoding metrics, and write the .tex table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--results-dir',
        type=Path,
        default=RESULTS_DIR,
        help='Root containing the encoding and decoding subfolders',
    )
    p.add_argument(
        '--encoding-subdir',
        default=ENCODING_SUBDIR,
        help='Encoding tree under --results-dir, with one subfolder per method',
    )
    p.add_argument(
        '--decoding-subdir',
        default=DECODING_SUBDIR,
        help='Decoding tree under --results-dir, with one subfolder per method',
    )
    p.add_argument(
        '--methods-encoding',
        nargs='+',
        default=DEFAULT_METHODS_ENCODING,
        help='Method folder names under the encoding tree',
    )
    p.add_argument(
        '--methods-decoding',
        nargs='+',
        default=DEFAULT_METHODS_DECODING,
        help='Method folder names under the decoding tree, aligned to --methods-encoding',
    )
    p.add_argument(
        '--method-labels',
        nargs='+',
        default=DEFAULT_METHOD_LABELS,
        help='Display label per method, aligned to --methods-encoding',
    )
    p.add_argument('--eids', nargs='*', default=None, help='EIDs to include (default: EID_SET)')
    p.add_argument(
        '--bps-source',
        choices=('stored', 'per-neuron'),
        default='per-neuron',
        help='Aggregate stored per-EID BPS, or recompute per-neuron BPS from gt/pred',
    )
    p.add_argument(
        '--layout',
        choices=('methods-as-rows', 'metrics-as-rows'),
        default='methods-as-rows',
        help='Methods down the first column (default) or across the header',
    )
    p.add_argument(
        '--encoding-label',
        default='Encoding (bps) $\\uparrow$',
        help='Encoding metric label',
    )
    p.add_argument(
        '--psnr-label', default='Decoding (PSNR) $\\uparrow$', help='PSNR metric label',
    )
    p.add_argument(
        '--ssim-label', default='Decoding (SSIM) $\\uparrow$', help='SSIM metric label',
    )
    p.add_argument('--caption', default=None, help='Table caption')
    p.add_argument('--label', default='tab:model-size-ablation', help='LaTeX \\label{} key')
    p.add_argument('--decimals', type=int, default=3, help='Decimal places for mean/SE')
    p.add_argument('--no-se', action='store_true', help='Omit the $\\pm$ se suffix in cells')
    p.add_argument('-o', '--output', type=Path, default=None, help='Output .tex path')

    args = p.parse_args()

    n_methods = len(args.methods_encoding)
    if len(args.methods_decoding) != n_methods or len(args.method_labels) != n_methods:
        raise ValueError(
            '--methods-encoding, --methods-decoding and --method-labels must have equal '
            f'lengths: {n_methods} vs {len(args.methods_decoding)} vs {len(args.method_labels)}'
        )

    results_dir = args.results_dir.expanduser().resolve()
    encoding_dir = results_dir / args.encoding_subdir
    decoding_dir = results_dir / args.decoding_subdir
    for tree in (encoding_dir, decoding_dir):
        if not tree.is_dir():
            raise FileNotFoundError(f'Results tree not found: {tree}')
    allowed = _allowed_eid_set(args.eids if args.eids is not None else sorted(EID_SET))
    show_se = not args.no_se

    _, means_cnn, _, se_cnn = collect_encoding_stats(
        encoding_dir, args.methods_encoding, allowed, bps_source=args.bps_source,
    )
    means_psnr, means_ssim, se_psnr, se_ssim = collect_decoding_stats(
        decoding_dir, args.methods_decoding, allowed,
    )

    encoding_cells = format_cells(means_cnn, se_cnn, decimals=args.decimals, show_se=show_se)
    psnr_cells = format_cells(means_psnr, se_psnr, decimals=args.decimals, show_se=show_se)
    ssim_cells = format_cells(means_ssim, se_ssim, decimals=args.decimals, show_se=show_se)

    method_labels = list(args.method_labels)
    metric_labels = [args.psnr_label, args.ssim_label, args.encoding_label]
    metric_cells = [psnr_cells, ssim_cells, encoding_cells]
    if args.layout == 'methods-as-rows':
        header = ['Method', *metric_labels]
        rows = [
            (method_label, [cells[idx] for cells in metric_cells])
            for idx, method_label in enumerate(method_labels)
        ]
    else:
        header = ['Metric', *method_labels]
        rows = list(zip(metric_labels, metric_cells))

    caption = args.caption or (
        'Neural encoding (bits per spike) and decoding (PSNR, SSIM) across model sizes.'
    )
    table = build_latex_table(header, rows, caption=caption, label=args.label)

    save_path = (
        args.output
        if args.output is not None
        else _default_output_path_helper(
            results_dir, 'model_size_ablation_table', 'model_size_ablation_table.tex',
        )
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(table)

    print(table)
    print(f'Saved LaTeX table to {save_path}')


if __name__ == '__main__':
    main()
