#!/usr/bin/env python3
"""LaTeX table of held-out BPS of the patch-embedding ridge encoders for every K.

Patch-embedding counterpart of ``neural_encoding_cls_token/latex_table_encoding_figure3.py``:
models are columns and the number of PCA components K are rows, cells are mean +/- SEM over
neurons (best model of each row in bold, via :func:`format_cells`).

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/latex_table_encoding_patch.py \
  --session all --folds 0
```
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.latex_table_encoding_figure3 import (  # noqa: E402
    format_cells,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    MODELS,
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    SESSION_EIDS,
    load_ridge_session_data,
    summarize,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402


def build_patch_latex_table(
    rows: list[dict],
    unit: str,
    *,
    decimals: int,
    caption: str,
    label: str,
) -> str:
    """Build the table, one row per K and one column per model.

    Args:
        rows: output of `summarize` with `error='neuron'`.
        unit: session name or `'all'`.
        decimals: decimal places of mean and SEM.
        caption: table caption.
        label: LaTeX label.

    Returns:
        The LaTeX source.
    """
    header = ' & '.join(['K', *(MODEL_LABELS[model] for model in MODELS)])
    lines = []
    for k in PCA_KS:
        by_model = {r['model']: r for r in rows if r['session'] == unit and r['k'] == k}
        means = np.array([by_model[m]['mean'] if m in by_model else np.nan for m in MODELS])
        errs = np.array([by_model[m]['err'] if m in by_model else np.nan for m in MODELS])
        cells = format_cells(means, errs, decimals=decimals)
        lines.append(' & '.join([str(k), *cells]) + ' \\\\')
    body = '\n'.join(lines)
    return (
        '\\begin{table}[ht!]\n'
        '\\centering\n'
        '\\small\n'
        f'\\caption{{{caption}}}\n'
        f'\\label{{{label}}}\n'
        f'\\begin{{tabular}}{{l*{{{len(MODELS)}}}{{c}}}}\n'
        '\\hline\n'
        f'{header} \\\\ \\hline\n'
        f'{body}\n'
        '\\hline\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )


def main() -> None:
    """Write the LaTeX table and print it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='all', choices=[*SESSION_EIDS, 'all'])
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--decimals', type=int, default=3)
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output', type=Path, default=None, help='output .tex path')
    args = parser.parse_args()

    sessions = list(SESSION_EIDS) if args.session == 'all' else [args.session]
    records, _ = load_ridge_session_data(sessions, args.neural_dir, folds=args.folds)
    rows = summarize(
        records,
        pooled=args.session == 'all',
        error='neuron',
        neuron_folds=args.folds,
    )
    scope = 'all sessions' if args.session == 'all' else f'session {args.session}'
    table = build_patch_latex_table(
        rows,
        args.session,
        decimals=args.decimals,
        caption=f'Held-out BPS of ridge encoders on PCA patch embeddings, {scope}.',
        label=f'tab:encoding-patch-{args.session}',
    )
    save_path = args.output or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_table',
        f'encoding_table_{args.session}.tex',
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(table)
    print(table)
    print(f'Saved LaTeX table to {save_path}')


if __name__ == '__main__':
    main()
