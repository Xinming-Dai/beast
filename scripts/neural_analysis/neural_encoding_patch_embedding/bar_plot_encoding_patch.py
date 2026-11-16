#!/usr/bin/env python3
"""Held-out BPS bar plot of the patch-embedding ridge encoders at one K.

Patch-embedding counterpart of ``neural_encoding_cls_token/bar_plot_encoding_figure3.py``: one bar
per model (DINO, NoGeom SABLE, Full SABLE), mean +/- SEM over neurons pooled over all
sessions, drawn with the same :func:`plot_encoding_bars`. Also writes a markdown table with the
mean and SEM of every session and of the pooled neurons next to the figure.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/bar_plot_encoding_patch.py \
  --k 6 --fold 0 --no-show
```
"""

import argparse
import sys
from pathlib import Path

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
    summarize,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402

BAR_MODELS = ('dino', 'nogeom', 'full')  # bar order, left to right


def build_markdown_table(rows: list[dict], units: list[str], k: int, folds: list[int]) -> str:
    """Markdown table of mean +/- SEM over neurons, one row per session plus the pooled row.

    Args:
        rows: output of `summarize` with `error='neuron'`, per session and pooled.
        units: session names followed by `'all'`.
        k: number of PCA components.
        folds: folds the neurons were averaged over.

    Returns:
        The markdown text.
    """
    header = ['Session', 'Neurons', *(MODEL_LABELS[model] for model in BAR_MODELS)]
    lines = [
        f'Held-out BPS (bits/spike), ridge, K={k}, fold(s) {folds}, mean +/- SEM over neurons',
        '',
        '| ' + ' | '.join(header) + ' |',
        '| ' + ' | '.join(['---'] * len(header)) + ' |',
    ]
    for unit in units:
        by_model = {r['model']: r for r in rows if r['session'] == unit and r['k'] == k}
        n_neurons = by_model[BAR_MODELS[0]]['n_neurons']
        cells = [f'{by_model[m]["mean"]:.4f} +/- {by_model[m]["err"]:.4f}' for m in BAR_MODELS]
        name = 'All sessions' if unit == 'all' else unit
        lines.append('| ' + ' | '.join([name, str(n_neurons), *cells]) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Write the pooled bar plot and the markdown table of per-session results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions', nargs='+', default=list(SESSION_EIDS), choices=SESSION_EIDS)
    parser.add_argument('--k', type=int, default=6, choices=PCA_KS)
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('-o', '--output-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    records, _ = load_ridge_session_data(args.sessions, args.neural_dir)
    save_dir = args.output_dir or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_barplot',
        'bar.png',
    ).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f'bar_k{args.k}_fold{"".join(map(str, args.folds))}'

    rows = []
    for pooled in (False, True):
        rows += summarize(records, pooled=pooled, error='neuron', neuron_folds=args.folds)
    pooled_by_model = {r['model']: r for r in rows if r['session'] == 'all' and r['k'] == args.k}
    means = np.array([pooled_by_model[model]['mean'] for model in BAR_MODELS])
    errs = np.array([pooled_by_model[model]['err'] for model in BAR_MODELS])
    plot_encoding_bars(
        [MODEL_LABELS[model] for model in BAR_MODELS],
        np.full_like(means, np.nan),
        means,
        np.full_like(errs, np.nan),
        errs,
        [(model_color(model), model_color(model)) for model in BAR_MODELS],
        plot_encoders='cnn',
        figsize=(3.2, 2.2),
        xtick_color='black',
        save_path=save_dir / f'{stem}.png',
        show=not args.no_show,
    )
    table = build_markdown_table(rows, [*args.sessions, 'all'], args.k, args.folds)
    (save_dir / f'{stem}.txt').write_text(table)
    print(save_dir / f'{stem}.png')
    print(table)


if __name__ == '__main__':
    main()
