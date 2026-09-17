#!/usr/bin/env python3
"""LaTeX table of encoding BPS across methods, aggregated over EIDs.

Mirrors :mod:`bar_plot_encoding_figure3` (same folder structure, method stats), but
emits a LaTeX table instead of a bar chart:

```bash
python scripts/neural_analysis/latex_table_encoding_figure3.py \
    --methods keypoints sable_dino
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.neural_analysis.bar_plot_encoding_figure3 import (
    EID_SET,
    _allowed_eid_set,
    collect_method_stats,
)
from scripts.neural_analysis.plot_helpers import (
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHODS,
    _default_output_path_helper,
)

def _label_to_key(label: str) -> str:
    """Slugify a row label into a LaTeX-label-safe key, e.g. 'Encoding (bps)' -> 'encoding-bps'."""
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "table"


def format_cells(
    means: np.ndarray,
    ses: np.ndarray,
    *,
    decimals: int,
) -> list[str]:
    """Per-method 'mean $\\pm$ se' cells, with the max-mean cell bolded."""
    if means.shape != ses.shape:
        raise ValueError(f"means and ses must have the same shape; got {means.shape} vs {ses.shape}")
    finite = np.isfinite(means)
    best_idx = int(np.nanargmax(means)) if np.any(finite) else None

    cells = []
    for i, (mean, se) in enumerate(zip(means, ses)):
        if not np.isfinite(mean):
            cells.append("--")
            continue
        text = f"{mean:.{decimals}f} $\\pm$ {se:.{decimals}f}"
        if i == best_idx:
            text = f"\\textbf{{{text}}}"
        cells.append(text)
    return cells


def build_latex_table(
    method_labels: list[str],
    cells: list[str],
    *,
    row_label: str,
    caption: str,
    label: str,
) -> str:
    n = len(method_labels)
    header = " & ".join(["", *method_labels])
    data_row = " & ".join([row_label, *cells])
    column_spec = f"l*{{{n}}}{{c}}"
    return (
        "\\begin{table}[ht!]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\hline\n"
        f"{header} \\\\ \\hline\n"
        f"{data_row} \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Root with one subfolder per method")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Method folder names")
    p.add_argument(
        "--method-labels",
        nargs="+",
        default=None,
        help="Display label per method (default: DEFAULT_METHOD_LABELS, aligned to --methods)",
    )
    p.add_argument("--eids", nargs="*", default=None, help="EIDs to include (default: EID_SET)")
    p.add_argument(
        "--bps-source",
        choices=("stored", "per-neuron"),
        default="per-neuron",
        help="Aggregate stored per-EID BPS, or recompute per-neuron BPS from gt/pred",
    )
    p.add_argument("--row-label", default="Encoding (bps)", help="Row label in the header column")
    p.add_argument("--caption", default=None, help="Table caption (default: derived from --row-label)")
    p.add_argument("--label", default=None, help="LaTeX \\label{} key (default: derived from --row-label)")
    p.add_argument("--decimals", type=int, default=3, help="Decimal places for mean/SE")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output .tex path")

    args = p.parse_args()

    method_labels: list[str]
    if args.method_labels is not None:
        method_labels = list(args.method_labels)
    else:
        default_label_by_method = dict(zip(DEFAULT_METHODS, DEFAULT_METHOD_LABELS))
        method_labels = [default_label_by_method.get(str(m), str(m)) for m in args.methods]
    if len(method_labels) != len(args.methods):
        raise ValueError(
            f"--method-labels must match --methods in length: {len(method_labels)} vs {len(args.methods)}"
        )

    results_dir = args.results_dir.expanduser().resolve()
    allowed = _allowed_eid_set(args.eids if args.eids is not None else sorted(EID_SET))

    _, means_cnn, _, se_cnn = collect_method_stats(
        results_dir, args.methods, allowed, bps_source=args.bps_source,
    )
    cells = format_cells(means_cnn, se_cnn, decimals=args.decimals)

    row_key = _label_to_key(args.row_label)
    caption = args.caption or f"{args.row_label} across methods."
    label = args.label or f"tab:{row_key}"

    table = build_latex_table(
        method_labels,
        cells,
        row_label=args.row_label,
        caption=caption,
        label=label,
    )

    save_path = (
        args.output
        if args.output is not None
        else _default_output_path_helper(results_dir, "encoding_table", "encoding_table.tex")
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(table)

    print(table)
    print(f"Saved LaTeX table to {save_path}")


if __name__ == "__main__":
    main()
