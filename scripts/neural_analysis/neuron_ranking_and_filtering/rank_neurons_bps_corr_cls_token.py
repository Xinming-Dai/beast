#!/usr/bin/env python3
"""Rank all neurons of all sessions for the CLS-token encoders and save the rankings.

Every ``encoding_results*.npy`` under the method folders of the results directory is scored per
neuron, for the ``cnn`` and ``rrr`` encoders it holds: BPS, the correlation of the predicted and
ground-truth PSTH, and the modulation of the predicted PSTH over time. One CSV per (EID, method,
encoder) is written to ``<results-dir>/neuron_ranking/<eid>/``, so the plotting scripts of
``neural_encoding_cls_token`` can look it up with ``--ranking``. By default everything is ranked, the
flags restrict the runs.

```bash
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_corr_cls_token.py
python scripts/neural_analysis/neuron_ranking_and_filtering/rank_neurons_bps_corr_cls_token.py \
  --eid 4b00df29-3769-43be-bb40-128b1cba6d35 --methods ResNet CLS --encoder cnn
```
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.top_bps_psth_compare_models_figure3 import (
    RESULTS_DIR,
    _gt_pred_for_encoder,
)
from scripts.neural_analysis.neuron_ranking_and_filtering.ranking_helpers import (
    DEFAULT_MIN_MODULATION,
    DEFAULT_MIN_PSTH_CORR,
    RANKING_DIRNAME,
    cls_ranking_path,
    pass_filters,
    rank_neurons,
    save_ranking,
)
from scripts.neural_analysis.plot_helpers import (
    EID_SET,
    iter_eid_encoding_npys,
    normalize_eid,
)

ENCODERS = ('cnn', 'rrr')


def find_method_dirs(results_dir: Path, methods: list[str] | None) -> list[Path]:
    """The method folders of a results directory, without the rankings folder.

    Args:
        results_dir: root directory containing one subfolder per method.
        methods: folder names to keep, None for all.

    Returns:
        The folders, sorted by name.
    """
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir()
        and path.name != RANKING_DIRNAME
        and (methods is None or path.name in methods)
    )


def main() -> None:
    """Write one ranking CSV per (EID, method, encoder) found in the results directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=RESULTS_DIR,
        help='root directory containing one subfolder per method',
    )
    parser.add_argument('--eid', nargs='+', default=sorted(EID_SET), choices=sorted(EID_SET))
    parser.add_argument('--methods', nargs='+', default=None, help='default: all method folders')
    parser.add_argument('--encoder', nargs='+', default=list(ENCODERS), choices=ENCODERS)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=f'folder of the CSVs (default: <results-dir>/{RANKING_DIRNAME})',
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser()
    output_dir = args.output_dir or results_dir / RANKING_DIRNAME
    n_written = 0
    for method_dir in find_method_dirs(results_dir, args.methods):
        for eid in args.eid:
            matches = sorted(iter_eid_encoding_npys(method_dir, frozenset({normalize_eid(eid)})))
            if not matches:
                continue
            for encoder in args.encoder:
                try:
                    gt, pred = _gt_pred_for_encoder(matches[0], eid, encoder)
                except KeyError:
                    continue
                df = rank_neurons(gt, pred)
                path = cls_ranking_path(output_dir, eid, method_dir.name, encoder)
                save_ranking(
                    df,
                    path,
                    {'eid': eid, 'method': method_dir.name, 'encoder': encoder},
                )
                n_pass = pass_filters(
                    df,
                    min_psth_corr=DEFAULT_MIN_PSTH_CORR,
                    min_modulation=DEFAULT_MIN_MODULATION,
                ).sum()
                print(f'{path}: {len(df)} neurons, {n_pass} pass the default PSTH filters')
                n_written += 1
    print(f'wrote {n_written} rankings to {output_dir}')


if __name__ == '__main__':
    main()
