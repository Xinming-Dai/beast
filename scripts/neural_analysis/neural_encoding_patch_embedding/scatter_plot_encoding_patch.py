#!/usr/bin/env python3
"""Per-neuron BPS scatter of two patch-embedding ridge encoders for one session.

Patch-embedding counterpart of ``neural_encoding_cls_token/scatter_plot_figure3.py``, drawn with
the same :func:`plot_bps_scatter`. Each point is one neuron with test spikes.

```bash
python scripts/neural_analysis/neural_encoding_patch_embedding/scatter_plot_encoding_patch.py \
  --session 78 --models full dino --k 6 --fold 0 --no-show
```
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.neural_analysis.neural_encoding_cls_token.scatter_plot_figure3 import (  # noqa: E402
    plot_bps_scatter,
)
from scripts.neural_analysis.neural_encoding_patch_embedding.patch_helpers import (  # noqa: E402
    MODEL_LABELS,
    MODELS,
    NEURAL_DIR,
    PCA_KS,
    RESULTS_DIR,
    SESSION_EIDS,
    load_ridge_session_data,
    neuron_bps,
)
from scripts.neural_analysis.plot_helpers import _default_output_path_helper  # noqa: E402


def main() -> None:
    """Write one scatter plot, model Y against model X."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default='78', choices=SESSION_EIDS)
    parser.add_argument(
        '--models',
        nargs=2,
        metavar=('MODEL_Y', 'MODEL_X'),
        default=('full', 'dino'),
        choices=MODELS,
        help='first model on the y-axis, second on the x-axis',
    )
    parser.add_argument('--k', type=int, default=6, choices=PCA_KS)
    parser.add_argument('--folds', type=int, nargs='+', default=[0])
    parser.add_argument('--neural-dir', type=Path, default=NEURAL_DIR)
    parser.add_argument('--panel-label', default=None)
    parser.add_argument('--show-bps', action='store_true')
    parser.add_argument('--no-axis-labels', action='store_true')
    parser.add_argument('--save-pdf', action='store_true')
    parser.add_argument('-o', '--output', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()

    model_y, model_x = args.models
    records, _ = load_ridge_session_data(
        [args.session],
        args.neural_dir,
        ks=(args.k,),
        folds=args.folds,
    )
    bps_y = neuron_bps(records, 'ridge', args.session, model_y, args.k, args.folds)
    bps_x = neuron_bps(records, 'ridge', args.session, model_x, args.k, args.folds)
    save_path = args.output or _default_output_path_helper(
        RESULTS_DIR,
        'encoding_scatter',
        f'scatter_{args.session}_{model_y}_vs_{model_x}_k{args.k}.png',
    )
    plot_bps_scatter(
        bps_x,
        bps_y,
        x_label=MODEL_LABELS[model_x],
        y_label=MODEL_LABELS[model_y],
        panel_label=args.panel_label,
        show_bps=args.show_bps,
        show_axis_labels=not args.no_axis_labels,
        save_path=save_path,
        also_save_pdf=args.save_pdf,
        show=not args.no_show,
    )
    print(save_path)


if __name__ == '__main__':
    main()
