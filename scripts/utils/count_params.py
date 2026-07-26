"""Report total and trainable parameter counts for a BEAST model config.

Usage:
    python scripts/utils/count_params.py configs/vit.yaml configs/vit_huge.yaml ...
"""

import argparse
import sys
from pathlib import Path

from beast.api.model import Model


def count_params(config_path: str) -> tuple[int, int]:
    """Instantiate a model from a config and count its parameters.

    Args:
        config_path: path to a BEAST model config yaml

    Returns:
        tuple of (total_params, trainable_params)
    """
    model = Model.from_config(config_path).model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main() -> None:
    """Print total and trainable parameter counts for each given config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configs', nargs='+', type=Path, help='config yaml paths')
    args = parser.parse_args()

    for config_path in args.configs:
        try:
            total_params, trainable_params = count_params(str(config_path))
        except Exception as e:  # noqa: BLE001 - report and continue with remaining configs
            print(f'{config_path}: FAILED ({e})', file=sys.stderr)
            continue
        print(
            f'{config_path}: total={total_params / 1e6:.1f}M '
            f'trainable={trainable_params / 1e6:.1f}M'
        )


if __name__ == '__main__':
    main()
