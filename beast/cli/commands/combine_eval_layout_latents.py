"""Command to pair eval-layout per-frame latents (ViT, ResNet) into neural-aligned trials."""

import argparse
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger('BEAST.CLI.COMBINE_EVAL_LAYOUT_LATENTS')


def register_parser(subparsers: Any) -> None:
    """Register the combine-eval-layout-latents command parser."""

    parser = subparsers.add_parser(
        'combine-eval-layout-latents',
        description=(
            'Pair eval-layout per-frame latents saved by `beast predict --save_latents` (one '
            '.npy per image, run once per camera) into the neural-aligned trials .npz schema '
            '(train_z_trials_time/val_z_trials_time/test_z_trials_time, neural_trial_idx, '
            'per-split intervals), using each camera directory\'s frame_index_mapping.json.'
        ),
        usage=(
            'beast combine-eval-layout-latents --left-input-dir <dir> --right-input-dir <dir> '
            '--left-latents-dir <dir> --right-latents-dir <dir> --output <path> [options]'
        ),
    )

    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--left-input-dir',
        type=Path,
        required=True,
        help='Eval-layout left-camera directory passed as `beast predict --input`',
    )
    required.add_argument(
        '--right-input-dir',
        type=Path,
        required=True,
        help='Eval-layout right-camera directory passed as `beast predict --input`',
    )
    required.add_argument(
        '--left-latents-dir',
        type=Path,
        required=True,
        help="Left camera's <output>/latents directory from `beast predict`",
    )
    required.add_argument(
        '--right-latents-dir',
        type=Path,
        required=True,
        help="Right camera's <output>/latents directory from `beast predict`",
    )
    required.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Path to the output .npz file',
    )

    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        metavar='SPLIT',
        help='Split names to assemble, in row order (default: train val test)',
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the combine-eval-layout-latents command execution."""

    from beast.inference import combine_eval_layout_latents

    _logger.info(
        f'Pairing splits {args.splits} from: {args.left_input_dir} and {args.right_input_dir}',
    )
    combine_eval_layout_latents(
        left_input_dir=args.left_input_dir,
        right_input_dir=args.right_input_dir,
        left_latents_dir=args.left_latents_dir,
        right_latents_dir=args.right_latents_dir,
        output_path=args.output,
        splits=tuple(args.splits),
    )
