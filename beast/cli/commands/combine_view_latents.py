"""Command to pair per-view latents from non-Sable models (ViT, ResNet) into (n, 2, dim) arrays."""

import argparse
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger('BEAST.CLI.COMBINE_VIEW_LATENTS')


def register_parser(subparsers: Any) -> None:
    """Register the combine-view-latents command parser."""

    parser = subparsers.add_parser(
        'combine-view-latents',
        description=(
            'Pair per-image latents saved by `beast predict --save_latents` (one .npy per '
            'image, under <latents_dir>/<view>/<frame_stem>.npy) into a single (n_frames, 2, '
            'dim) .npz array, matching frames by stem across two view subdirectories.'
        ),
        usage='beast combine-view-latents --latents-dir <dir> --output <path> [options]',
    )

    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--latents-dir',
        type=Path,
        required=True,
        help="Directory containing one subdirectory per view (beast predict's <output>/latents)",
    )
    required.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Path to the output .npz file',
    )

    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--views',
        nargs=2,
        default=['left', 'right'],
        metavar=('VIEW_A', 'VIEW_B'),
        help='Names of the two view subdirectories to pair, in output order (default: left right)',
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the combine-view-latents command execution."""

    from beast.inference import combine_view_latents

    _logger.info(f'Pairing views {args.views} from: {args.latents_dir}')
    combine_view_latents(
        latents_dir=args.latents_dir,
        output_path=args.output,
        views=tuple(args.views),
    )
