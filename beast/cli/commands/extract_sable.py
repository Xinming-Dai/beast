"""Command to run the SABLE IBL stereo extraction pipeline."""

import argparse
import logging
from typing import Any

from beast.cli.types import config_file
from beast.preprocess.config_sable import load_sable_config, validate_sable_config
from beast.preprocess.extraction_sable import run_pipeline

_logger = logging.getLogger('BEAST.CLI.EXTRACT_SABLE')


def register_parser(subparsers: Any) -> None:
    """Register the extract_sable command parser."""
    parser = subparsers.add_parser(
        'extract_sable',
        description=(
            'Run the SABLE IBL stereo extraction pipeline '
            '(stats, trim, downsample, extract, VDA depth, LitPose CSV→npy).'
        ),
        usage='beast extract_sable --config <config_path> [options]',
    )

    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--config', '-c',
        type=config_file,
        required=True,
        help='path to SABLE extraction config file (YAML)',
    )

    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--skip-stats',
        action='store_true',
        default=False,
        help='skip the video stats step',
    )
    optional.add_argument(
        '--sessionids',
        nargs='+',
        default=None,
        metavar='SESSION_ID',
        help=(
            'optional list of session IDs; overrides the sessionids field in the config. '
            'If omitted, uses the config sessionids (null = all sessions).'
        ),
    )
    optional.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='re-run steps even if output files already exist',
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the extract_sable command execution."""

    cfg = load_sable_config(args.config)
    validate_sable_config(cfg)

    # CLI --sessionids overrides the config value
    if args.sessionids is not None:
        cfg = cfg.model_copy(update={'sessionids': args.sessionids})

    run_pipeline(cfg, skip_stats=args.skip_stats, overwrite=args.overwrite)
    _logger.info('extract_sable pipeline complete')
