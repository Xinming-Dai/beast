"""Command to pair eval-layout per-frame img_tokens (+ ids_restore) into neural-aligned trials."""

import argparse
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger('BEAST.CLI.COMBINE_EVAL_LAYOUT_IMG_TOKENS')


def register_parser(subparsers: Any) -> None:
    """Register the combine-eval-layout-img-tokens command parser."""

    parser = subparsers.add_parser(
        'combine-eval-layout-img-tokens',
        description=(
            'Pair eval-layout per-frame img_tokens + ids_restore saved by '
            '`beast predict --save_latents --return-img-tokens` (one .npy pair per image, run '
            'once per camera) into the neural-aligned trials .npz schema '
            '(train_z_trials_time/val_z_trials_time/test_z_trials_time, neural_trial_idx, '
            "per-split intervals), using each camera directory's frame_index_mapping.json. "
            'ids_restore is written to a separate sidecar .npz next to the main trials file.'
        ),
        usage=(
            'beast combine-eval-layout-img-tokens --left-input-dir <dir> --right-input-dir <dir> '
            '--left-img-tokens-dir <dir> --right-img-tokens-dir <dir> '
            '--left-ids-restore-dir <dir> --right-ids-restore-dir <dir> --output <path> [options]'
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
        '--left-img-tokens-dir',
        type=Path,
        required=True,
        help="Left camera's <output>/img_tokens directory from `beast predict`",
    )
    required.add_argument(
        '--right-img-tokens-dir',
        type=Path,
        required=True,
        help="Right camera's <output>/img_tokens directory from `beast predict`",
    )
    required.add_argument(
        '--left-ids-restore-dir',
        type=Path,
        required=True,
        help="Left camera's <output>/ids_restore directory from `beast predict`",
    )
    required.add_argument(
        '--right-ids-restore-dir',
        type=Path,
        required=True,
        help="Right camera's <output>/ids_restore directory from `beast predict`",
    )
    required.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help=(
            'Path to the output img_tokens trials .npz file. Required unless --batch-size is '
            'given (sharded mode uses --output-dir instead).'
        ),
    )

    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--ids-restore-output',
        type=Path,
        default=None,
        help=(
            'Path to the output ids_restore sidecar .npz file (default: '
            "'ids_restore_<output name>' next to --output). Ignored in sharded mode."
        ),
    )
    optional.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        metavar='SPLIT',
        help='Split names to assemble, in row order (default: train val test)',
    )
    optional.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=(
            'When given, switch to sharded mode: write img_tokens_batch*.npz shards of this '
            'many (trial, timebin) rows each, instead of one combined trials .npz. Avoids ever '
            'materializing the full combined array (large for per-patch img_tokens). Requires '
            '--output-dir and --session-id; --output / --ids-restore-output are ignored.'
        ),
    )
    optional.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Sharded mode only: root directory for the img_tokens/<session_id>/<split>/ shards.',
    )
    optional.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Sharded mode only: session/EID name, used for the shard path and row metadata.',
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the combine-eval-layout-img-tokens command execution."""

    if args.batch_size is not None:
        from beast.inference import extract_eval_layout_img_token_batches

        if args.output_dir is None or args.session_id is None:
            raise SystemExit('--batch-size requires --output-dir and --session-id.')
        _logger.info(
            f'Sharding splits {args.splits} (batch_size={args.batch_size}) from: '
            f'{args.left_input_dir} and {args.right_input_dir} -> {args.output_dir}',
        )
        extract_eval_layout_img_token_batches(
            left_input_dir=args.left_input_dir,
            right_input_dir=args.right_input_dir,
            left_img_tokens_dir=args.left_img_tokens_dir,
            right_img_tokens_dir=args.right_img_tokens_dir,
            left_ids_restore_dir=args.left_ids_restore_dir,
            right_ids_restore_dir=args.right_ids_restore_dir,
            output_dir=args.output_dir,
            session_id=args.session_id,
            batch_size=args.batch_size,
            splits=tuple(args.splits),
        )
        return

    from beast.inference import combine_eval_layout_img_tokens

    if args.output is None:
        raise SystemExit('--output is required unless --batch-size is given.')

    _logger.info(
        f'Pairing splits {args.splits} from: {args.left_input_dir} and {args.right_input_dir}',
    )
    combine_eval_layout_img_tokens(
        left_input_dir=args.left_input_dir,
        right_input_dir=args.right_input_dir,
        left_img_tokens_dir=args.left_img_tokens_dir,
        right_img_tokens_dir=args.right_img_tokens_dir,
        left_ids_restore_dir=args.left_ids_restore_dir,
        right_ids_restore_dir=args.right_ids_restore_dir,
        output_path=args.output,
        ids_restore_output_path=args.ids_restore_output,
        splits=tuple(args.splits),
    )
