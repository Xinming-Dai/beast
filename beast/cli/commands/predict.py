"""Command to run model inference on videos or scene datasets for ERayZer model."""

import logging
from pathlib import Path

_logger = logging.getLogger('BEAST.CLI.PREDICT')


def register_parser(subparsers):
    """Register the predict command parser."""

    parser = subparsers.add_parser(
        'predict',
        description=(
            'Run inference using a trained model. '
            'For ERayZer models, --input should point to the scene dataset file '
            '(e.g. an IBL camera-pairs .txt).'
        ),
        usage='beast predict --model <model_dir> --input <path> [options]',
    )

    # Required arguments
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--model', '-m',
        type=Path,
        required=True,
        help='Directory containing trained model checkpoint and config.yaml',
    )
    required.add_argument(
        '--input', '-i',
        type=Path,
        help=(
            'Input path. For video/image models: video file or directory of images/videos. '
            'For ERayZer: scene dataset file (e.g. IBL camera-pairs .txt).'
        ),
    )

    # Optional arguments
    optional = parser.add_argument_group('options')
    optional.add_argument(
        '--output', '-o',
        type=Path,
        help='Directory to save prediction results (default: <model_dir>/inference)',
    )
    optional.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)',
    )
    optional.add_argument(
        '--save_latents', '-l',
        action='store_true',
        help='Extract and save latent features (non-ERayZer models)',
    )
    optional.add_argument(
        '--save_reconstructions', '-r',
        action='store_true',
        help='Extract and save reconstructions (non-ERayZer models)',
    )

    # ERayZer-specific options
    erayzer_group = parser.add_argument_group('ERayZer options')
    erayzer_group.add_argument(
        '--vda-cache-root',
        type=Path,
        help='Root directory of precomputed VDA depth cache',
    )
    erayzer_group.add_argument(
        '--correspondence-cache-root',
        type=Path,
        help='Root directory of precomputed correspondence cache',
    )
    erayzer_group.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        metavar='SPLIT',
        help='Dataset splits to run inference on (default: train val)',
    )
    erayzer_group.add_argument(
        '--save-visuals',
        action='store_true',
        help='Save render-vs-target PNG grids alongside PLY point clouds',
    )
    erayzer_group.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Stop after this many batches (useful for smoke tests)',
    )


def handle(args):
    """Handle the predict command execution."""

    from beast.api.model import Model
    from beast.models.erayzer import ERayZer

    _logger.info(f'Loading model from: {args.model}')
    model = Model.from_dir(args.model)

    if isinstance(model.model, ERayZer):
        _handle_erayzer(args, model)
    else:
        _handle_video_or_images(args, model)


def _handle_erayzer(args, model):
    """Run ERayZer inference over a scene dataset."""

    if args.input is None:
        _logger.error('ERayZer models require --input pointing to the scene dataset file')
        return

    output_dir = args.output or args.model / 'inference'
    _logger.info(f'Running ERayZer inference on: {args.input}')
    _logger.info(f'Output directory: {output_dir}')

    model.infer_erayzer(
        dataset_path=args.input,
        output_dir=output_dir,
        vda_cache_root=args.vda_cache_root,
        correspondence_cache_root=args.correspondence_cache_root,
        splits=args.splits,
        save_visuals=args.save_visuals,
        max_batches=args.max_batches,
    )


def _handle_video_or_images(args, model):
    """Run video/image inference for non-ERayZer models."""

    if args.input is None:
        _logger.error('--input is required')
        return

    _logger.info(f'Running inference with model from: {args.model}')
    _logger.info(f'Input: {args.input}')
    _logger.info(f'Output directory: {args.output or args.model}')
    if not args.save_latents and not args.save_reconstructions:
        _logger.warning(
            'did not detect --save_latents or --save_reconstructions; no outputs will be saved'
        )

    # Run prediction
    if args.input.is_file():
        # Single video inference
        model.predict_video(
            video_file=args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            save_latents=args.save_latents,
            save_reconstructions=args.save_reconstructions,
        )

    elif args.input.is_dir():

        videos = list(args.input.rglob('*.mp4')) + list(args.input.rglob('*.avi'))
        num_videos = len(videos)

        num_images = len(
            list(args.input.rglob('*.png'))
            + list(args.input.rglob('*.jpg'))
            + list(args.input.rglob('*.jpeg'))
        )

        if num_videos > 0 and num_images > 0:
            _logger.error(f'Found both videos and images in {args.input}; aborting')
            return
        elif num_videos > 0:
            for video_file in videos:
                _logger.info(f'Running inference on {video_file}')
                model.predict_video(
                    video_file=video_file,
                    output_dir=args.output,
                    batch_size=args.batch_size,
                    save_latents=args.save_latents,
                    save_reconstructions=args.save_reconstructions,
                )
        else:
            model.predict_images(
                image_dir=args.input,
                output_dir=args.output,
                batch_size=args.batch_size,
                save_latents=args.save_latents,
                save_reconstructions=args.save_reconstructions,
            )
