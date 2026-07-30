"""Command to run model inference on videos or scene datasets for Sable model."""

import argparse
import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger('BEAST.CLI.PREDICT')


def register_parser(subparsers: Any) -> None:
    """Register the predict command parser."""

    parser = subparsers.add_parser(
        'predict',
        description=(
            'Run inference using a trained model. '
            'For Sable models, --input should point to the scene dataset file '
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
            'For Sable: scene dataset file (e.g. IBL camera-pairs .txt).'
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
        help='Extract and save latent features (non-Sable models)',
    )
    optional.add_argument(
        '--save_reconstructions', '-r',
        action='store_true',
        help='Extract and save reconstructions (non-Sable models)',
    )

    # Sable-specific options
    sable_group = parser.add_argument_group('Sable options')
    sable_group.add_argument(
        '--vda-cache-root',
        type=Path,
        help='Root directory of precomputed VDA depth cache',
    )
    sable_group.add_argument(
        '--correspondence-cache-root',
        type=Path,
        help='Root directory of precomputed correspondence cache',
    )
    sable_group.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val', 'test'],
        metavar='SPLIT',
        help='Dataset splits to run inference on (default: train val test)',
    )
    sable_group.add_argument(
        '--save-visuals',
        action='store_true',
        help='Save render-vs-target PNG grids alongside PLY point clouds',
    )
    sable_group.add_argument(
        '--save-camera-pointcloud-scene',
        action='store_true',
        help=(
            'Save .glb scenes (point cloud + camera frustums) alongside PLY point clouds; '
            'overlays ground-truth camera frustums (black) when '
            '--load-gt-camera-params-for-vis is also set'
        ),
    )
    sable_group.add_argument(
        '--load-gt-camera-params-for-vis',
        action='store_true',
        help=(
            'For Cheese3D datasets, load ground-truth camera calibration for the '
            '--save-camera-pointcloud-scene overlay, regardless of the saved training '
            "config's training.use_camera_params"
        ),
    )
    sable_group.add_argument(
        '--disable-init-gs',
        action='store_true',
        help=(
            'Override the trained config\'s model.gaussians.init_gs to skip VDA-depth-based '
            'Gaussian initialization/regularization at inference. Rendered output is '
            'unaffected (the render path never depends on init_gs); depth_output will be '
            'None and inference is faster'
        ),
    )
    sable_group.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Stop after this many batches (useful for smoke tests)',
    )
    sable_group.add_argument(
        '--session-names',
        nargs='+',
        metavar='SESSION',
        help='session IDs to load (default: use value from saved training config)',
    )
    sable_group.add_argument(
        '--max-files-per-session',
        type=int,
        default=None,
        help='Cap saved PLY/GLB files per session and group outputs into per-session subfolders',
    )

    # Sable latent extraction options
    latent_group = parser.add_argument_group('Sable latent extraction options')
    latent_group.add_argument(
        '--extract-latents',
        action='store_true',
        help='Extract per-pair latent tensors instead of PLY/GLB/PNG outputs',
    )
    latent_group.add_argument(
        '--return-frame-z',
        action='store_true',
        help='Save frame_z (3D-aware frame CLS tokens)',
    )
    latent_group.add_argument(
        '--return-dino-z',
        action='store_true',
        help='Save dino_z (DINO featurizer CLS tokens)',
    )
    latent_group.add_argument(
        '--return-cat-z',
        action='store_true',
        help='Save combined_z = cat([frame_z, dino_z], dim=-1)',
    )
    latent_group.add_argument(
        '--return-img-tokens',
        action='store_true',
        help='Save img_tokens (geometry-encoder image patch tokens); independent of the other '
             'latent types and may be combined with any of them',
    )
    latent_group.add_argument(
        '--return-all-z',
        action='store_true',
        help='Shorthand for --return-frame-z --return-dino-z --return-cat-z; '
             'mutually exclusive with the individual --return-frame-z/--return-dino-z/'
             '--return-cat-z flags',
    )
    latent_group.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume-skip; always recompute and overwrite existing latent files',
    )
    latent_group.add_argument(
        '--latent-batch-size',
        type=int,
        default=None,
        help=(
            'Override training.batch_size_per_gpu for latent extraction (default: use the '
            'saved training config\'s batch size)'
        ),
    )


def handle(args: argparse.Namespace) -> None:
    """Handle the predict command execution."""

    from beast.api.model import Model
    from beast.models.sable import Sable

    _logger.info(f'Loading model from: {args.model}')
    model = Model.from_dir(args.model)

    if isinstance(model.model, Sable):
        if args.disable_init_gs:
            _logger.info('disabling init_gs for inference (model.gaussians.init_gs override)')
            model.model.init_gs = False
        _handle_sable(args, model)
    else:
        _handle_video_or_images(args, model)


def _handle_sable(args, model):
    """Run Sable inference over a scene dataset."""

    if args.input is None:
        _logger.error('Sable models require --input')
        return

    output_dir = args.output or args.model / 'inference'

    if args.extract_latents:
        _handle_sable_extract_latents(args, model, output_dir)
        return

    _logger.info(f'Running Sable inference on: {args.input}')
    _logger.info(f'Output directory: {output_dir}')

    model.infer_sable(
        dataset_path=args.input,
        output_dir=output_dir,
        vda_cache_root=args.vda_cache_root,
        correspondence_cache_root=args.correspondence_cache_root,
        splits=args.splits,
        save_visuals=args.save_visuals,
        save_camera_pointcloud_scene=args.save_camera_pointcloud_scene,
        load_gt_camera_params_for_vis=args.load_gt_camera_params_for_vis,
        max_batches=args.max_batches,
        session_names=args.session_names,
        max_files_per_session=args.max_files_per_session,
    )


def _handle_sable_extract_latents(args, model, output_dir):
    """Extract per-pair Sable latent tensors instead of PLY/GLB/PNG outputs."""

    individual_flags = {
        'frame_z': args.return_frame_z,
        'dino_z': args.return_dino_z,
        'combined_z': args.return_cat_z,
    }
    if args.return_all_z and any(individual_flags.values()):
        _logger.error(
            '--return-all-z cannot be combined with --return-frame-z/--return-dino-z/'
            '--return-cat-z',
        )
        return

    if args.return_all_z:
        latent_types = list(individual_flags)
    else:
        latent_types = [name for name, requested in individual_flags.items() if requested]
    if args.return_img_tokens:
        latent_types.append('img_tokens')

    if not latent_types:
        _logger.error(
            '--extract-latents requires at least one of --return-all-z/--return-frame-z/'
            '--return-dino-z/--return-cat-z/--return-img-tokens',
        )
        return

    _logger.info(f'Extracting Sable latents {latent_types} from: {args.input}')
    _logger.info(f'Output directory: {output_dir}')

    model.extract_sable_latents(
        dataset_path=args.input,
        output_dir=output_dir,
        vda_cache_root=args.vda_cache_root,
        correspondence_cache_root=args.correspondence_cache_root,
        splits=args.splits,
        latent_types=latent_types,
        max_batches=args.max_batches,
        session_names=args.session_names,
        resume=not args.no_resume,
        batch_size=args.latent_batch_size,
    )


def _handle_video_or_images(args, model):
    """Run video/image inference for non-Sable models."""

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
