"""Shared driver for decoding neurally-estimated flat per-frame latents back into frames.

Models whose per-frame latent is a single flat vector (resnet's 768-dim bottleneck, the PCA
autoencoder's component scores) share the whole decode-and-score pipeline: load step3's
per-trial `img_tokens_estimated*.npz` (`z` shape `(K, T, 2, D)`), run each `(batch, D)` slab
through a model-specific decode function, un-normalize, optionally resize/mask, save PNGs and
compute PSNR/SSIM against the ground-truth eval frames. Only the decode function differs, so
`beast.sable_encoding_decoding.resnet.decode_resnet_latents` and
`beast.sable_encoding_decoding.pca.decode_pca_latents` are thin wrappers around this module.

`--image-size` resizes the decoded frames to another square size (e.g. 320, SABLE's native
resolution) before saving, masking and scoring, and loads targets/masks at that size when metrics
are on, so saved frames and metrics are comparable across baselines.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from beast.api.model import Model
from beast.inference import ImagePredictionHandler
from beast.logging import log_step
from beast.sable_encoding_decoding.img_token.decode_beast_tokens import load_estimated_tokens_dir
from beast.sable_encoding_decoding.img_token.target_frames import (
    load_frame_index_mapping,
    load_source_frame_index_mapping,
    load_target_images_for_trials,
    load_target_masks_for_trials,
)
from beast.sable_encoding_decoding.render.decode_utils import (
    _print_combined_metrics_summary,
    parse_neural_trial_index_arg,
    reconstruction_output_location,
)
from beast.sable_encoding_decoding.render.metrics import (
    collect_psnr_ssim_metrics_block,
    reassemble_flat_row_metrics,
    resize_image_batch,
    resolve_metrics_npz_path,
    save_psnr_ssim_metrics_npz,
)

DecodeLatentsBatch = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]


def parse_flat_latent_decode_args(
    argv: list[str] | None,
    description: str,
) -> argparse.Namespace:
    """Parse CLI arguments shared by every flat-latent decode entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.
        description: parser description shown by `--help` (the wrapper module's docstring).

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--model-dir', type=Path, required=True, help='trained model directory')
    ap.add_argument(
        '--estimated-dir',
        type=Path,
        required=True,
        help='step3 unproject.py output directory (img_tokens_estimated_neuraltrial*.npz)',
    )
    ap.add_argument('--out-dir', type=Path, required=True, help='directory for decoded frames')
    ap.add_argument('--batch-size', type=int, default=64, help='frames per decode batch')
    ap.add_argument(
        '--neural-trial-index',
        type=parse_neural_trial_index_arg,
        default=None,
        metavar='IDS',
        help='comma-separated neural_trial_idx values to keep',
    )
    ap.add_argument('--device', type=str, default='cuda:0', help='torch device for decoding')
    ap.add_argument(
        '--target-frame-mapping-left',
        type=Path,
        default=None,
        help='left-camera eval-layout input dir (frame_index_mapping.json), for PSNR/SSIM',
    )
    ap.add_argument(
        '--target-frame-mapping-right',
        type=Path,
        default=None,
        help='right-camera eval-layout input dir (frame_index_mapping.json), for PSNR/SSIM',
    )
    ap.add_argument(
        '--use-segmentation-mask',
        action='store_true',
        help=(
            'zero out background pixels (per precomputed SAM3 masks) in both render and target '
            'before saving/metrics; requires --segmentation-root, --eid, and '
            '--target-frame-mapping-left/-right'
        ),
    )
    ap.add_argument(
        '--segmentation-root',
        type=Path,
        default=None,
        help=(
            'root directory precomputed segmentation masks were written under (see '
            'beast.preprocess.sable.precompute_sam3_masks_eval)'
        ),
    )
    ap.add_argument('--eid', type=str, default=None, help='session id (mask subdirectory)')
    ap.add_argument(
        '--image-size',
        type=int,
        default=None,
        help=(
            'resize the decoded frames (bilinear) to this square size before masking, saving '
            'and PSNR/SSIM, and load ground-truth frames/masks at it, e.g. 320 to match SABLE; '
            "default: the model's image_size."
        ),
    )
    ap.add_argument(
        '--metrics-npz',
        type=Path,
        default=None,
        help='output .npz for PSNR/SSIM metrics (default: <out-dir>/psnr_ssim_metrics.npz)',
    )
    ap.add_argument(
        '--metrics-only',
        action='store_true',
        help=(
            'skip saving decoded-frame PNGs; only compute/save PSNR/SSIM metrics. Requires '
            '--target-frame-mapping-left/-right.'
        ),
    )
    args = ap.parse_args(argv)

    have_target_frames = (
        args.target_frame_mapping_left is not None or args.target_frame_mapping_right is not None
    )
    if have_target_frames and (
        args.target_frame_mapping_left is None or args.target_frame_mapping_right is None
    ):
        ap.error(
            '--target-frame-mapping-left and --target-frame-mapping-right must be given together',
        )
    if args.use_segmentation_mask and (
        args.segmentation_root is None or args.eid is None or not have_target_frames
    ):
        ap.error(
            '--use-segmentation-mask requires --segmentation-root, --eid, and '
            '--target-frame-mapping-left/-right',
        )
    if args.metrics_only and not have_target_frames:
        ap.error(
            '--metrics-only requires --target-frame-mapping-left/-right (nothing to score '
            'otherwise)',
        )
    return args


def run_flat_latent_decode(
    args: argparse.Namespace,
    decode_latents_batch: DecodeLatentsBatch,
    latent_name: str,
) -> None:
    """Decode estimated flat latents to frames and (optionally) score them against targets.

    Args:
        args: namespace from `parse_flat_latent_decode_args`.
        decode_latents_batch: maps `(model, z)` with `z` of shape `(batch, D)` to reconstructed
            frames of shape `(batch, channels, height, width)` in the model's normalized space.
        latent_name: short model label (e.g. `'resnet'`, `'pca'`) used in log messages only.
    """
    log_step(f'Loading estimated {latent_name} latents from: {args.estimated_dir}', level='info')
    z, trial_split_labels, neural_trial_idx, _paths = load_estimated_tokens_dir(
        args.estimated_dir, neural_trial_index=args.neural_trial_index,
    )
    k, t, v, d = z.shape

    log_step(f'Loading model from: {args.model_dir}', level='info')
    loaded = Model.from_dir(args.model_dir)
    model = loaded.model
    model.to(args.device)
    model.eval()

    handler = ImagePredictionHandler(args.out_dir, args.out_dir)

    flat_z = z.reshape(k * t * v, d)

    target = None
    target_masks = None
    trial_idx_flat = bin_idx_flat = split_flat = None
    metrics_source = args.estimated_dir
    if args.target_frame_mapping_left is not None:
        # targets/masks are loaded at the scoring size; renders are resized to it below
        image_size = args.image_size or int(
            loaded.config['model']['model_params']['image_size'],
        )
        unique_splits = sorted(set(trial_split_labels))
        mapping_left = {
            sp: load_frame_index_mapping(args.target_frame_mapping_left, sp)
            for sp in unique_splits
        }
        mapping_right = {
            sp: load_frame_index_mapping(args.target_frame_mapping_right, sp)
            for sp in unique_splits
        }
        log_step('Loading ground-truth target frames for PSNR/SSIM metrics', level='info')
        target_full = load_target_images_for_trials(
            trial_split_labels, neural_trial_idx, t, mapping_left, mapping_right, image_size,
        ).numpy()
        target = target_full.reshape(k * t * v, *target_full.shape[-3:])

        trial_idx_full = np.broadcast_to(np.asarray(neural_trial_idx)[:, None, None], (k, t, v))
        bin_idx_full = np.broadcast_to(np.arange(t)[None, :, None], (k, t, v))
        split_full = np.broadcast_to(
            np.asarray(trial_split_labels, dtype=object)[:, None, None], (k, t, v),
        )
        trial_idx_flat = trial_idx_full.reshape(k * t * v).astype(np.int64)
        bin_idx_flat = bin_idx_full.reshape(k * t * v).astype(np.int64)
        split_flat = split_full.reshape(k * t * v).astype(str)

        if args.use_segmentation_mask:
            mask_index_left = {
                sp: load_source_frame_index_mapping(args.target_frame_mapping_left, sp, 'left')
                for sp in unique_splits
            }
            mask_index_right = {
                sp: load_source_frame_index_mapping(args.target_frame_mapping_right, sp, 'right')
                for sp in unique_splits
            }
            log_step('Loading segmentation masks', level='info')
            masks_full = load_target_masks_for_trials(
                trial_split_labels,
                neural_trial_idx,
                t,
                mask_index_left,
                mask_index_right,
                args.segmentation_root,
                args.eid,
                image_size,
            ).numpy()
            target_masks = masks_full.reshape(k * t * v, *masks_full.shape[-3:])

    psnr_blocks, ssim_blocks, trial_blocks, bin_blocks, split_blocks = [], [], [], [], []
    num_decoded = 0
    with torch.no_grad():
        for start in range(0, flat_z.shape[0], args.batch_size):
            end = min(start + args.batch_size, flat_z.shape[0])
            z_batch = torch.from_numpy(flat_z[start:end]).to(args.device)

            # un-normalize before masking, so masked-out pixels are pixel-black (0), not the
            # ImageNet mean color, and render/target share the same [0, 1] scale downstream
            render = handler.unnormalize_batch(decode_latents_batch(model, z_batch))

            if args.image_size is not None:
                if start == 0:
                    log_step(
                        f'Resizing decoded frames from {tuple(render.shape[-2:])} to '
                        f'{args.image_size}x{args.image_size} before '
                        'masking/saving/PSNR-SSIM',
                        level='info',
                    )
                render = resize_image_batch(
                    render.unsqueeze(1), args.image_size,
                ).squeeze(1)

            if target_masks is not None:
                mask_batch = torch.from_numpy(target_masks[start:end]).to(
                    device=render.device, dtype=render.dtype,
                )
                render = render * mask_batch

            if not args.metrics_only:
                for i in range(render.shape[0]):
                    row = start + i
                    batch_dir, filename = reconstruction_output_location(row, t, v)
                    handler.save_reconstruction(
                        render[i], batch_dir, row, filename, normalized=False,
                    )
            num_decoded += render.shape[0]

            if target is not None:
                target_batch = torch.from_numpy(target[start:end]).to(args.device)
                if target_masks is not None:
                    target_batch = target_batch * mask_batch
                psnr, ssim, trial_idx, bin_idx, split_labels, _ = collect_psnr_ssim_metrics_block(
                    render.unsqueeze(1),
                    target_batch.unsqueeze(1),
                    metrics_source,
                    k_trials=end - start,
                    t_bins=1,
                    neural_trial_idx=(
                        trial_idx_flat[start:end] if trial_idx_flat is not None else None
                    ),
                    neural_bin_idx=(
                        bin_idx_flat[start:end].reshape(-1, 1)
                        if bin_idx_flat is not None
                        else None
                    ),
                    trial_split=split_flat[start:end] if split_flat is not None else None,
                )
                psnr_blocks.append(psnr)
                ssim_blocks.append(ssim)
                trial_blocks.append(trial_idx)
                bin_blocks.append(bin_idx)
                split_blocks.append(split_labels)

    log_step(f'Decoded {num_decoded} frames to: {args.out_dir}', level='info')

    if psnr_blocks:
        psnr, ssim, neural_trial_idx_out, neural_bin_idx_out, trial_split_out = (
            reassemble_flat_row_metrics(
                psnr_blocks, ssim_blocks, trial_blocks, bin_blocks, split_blocks,
                k_trials=k, t_bins=t, views=v,
            )
        )
        metrics_path = resolve_metrics_npz_path(args.metrics_npz, args.out_dir)
        saved_metrics = save_psnr_ssim_metrics_npz(
            metrics_path,
            psnr_blocks=[psnr],
            ssim_blocks=[ssim],
            neural_trial_blocks=[neural_trial_idx_out],
            neural_bin_blocks=[neural_bin_idx_out],
            trial_split_blocks=[trial_split_out],
            source_file_rows=[str(metrics_source)] * k,
        )
        log_step(f'Saved PSNR/SSIM metrics to: {metrics_path}', level='info')
        _print_combined_metrics_summary(metrics_path, saved_metrics)
