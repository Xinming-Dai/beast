"""Decode neurally-estimated resnet frame latents back into reconstructed frames.

Unlike beast's ViT-MAE `decode_beast_tokens.py`, this needs no `ids_restore`/patch-grid
bookkeeping: `ResnetAutoencoder`'s decode is just `latents_to_decoder(z) -> decoder(features)`,
run directly on the flat 768-dim latent already saved by `step1_resnet_latent.sh` and carried
through the PCA-compress / neural-decode / unproject steps unchanged (see
`beast.sable_encoding_decoding.img_token.unproject`). The camera axis (left/right) is already a
plain leading dim of size 2 in resnet's saved latents (`L=1` token per camera), so — unlike
beast's shard-layout `2*L` merged axis — no un-merge step is needed either.
"""

import argparse
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
from beast.sable_encoding_decoding.render.decode_utils import _print_combined_metrics_summary
from beast.sable_encoding_decoding.render.metrics import (
    collect_psnr_ssim_metrics_block,
    resolve_metrics_npz_path,
    save_psnr_ssim_metrics_npz,
)


def decode_latents_batch(
    model: torch.nn.Module,
    z: torch.Tensor,
) -> torch.Tensor:
    """Run a batch of flat resnet latents back through the model's own decoder.

    Args:
        model: a loaded `beast.models.resnets.ResnetAutoencoder` (or `.model` of a `Model`).
        z: flat latents, shape `(batch, num_latents)`.

    Returns:
        Reconstructed frames, shape `(batch, channels, height, width)`.
    """
    features = model.latents_to_decoder(z)
    return model.decoder(features)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the resnet frame-latent decode entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True, help='trained resnet model directory')
    ap.add_argument(
        '--estimated-dir',
        type=Path,
        required=True,
        help='step3 unproject.py output directory (img_tokens_estimated_neuraltrial*.npz)',
    )
    ap.add_argument('--out-dir', type=Path, required=True, help='directory for decoded frames')
    ap.add_argument('--batch-size', type=int, default=64, help='frames per decode batch')
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
    return args


def main(argv: list[str] | None = None) -> None:
    """Run the resnet frame-latent decode pipeline end to end (CLI entry point)."""
    args = parse_args(argv)

    log_step(f'Loading estimated resnet latents from: {args.estimated_dir}', level='info')
    z, trial_split_labels, neural_trial_idx, _paths = load_estimated_tokens_dir(
        args.estimated_dir,
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
        image_size = int(loaded.config['model']['model_params']['image_size'])
        unique_splits = sorted(set(trial_split_labels))
        mapping_left = {
            sp: load_frame_index_mapping(args.target_frame_mapping_left, sp) for sp in unique_splits
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

            render = decode_latents_batch(model, z_batch)

            if target_masks is not None:
                mask_batch = torch.from_numpy(target_masks[start:end]).to(
                    device=render.device, dtype=render.dtype,
                )
                render = render * mask_batch

            for i in range(render.shape[0]):
                row = start + i
                handler.save_reconstruction(render[i], 'decoded', row, Path(f'row{row:06d}.png'))
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
        metrics_path = resolve_metrics_npz_path(None, args.out_dir)
        saved_metrics = save_psnr_ssim_metrics_npz(
            metrics_path,
            psnr_blocks=psnr_blocks,
            ssim_blocks=ssim_blocks,
            neural_trial_blocks=trial_blocks,
            neural_bin_blocks=bin_blocks,
            trial_split_blocks=split_blocks,
            source_file_rows=[str(metrics_source)] * sum(b.shape[0] for b in trial_blocks),
        )
        log_step(f'Saved PSNR/SSIM metrics to: {metrics_path}', level='info')
        _print_combined_metrics_summary(metrics_path, saved_metrics)


if __name__ == '__main__':
    main()
