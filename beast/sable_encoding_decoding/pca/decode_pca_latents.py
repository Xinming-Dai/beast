"""Decode neurally-estimated PCA-autoencoder frame latents back into reconstructed frames.

`beast.models.pca.PCAAutoencoder` has no decoder network: its latent is the vector of
component scores `z = (x - mu) @ W.T`, and the reconstruction is the linear inverse
`xhat = z @ W + mu` reshaped to `(channels, image_size, image_size)`. The per-frame latents
written by `step1_latent.sh` (`frame_z_trials.npz`, `z` shape `(K, T, 2, n_components)`) have
exactly resnet's flat layout, so they go through the same PCA-compress / neural-decode /
unproject steps, and this module only supplies the PCA-specific decode for step4.

Everything after the decode call (un-normalize, resize, mask, save, PSNR/SSIM) lives in
`beast.sable_encoding_decoding.render.decode_flat_latents`; see it for the CLI flags.
"""

import argparse

import torch

from beast.sable_encoding_decoding.render.decode_flat_latents import (
    parse_flat_latent_decode_args,
    run_flat_latent_decode,
)


def decode_latents_batch(
    model: torch.nn.Module,
    z: torch.Tensor,
) -> torch.Tensor:
    """Map a batch of PCA component scores back to (normalized) image space.

    Mirrors the reconstruction half of `PCAAutoencoder.forward`: `z @ W + mu`, reshaped to
    the image geometry recorded in the model config (same defaults as the model's `__init__`).

    Args:
        model: a loaded `beast.models.pca.PCAAutoencoder` (or `.model` of a `Model`).
        z: flat latents, shape `(batch, n_components)`.

    Returns:
        Reconstructed frames, shape `(batch, num_channels, image_size, image_size)`.
    """
    params = model.config['model']['model_params']
    image_size = int(params.get('image_size', 224))
    num_channels = int(params.get('num_channels', 3))
    recon = z @ model.pca_components + model.pca_mean
    return recon.reshape(z.shape[0], num_channels, image_size, image_size)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the PCA frame-latent decode entry point.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    return parse_flat_latent_decode_args(argv, description=__doc__)


def main(argv: list[str] | None = None) -> None:
    """Run the PCA frame-latent decode pipeline end to end (CLI entry point)."""
    run_flat_latent_decode(parse_args(argv), decode_latents_batch, latent_name='pca')


if __name__ == '__main__':
    main()
