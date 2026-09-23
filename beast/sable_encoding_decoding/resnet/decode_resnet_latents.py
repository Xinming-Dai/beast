"""Decode neurally-estimated resnet frame latents back into reconstructed frames.

Unlike beast's ViT-MAE `decode_beast_tokens.py`, this needs no `ids_restore`/patch-grid
bookkeeping: `ResnetAutoencoder`'s decode is just `latents_to_decoder(z) -> decoder(features)`,
run directly on the flat 768-dim latent already saved by `step1_resnet_latent.sh` and carried
through the PCA-compress / neural-decode / unproject steps unchanged (see
`beast.sable_encoding_decoding.img_token.unproject`). The camera axis (left/right) is already a
plain leading dim of size 2 in resnet's saved latents (`L=1` token per camera), so — unlike
beast's shard-layout `2*L` merged axis — no un-merge step is needed either.

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
    return parse_flat_latent_decode_args(argv, description=__doc__)


def main(argv: list[str] | None = None) -> None:
    """Run the resnet frame-latent decode pipeline end to end (CLI entry point)."""
    run_flat_latent_decode(parse_args(argv), decode_latents_batch, latent_name='resnet')


if __name__ == '__main__':
    main()
