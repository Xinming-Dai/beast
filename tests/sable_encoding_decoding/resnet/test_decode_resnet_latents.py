import copy

import numpy as np
import torch

from beast.models.resnets import ResnetAutoencoder
from beast.sable_encoding_decoding.img_token.decode_beast_tokens import load_estimated_tokens_dir
from beast.sable_encoding_decoding.resnet.decode_resnet_latents import decode_latents_batch


class TestDecodeLatentsBatch:
    """Test the function decode_latents_batch."""

    def test_decode_latents_batch_matches_direct_reconstruction(self, config_ae):
        config = copy.deepcopy(config_ae)
        model = ResnetAutoencoder(config)
        model.eval()

        num_latents = config['model']['model_params']['num_latents']
        z = torch.randn(4, num_latents)
        with torch.no_grad():
            render = decode_latents_batch(model, z)
            expected = model.decoder(model.latents_to_decoder(z))

        assert render.shape == (4, 3, 224, 224)
        torch.testing.assert_close(render, expected)

    def test_decode_latents_batch_matches_forward_pass(self, config_ae):
        config = copy.deepcopy(config_ae)
        model = ResnetAutoencoder(config)
        model.eval()

        image = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            xhat, z = model.forward(image)
            render = decode_latents_batch(model, z)

        torch.testing.assert_close(render, xhat)


class TestLoadEstimatedResnetLatents:
    """Test loading resnet's per-trial estimated-latent npz files with the shared img_token
    loader (no ids_restore/patch grid, so load_estimated_tokens_dir is reused unchanged).
    """

    def test_load_estimated_tokens_dir_reads_resnet_shaped_latents(self, tmp_path):
        t_bins, num_cameras, num_latents = 5, 2, 12
        z = np.random.default_rng(0).normal(size=(1, t_bins, num_cameras, num_latents))
        np.savez(
            tmp_path / 'img_tokens_estimated_neuraltrial0007.npz',
            z=z.astype(np.float32),
            neural_trial_idx=np.int64(7),
            trial_split=np.array(['test'], dtype=object),
        )

        loaded_z, split_labels, neural_trial_idx, paths = load_estimated_tokens_dir(tmp_path)

        assert loaded_z.shape == (1, t_bins, num_cameras, num_latents)
        assert split_labels == ['test']
        assert neural_trial_idx.tolist() == [7]
        assert len(paths) == 1
