import torch

from beast.models.pca import PCAAutoencoder
from beast.sable_encoding_decoding.pca.decode_pca_latents import decode_latents_batch


class TestDecodeLatentsBatch:
    """Test the function decode_latents_batch."""

    def test_decode_latents_batch_output_shape(self, config_pca_small):
        model = PCAAutoencoder(config_pca_small)
        model.eval()

        z = torch.randn(5, 4)
        with torch.no_grad():
            render = decode_latents_batch(model, z)

        assert render.shape == (5, 3, 8, 8)

    def test_decode_latents_batch_matches_forward_pass(self, config_pca_small):
        model = PCAAutoencoder(config_pca_small)
        model.eval()

        image = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            xhat, z = model.forward(image)
            render = decode_latents_batch(model, z)

        torch.testing.assert_close(render, xhat)

    def test_decode_latents_batch_matches_linear_inverse(self, config_pca_small):
        model = PCAAutoencoder(config_pca_small)
        model.eval()

        z = torch.randn(3, 4)
        with torch.no_grad():
            render = decode_latents_batch(model, z)
            expected = (z @ model.pca_components + model.pca_mean).reshape(3, 3, 8, 8)

        torch.testing.assert_close(render, expected)
