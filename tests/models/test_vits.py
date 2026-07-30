import copy

import torch

from beast.models.vits import VisionTransformer


class TestVisionTransformer:

    def test_forward(self, config_vit):
        config = copy.deepcopy(config_vit)
        input = torch.randn((5, 3, 224, 224))
        model = VisionTransformer(config)
        results = model.forward(input)
        assert 'latents' in results
        assert 'loss' in results
        assert 'reconstructions' in results
        assert results['reconstructions'].shape[0] == input.shape[0]

    def test_get_model_outputs(self, config_vit):
        config = copy.deepcopy(config_vit)
        input = torch.randn((5, 3, 224, 224))
        batch_dict = {'image': input}
        model = VisionTransformer(config)
        results = model.get_model_outputs(batch_dict)
        assert 'latents' in results
        assert 'loss' in results
        assert 'reconstructions' in results
        assert 'images' in results
        assert results['images'].shape == input.shape

    def test_predict_step_return_reconstructions(self, config_vit):
        config = copy.deepcopy(config_vit)
        model = VisionTransformer(config)
        model.eval()

        batch_dict = {
            'image': torch.randn(2, 3, 224, 224),
            'video': ['vid_a', 'vid_b'],
            'idx': torch.tensor([0, 1]),
            'image_path': ['/fake/0.png', '/fake/1.png'],
        }

        model.return_reconstructions = True
        result = model.predict_step(batch_dict, 0)
        assert 'reconstructions' in result

        model.return_reconstructions = False
        result = model.predict_step(batch_dict, 0)
        assert 'reconstructions' not in result

    def test_predict_step_return_img_tokens(self, config_vit):
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['random_init'] = True
        config['model']['model_params']['return_img_tokens'] = True
        model = VisionTransformer(config)
        model.eval()
        model.return_reconstructions = False

        num_patches = (224 // 16) ** 2
        batch_dict = {
            'image': torch.randn(2, 3, 224, 224),
            'video': ['vid_a', 'vid_b'],
            'idx': torch.tensor([0, 1]),
            'image_path': ['/fake/0.png', '/fake/1.png'],
        }

        hidden_size = config['model']['model_params']['hidden_size']
        result = model.predict_step(batch_dict, 0)
        assert 'img_tokens' in result
        assert 'ids_restore' in result
        # img_tokens includes the CLS token (num_patches + 1), matching what the MAE decoder
        # expects for reconstruction
        assert result['img_tokens'].shape == (2, num_patches + 1, hidden_size)
        assert result['ids_restore'].shape == (2, num_patches)
        # latents still collapses to the CLS token even when img_tokens is also requested
        assert result['latents'].shape == (2, hidden_size)

    def test_predict_step_without_return_img_tokens_omits_keys(self, config_vit):
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['random_init'] = True
        model = VisionTransformer(config)
        model.eval()
        model.return_reconstructions = False

        batch_dict = {
            'image': torch.randn(2, 3, 224, 224),
            'video': ['vid_a', 'vid_b'],
            'idx': torch.tensor([0, 1]),
            'image_path': ['/fake/0.png', '/fake/1.png'],
        }

        result = model.predict_step(batch_dict, 0)
        assert 'img_tokens' not in result
        assert 'ids_restore' not in result


class TestVisionTransformerIntegration:
    """Integration tests that train and run inference on a ViT autoencoder."""

    def test_integration_basic(self, config_vit, run_model_test) -> None:
        """Test ViT autoencoder with basic masked autoencoder loss."""
        config = copy.deepcopy(config_vit)
        run_model_test(config=config)

    def test_integration_contrastive(self, config_vit, run_model_test) -> None:
        """Test ViT autoencoder with contrastive learning (infoNCE) enabled."""
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_infoNCE'] = True
        run_model_test(config=config)

    def test_integration_perceptual_loss(self, config_vit, run_model_test) -> None:
        """Test ViT autoencoder with AlexNet perceptual loss enabled."""
        config = copy.deepcopy(config_vit)
        config['model']['model_params']['use_perceptual_loss'] = True
        run_model_test(config=config)
