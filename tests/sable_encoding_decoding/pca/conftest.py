import pytest


@pytest.fixture
def config_pca_small() -> dict:
    """Minimal PCA autoencoder config with a random 4-component subspace over 3x8x8 images."""
    return {
        'model': {
            'seed': 0,
            'model_class': 'pca',
            'model_params': {
                'image_size': 8,
                'num_channels': 3,
                'n_components': 4,
                'pca_pickle_path': None,
            },
        },
        'optimizer': {'type': 'Adam', 'lr': 1e-3, 'wd': 0.0, 'scheduler': 'cosine'},
        'training': {'train_batch_size': 2, 'num_gpus': 1, 'num_nodes': 1, 'num_epochs': 1},
    }
