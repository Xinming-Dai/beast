"""Linear PCA autoencoder implementation.

Trainable linear autoencoder in PCA form: ``latents = (x - mu) @ W.T`` and
``reconstruction = latents @ W + mu``, with ``mu`` and ``W`` stored as
``nn.Parameter`` so the base Lightning training loop can fine-tune them via
gradient descent (the subspace is not constrained to stay orthonormal after the
first optimizer step). ``W``/``mu`` can optionally be initialized from a
pre-fitted PCA model (see ``beast fit-pca`` / :func:`save_pca_model`), or start
from a small random subspace when only ``n_components`` is given.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from typeguard import typechecked

from beast.models.base import BaseLightningModel


class GPU_PCA:
    """PCA parameters held in NumPy, matching :func:`save_pca_model`/:func:`load_pca_model`."""

    def __init__(
        self,
        components: np.ndarray,
        mean: np.ndarray,
        explained_variance: np.ndarray,
    ) -> None:
        """Store fitted PCA parameters.

        Args:
            components: principal axes, shape (n_components, n_features)
            mean: per-feature mean, shape (n_features,)
            explained_variance: variance explained by each component, shape (n_components,)
        """
        self.components_ = np.asarray(components, dtype=np.float32)
        self.mean_ = np.asarray(mean, dtype=np.float32).ravel()
        self.explained_variance_ = np.asarray(explained_variance, dtype=np.float32).ravel()
        self.n_components_ = int(self.components_.shape[0])


@typechecked
def save_pca_model(pca_model: GPU_PCA, output_path: str | Path) -> None:
    """Save a fitted :class:`GPU_PCA` to a pickle file.

    Args:
        pca_model: fitted PCA parameters
        output_path: path to write the pickle file
    """
    model_dict: dict[str, Any] = {
        'components_': np.asarray(pca_model.components_, dtype=np.float32),
        'mean_': np.asarray(pca_model.mean_, dtype=np.float32),
        'explained_variance_': np.asarray(pca_model.explained_variance_, dtype=np.float32),
        'n_components_': pca_model.n_components_,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(model_dict, f)


@typechecked
def load_pca_model(input_path: str | Path) -> GPU_PCA:
    """Load a fitted :class:`GPU_PCA` from a pickle file written by :func:`save_pca_model`.

    Args:
        input_path: path to the pickle file

    Returns:
        the loaded PCA parameters
    """
    with open(input_path, 'rb') as f:
        model_dict = pickle.load(f)
    return GPU_PCA(
        components=model_dict['components_'],
        mean=model_dict['mean_'],
        explained_variance=model_dict['explained_variance_'],
    )


@typechecked
class PCAAutoencoder(BaseLightningModel):
    """Trainable linear autoencoder: ``latents = (x - mu) @ W.T``, ``xhat = latents @ W + mu``.

    ``mu`` and ``W`` are ``nn.Parameter``s, so reconstruction loss backprops and the
    optimizer configured via ``config['optimizer']`` updates them with the same
    schedule as other :class:`BaseLightningModel` subclasses.

    Model parameters (``config['model']['model_params']``):
        image_size: input image side length (default 224)
        num_channels: number of image channels (default 3)
        n_components: latent size; inferred from ``pca_pickle_path`` if omitted
        pca_pickle_path: optional pickle from :func:`save_pca_model` used to
            initialize ``mu``/``W``; when omitted, ``n_components`` is required
            and ``W`` starts as a small random subspace
    """

    def __init__(self, config: dict) -> None:
        """Initialize the mean/component parameters from a pickle or randomly."""
        super().__init__(config)
        params = config['model']['model_params']
        image_size = int(params.get('image_size', 224))
        num_channels = int(params.get('num_channels', 3))
        self._flat_dim = num_channels * image_size * image_size

        pca_path = params.get('pca_pickle_path')
        if pca_path:
            pca = load_pca_model(Path(pca_path))
            mean_t = torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32))
            comp_t = torch.from_numpy(np.asarray(pca.components_, dtype=np.float32))
            if comp_t.shape[1] != self._flat_dim:
                raise ValueError(
                    f'PCA feature dim {comp_t.shape[1]} != '
                    f'num_channels * image_size ** 2 = {self._flat_dim}'
                )
        else:
            n_comp = params.get('n_components')
            if n_comp is None:
                raise ValueError(
                    'Provide model_params.pca_pickle_path or model_params.n_components.'
                )
            mean_t = torch.zeros(self._flat_dim, dtype=torch.float32)
            n_comp = int(n_comp)
            comp_t = torch.randn(n_comp, self._flat_dim, dtype=torch.float32)
            # keep the random init small so early reconstructions aren't degenerate
            comp_t.mul_(1.0 / np.sqrt(self._flat_dim))

        self.pca_mean = nn.Parameter(mean_t)
        self.pca_components = nn.Parameter(comp_t)

    def forward(
        self,
        x: Float[torch.Tensor, 'batch channels height width'],
    ) -> tuple[
        Float[torch.Tensor, 'batch channels height width'],
        Float[torch.Tensor, 'batch n_components'],
    ]:
        """Project input images onto principal components and reconstruct.

        Returns:
            tuple of (reconstructed_images, latents)
        """
        b = x.shape[0]
        flat = x.reshape(b, -1)
        centered = flat - self.pca_mean
        z = centered @ self.pca_components.T
        recon = z @ self.pca_components + self.pca_mean
        xhat = recon.reshape_as(x)
        return xhat, z

    def get_model_outputs(
        self,
        batch_dict: dict,
        return_images: bool = True,
        return_reconstructions: bool = True,
    ) -> dict:
        """Run forward pass and return results dict with optional images and reconstructions.

        Args:
            batch_dict: dict containing 'image' tensor
            return_images: whether to include input images in results
            return_reconstructions: whether to include reconstructions in results

        Returns:
            dict with 'latents', and optionally 'images' and 'reconstructions'
        """
        x = batch_dict['image']
        xhat, z = self.forward(x)
        results_dict = {
            'latents': z,
        }
        if return_images:
            results_dict['images'] = x
        if return_reconstructions:
            results_dict['reconstructions'] = xhat
        return results_dict

    def compute_loss(
        self,
        stage: str | None,
        images: Float[torch.Tensor, 'batch channels height width'],
        reconstructions: Float[torch.Tensor, 'batch channels height width'],
        latents: Float[torch.Tensor, 'batch n_components'],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Compute MSE reconstruction loss between input images and reconstructions.

        Args:
            stage: training stage ('train', 'val', 'test', or None)
            images: original input images
            reconstructions: model reconstructions
            latents: latent representations (unused; required by base class signature)
            **kwargs: additional keyword arguments (ignored)

        Returns:
            tuple of (loss tensor, list of logging dicts)
        """
        mse_loss = F.mse_loss(images, reconstructions, reduction='mean')
        log_list = [
            {'name': f'{stage}_mse', 'value': mse_loss}
        ]
        return mse_loss, log_list

    def predict_step(self, batch_dict: dict, batch_idx: int) -> dict:
        """Run inference on a single batch and return latents with metadata.

        Args:
            batch_dict: dict containing 'image', 'video', 'idx', 'image_path'
            batch_idx: index of the current batch

        Returns:
            dict with 'latents', optional 'reconstructions', and 'metadata'
        """
        results_dict = self.get_model_outputs(
            batch_dict,
            return_images=False,
            return_reconstructions=self.return_reconstructions,
        )
        results_dict['metadata'] = {
            'video': batch_dict['video'],
            'idx': batch_dict['idx'],
            'image_paths': batch_dict['image_path'],
        }
        return results_dict
