"""High-level Model API for training and running inference with BEAST models."""

import contextlib
import logging
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch

from beast.config import get_beast_config_class
from beast.inference import predict_images, predict_video
from beast.io import load_config
from beast.logging import log_step
from beast.models.base import BaseLightningModel
from beast.models.registry import MODEL_REGISTRY, TRAIN_REGISTRY

_logger = logging.getLogger(__name__)


# TODO: Replace with contextlib.chdir in python 3.11.
@contextlib.contextmanager
def chdir(dir: str | Path) -> Generator[None, None, None]:
    """Context manager that temporarily changes the working directory.

    Parameters
    ----------
    dir: directory to change to for the duration of the context

    """
    pwd = os.getcwd()
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(pwd)


class Model:
    """High-level API wrapper for BEAST models.

    This class manages both the model and the training/inference processes.
    """

    def __init__(
        self,
        model: BaseLightningModel,
        config: dict[str, Any],
        model_dir: str | Path | None = None
    ) -> None:
        """Initialize with model and config."""
        self.model = model
        self.config = config
        self.model_dir = Path(model_dir) if model_dir is not None else None

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> 'Model':
        """Load a model from a directory.

        Parameters
        ----------
        model_dir: Path to directory containing model checkpoint and config

        Returns
        -------
        Initialized model wrapper

        """

        model_dir = Path(model_dir)

        config_path = model_dir / 'config.yaml'
        config = load_config(config_path)

        model_type = config['model'].get('model_class', '').lower()
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = MODEL_REGISTRY[model_type]
        model = model_class(config)

        _logger.info(f'Loaded a {model_class} model')

        # Load best weights
        checkpoint_path = list(model_dir.rglob('*best.ckpt'))[0]
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
        _logger.info(f'Loaded model weights from {checkpoint_path}')

        return cls(model, config, model_dir)

    @classmethod
    def from_config(cls, config_path: str | Path | dict) -> 'Model':
        """Create a new model from a config file.

        Parameters
        ----------
        config_path: Path to config file or config dict

        Returns
        -------
        Initialized model wrapper

        """
        if not isinstance(config_path, dict):
            config = load_config(config_path)
        else:
            model_class = (config_path.get('model') or {}).get('model_class', '')
            config = get_beast_config_class(model_class).model_validate(config_path).model_dump()

        model_type = config['model'].get('model_class', '').lower()
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = MODEL_REGISTRY[model_type]
        log_step(f"Creating {model_type} model instance", level='debug')
        log_step(
            f"About to call {model_class.__name__}.__init__() - this may take several"
            ' minutes if downloading pretrained weights',
            level='debug',
        )
        init_start = time.time()
        model = model_class(config)
        init_duration = time.time() - init_start
        log_step(f"Model initialization completed in {init_duration:.2f} seconds", level='debug')

        _logger.info(f'Initialized a {model_class} model')

        return cls(model, config, model_dir=None)

    def train(self, output_dir: str | Path = 'runs/default') -> None:
        """Train the model using PyTorch Lightning.

        Dispatches to an Sable-specific training loop for Sable models and
        to the generic beast training loop for all other model types.

        Parameters
        ----------
        output_dir: Directory to save checkpoints

        """
        self.model_dir = Path(output_dir)
        model_type = self.config['model'].get('model_class', '').lower()
        train_fn = TRAIN_REGISTRY[model_type]
        with chdir(self.model_dir):
            self.model = train_fn(self.config, self.model, output_dir=self.model_dir)

    def infer_sable(
        self,
        dataset_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        vda_cache_root: str | Path | None = None,
        correspondence_cache_root: str | Path | None = None,
        splits: list[str] | None = None,
        save_visuals: bool = False,
        max_batches: int | None = None,
        session_names: list[str] | str | None = None,
    ) -> dict[str, Any]:
        """Run Sable inference over a scene dataset and save PLY point clouds.

        Args:
            dataset_path: path to the scene dataset. For IBL: raw frames root
                (``leftCamera.video/`` / ``rightCamera.video/`` layout). For
                Cheese3D: root Cheese3D directory.
            output_dir: root directory for outputs; defaults to <model_dir>/inference.
            vda_cache_root: root directory of precomputed VDA depth cache. When
                ``None``, the value from the saved training config is used.
            correspondence_cache_root: root directory of precomputed correspondence
                cache. When ``None``, the value from the saved training config is used.
            splits: dataset splits to run inference on (default: ['train', 'val']).
            save_visuals: whether to also save render-vs-target PNG grids.
            max_batches: stop after this many batches; None runs the full dataset.
            session_names: session IDs to load. Accepts a list or a single string.
                When ``None``, the value from the saved training config is used.

        Returns:
            dict with keys 'output_dir', 'num_batches', 'ply_files', 'vis_files'.
        """
        from beast.inference import infer_sable as _infer_sable

        config = {**self.config}
        config['inference'] = True
        config['training'] = {**config.get('training', {})}
        if dataset_path is not None:
            config['training']['dataset_path'] = str(dataset_path)
        if session_names is not None:
            config['training']['session_names'] = session_names
        if vda_cache_root is not None:
            config['model'] = {**config.get('model', {})}
            config['model']['vda'] = {**config['model'].get('vda', {})}
            config['model']['vda']['cache_root'] = str(vda_cache_root)
        if correspondence_cache_root is not None:
            config['model'] = config.get('model', {})
            config['model']['merge_pcd'] = {**config['model'].get('merge_pcd', {})}
            config['model']['merge_pcd']['correspondence_cache_root'] = str(correspondence_cache_root)

        output_dir = Path(output_dir) if output_dir else (self.model_dir or Path('inference'))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        return _infer_sable(
            config=config,
            model=self.model,
            output_dir=output_dir,
            save_visuals=save_visuals,
            max_batches=max_batches,
            include_splits=splits,
        )

    def predict_images(
        self,
        image_dir: str | Path,
        output_dir: str | Path | None = None,
        batch_size: int = 32,
        save_latents: bool = True,
        save_reconstructions: bool = True,
    ) -> dict[str, Any]:
        """Run inference on a possibly nested directory of images.

        Parameters
        ----------
        image_dir: absolute path to possibly nested image directories
        output_dir: absolute path to directory where results are saved
        batch_size: batch size for inference
        save_latents: save latents for each image as a numpy file
        save_reconstructions: save reconstructed images

        Returns
        -------
        Predictions and latents

        """
        image_dir = Path(image_dir)
        if self.model_dir is None:
            raise ValueError('model_dir is None; call train() before predict_images()')
        outputs = predict_images(
            model=self.model,
            output_dir=output_dir or self.model_dir / 'image_predictions' / image_dir.stem,
            source_dir=image_dir,
            batch_size=batch_size,
            save_latents=save_latents,
            save_reconstructions=save_reconstructions,
        )
        return outputs

    def predict_video(
        self,
        video_file: str | Path,
        output_dir: str | Path | None = None,
        batch_size: int = 32,
        save_latents: bool = True,
        save_reconstructions: bool = True,
    ) -> dict[str, Any]:
        """Run inference on a single video.

        Parameters
        ----------
        video_file: absolute path to video file (mp4 or avi)
        output_dir: absolute path to directory where results are saved
        batch_size: batch size for inference
        save_latents: save latents for each image as a numpy file
        save_reconstructions: save reconstructed images

        Returns
        -------
        Inference results dict from the video prediction handler

        """
        video_file = Path(video_file)
        if self.model_dir is None:
            raise ValueError('model_dir is None; call train() before predict_video()')
        return predict_video(
            model=self.model,
            output_dir=output_dir or self.model_dir / 'video_predictions',
            video_file=video_file,
            batch_size=batch_size,
            save_latents=save_latents,
            save_reconstructions=save_reconstructions,
        )
