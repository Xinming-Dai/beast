"""High-level Model API for training and running inference with BEAST models."""

import contextlib
import logging
import os
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch

from beast.inference import predict_images, predict_video
from beast.io import load_config, validate_config
from beast.logging import log_step
from beast.models.base import BaseLightningModel
from beast.models.pca import PCAAutoencoder
from beast.models.resnets import ResnetAutoencoder
from beast.models.sable import Sable
from beast.models.vits import VisionTransformer
from beast.train import train

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

    MODEL_REGISTRY = {
        'vit': VisionTransformer,
        'resnet': ResnetAutoencoder,
        'sable': Sable,
        'pca': PCAAutoencoder,
    }

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
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
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
            config = validate_config(config_path)

        model_type = config['model'].get('model_class', '').lower()
        if model_type not in cls.MODEL_REGISTRY:
            raise ValueError(f'Unknown model type: {model_type}')

        # Initialize the LightningModule
        model_class = cls.MODEL_REGISTRY[model_type]
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
        with chdir(self.model_dir):
            if isinstance(self.model, Sable):
                from beast.train_sable import train_sable
                self.model = train_sable(self.config, self.model, output_dir=self.model_dir)
            else:
                self.model = train(self.config, self.model, output_dir=self.model_dir)

    def infer_sable(
        self,
        dataset_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        vda_cache_root: str | Path | None = None,
        correspondence_cache_root: str | Path | None = None,
        splits: list[str] | None = None,
        save_visuals: bool = False,
        save_render_views: bool = False,
        save_pointclouds: bool = True,
        save_camera_pointcloud_scene: bool = False,
        load_gt_camera_params_for_vis: bool = False,
        compute_metrics: bool = False,
        use_segmentation_mask: bool = False,
        segmentation_root: str | Path | None = None,
        max_batches: int | None = None,
        session_names: list[str] | str | None = None,
        max_files_per_session: int | None = None,
        neural_input_dir: str | Path | None = None,
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
            save_render_views: whether to save one render-only PNG per view per sample,
                in addition to the combined grid from ``save_visuals``.
            save_pointclouds: whether to save ``.ply`` Gaussian-center point clouds.
            save_camera_pointcloud_scene: whether to save ``.glb`` scenes (point cloud +
                camera frustums, with a ground-truth overlay when
                ``load_gt_camera_params_for_vis`` is also set).
            load_gt_camera_params_for_vis: for Cheese3D datasets, load ground-truth
                camera calibration into ``gt_c2w``/``gt_fxfycxcy`` for visualization,
                overriding the saved training config's
                ``training.load_gt_camera_params_for_vis``.
            compute_metrics: whether to compute per-view PSNR/SSIM on the predicted
                renders and save them to ``output_dir/psnr_ssim_metrics.npz``.
            use_segmentation_mask: whether to apply segmentation masks (zeroing the
                background) to renders/targets before metrics and saved PNGs. Requires
                segmentation masks for this dataset/split; overrides the saved training
                config's ``training.use_segmentation.enabled``.
            segmentation_root: overrides ``training.use_segmentation.cache_root``; root
                directory containing
                ``segmentation_masks/{session_id}/{cam}/mask{frame_idx:08d}.png``. Only
                used with ``use_segmentation_mask``.
            max_batches: stop after this many batches; None runs the full dataset.
            session_names: session IDs to load. Accepts a list or a single string.
                When ``None``, the value from the saved training config is used.
            max_files_per_session: cap on the number of PLY/GLB files saved per
                session; when set, outputs are grouped into per-session subfolders.
                ``None`` (default) saves every item into the flat, unlimited layout.
            neural_input_dir: neural-data root laid out as
                ``{neural_input_dir}/{session_id}/{session_id}_aligned.npz``. With
                ``compute_metrics``, organizes PSNR/SSIM per session as
                ``[K neural trials, T neural bins, V views]`` aligned to that file and
                writes ``output_dir/{session_id}/psnr_ssim_metrics.npz`` instead of the
                flat ``output_dir/psnr_ssim_metrics.npz``.

        Returns:
            dict with keys 'output_dir', 'num_batches', 'ply_files',
            'camera_pointcloud_scene_glb_files', 'vis_files', 'render_view_files',
            'metrics_npz', 'neural_metrics_npz', 'average_psnr', 'average_ssim'.
        """
        from beast.inference import infer_sable as _infer_sable

        config = {**self.config}
        config['inference'] = True
        config['training'] = {**config.get('training', {})}
        if dataset_path is not None:
            config['training']['dataset_path'] = str(dataset_path)
        if session_names is not None:
            config['training']['session_names'] = session_names
        if load_gt_camera_params_for_vis:
            config['training']['load_gt_camera_params_for_vis'] = True
        if use_segmentation_mask:
            seg_cfg = {**config['training'].get('use_segmentation', {})}
            seg_cfg['enabled'] = True
            if segmentation_root is not None:
                seg_cfg['cache_root'] = str(segmentation_root)
            config['training']['use_segmentation'] = seg_cfg
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
            save_render_views=save_render_views,
            save_pointclouds=save_pointclouds,
            save_camera_pointcloud_scene=save_camera_pointcloud_scene,
            compute_metrics=compute_metrics,
            require_segmentation_mask=use_segmentation_mask,
            max_batches=max_batches,
            include_splits=splits,
            max_files_per_session=max_files_per_session,
            neural_input_dir=neural_input_dir,
        )

    def extract_sable_latents(
        self,
        dataset_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        vda_cache_root: str | Path | None = None,
        correspondence_cache_root: str | Path | None = None,
        splits: list[str] | None = None,
        latent_types: list[str] | None = None,
        max_batches: int | None = None,
        session_names: list[str] | str | None = None,
        resume: bool = True,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Extract and save per-batch Sable latent tensors for downstream encoding/decoding.

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
            latent_types: subset of ``['frame_z', 'dino_z', 'combined_z', 'img_tokens']``;
                ``None`` exports all four.
            max_batches: stop after this many batches; None runs the full dataset.
            session_names: session IDs to load. Accepts a list or a single string.
                When ``None``, the value from the saved training config is used.
            resume: when ``True`` (default), skip a batch's forward pass entirely if every
                requested output file for that batch already exists, and skip the whole run
                immediately if the combined trials output already exists.
            batch_size: overrides the saved training config's ``training.batch_size_per_gpu``
                when given.

        Returns:
            dict with keys 'output_dir', 'num_batches', 'num_batches_skipped', 'saved_files',
            'combined_trials_files'.
        """
        from beast.inference import extract_sable_latents as _extract_sable_latents

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

        return _extract_sable_latents(
            config=config,
            model=self.model,
            output_dir=output_dir,
            latent_types=latent_types,
            max_batches=max_batches,
            include_splits=splits,
            resume=resume,
            batch_size=batch_size,
        )

    def predict_images(
        self,
        image_dir: str | Path,
        output_dir: str | Path | None = None,
        batch_size: int = 32,
        save_latents: bool = True,
        save_reconstructions: bool = True,
        save_img_tokens: bool = False,
        compute_metrics: bool = False,
        use_segmentation_mask: bool = False,
        segmentation_root: str | Path | None = None,
        mask_session_id: str | None = None,
        mask_camera_role: str | None = None,
        save_render_views: bool = False,
    ) -> dict[str, Any]:
        """Run inference on a possibly nested directory of images.

        Parameters
        ----------
        image_dir: absolute path to possibly nested image directories
        output_dir: absolute path to directory where results are saved
        batch_size: batch size for inference
        save_latents: save latents for each image as a numpy file
        save_reconstructions: save reconstructed images
        save_img_tokens: save the per-patch token grid and its matching ids_restore, for later
            decoding a frame from saved tokens
        compute_metrics: whether to compute per-sample PSNR/SSIM against the input image and
            save them to output_dir/psnr_ssim_metrics.npz
        use_segmentation_mask: whether to zero out the background (via segmentation_root masks)
            in reconstructions and inputs before metrics/PNG saving
        segmentation_root: root directory holding a mask PNG for every image; required when
            use_segmentation_mask is True. By default, masks are resolved by mirroring the
            image's path relative to image_dir; when mask_session_id and mask_camera_role are
            also given, masks are instead resolved via each image's eval-layout
            frame_index_mapping.json (see beast.data.datasets.BaseDataset)
        mask_session_id: session id segment of the eval-layout mask path; required together
            with mask_camera_role
        mask_camera_role: 'left' or 'right', for eval-layout mask resolution; required
            together with mask_session_id
        save_render_views: whether to save one render-only PNG per sample under
            output_dir/png_render_only/

        Returns
        -------
        Predictions and latents

        """
        image_dir = Path(image_dir)
        if self.model_dir is None:
            raise ValueError('model_dir is None; call train() before predict_images()')
        if use_segmentation_mask and segmentation_root is None:
            raise ValueError('segmentation_root must be set when use_segmentation_mask is True')
        if (mask_session_id is None) != (mask_camera_role is None):
            raise ValueError('mask_session_id and mask_camera_role must be set together')
        outputs = predict_images(
            model=self.model,
            output_dir=output_dir or self.model_dir / 'image_predictions' / image_dir.stem,
            source_dir=image_dir,
            batch_size=batch_size,
            save_latents=save_latents,
            save_reconstructions=save_reconstructions,
            save_img_tokens=save_img_tokens,
            compute_metrics=compute_metrics,
            use_segmentation_mask=use_segmentation_mask,
            segmentation_root=segmentation_root,
            mask_session_id=mask_session_id,
            mask_camera_role=mask_camera_role,
            save_render_views=save_render_views,
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
