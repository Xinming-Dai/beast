"""Inference handlers for saving model predictions on images and videos."""

import json
import logging
import os
import pickle
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import yaml
from PIL import Image
from torchvision import transforms
from typeguard import typechecked

from beast.logging import log_step
from beast.data.datasets import _IMAGENET_MEAN, _IMAGENET_STD, BaseDataset
from beast.data.video import VideoFrameIterator
from beast.models.base import BaseLightningModel
from beast.models.model_utils.utils_icp import (
    apply_similarity_transform_to_poses,
    estimate_camera_similarity_transform,
)
from beast.models.model_utils.utils_vis import add_scene_cam

_logger = logging.getLogger(__name__)


class ImagePredictionHandler:
    """Handles saving predictions while preserving directory structure."""

    def __init__(self, output_dir: str | Path, source_dir: str | Path) -> None:
        """Initialize handler with output and source directories.

        Parameters
        ----------
        output_dir: directory where predictions will be saved
        source_dir: root directory of source images, used to preserve directory structure

        """
        self.output_dir = Path(output_dir)
        self.source_dir = Path(source_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # store metadata for each prediction
        self.metadata = []

        # for normalization
        self.mean = torch.Tensor(_IMAGENET_MEAN).view(1, 1, 3)
        self.std = torch.Tensor(_IMAGENET_STD).view(1, 1, 3)

    def tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor (C, H, W) to PIL Image."""
        # Handle different tensor formats
        if tensor.dim() == 4:  # (B, C, H, W) - take first batch item
            tensor = tensor[0]

        # Convert from (C, H, W) to (H, W, C)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)

        # ensure values are in [0, 255] range
        # This gets you back to [0, 1]
        tensor = tensor * self.std + self.mean
        # after getting to [0, 1], scale to [0, 255]
        tensor = torch.clamp(tensor, 0, 1)  # Ensure [0, 1] range
        tensor = tensor * 255.0

        # Convert to uint8 numpy array
        np_array = tensor.detach().cpu().numpy().astype(np.uint8)

        # Handle grayscale vs RGB
        # if np_array.shape[2] == 1:
        #     np_array = np_array.squeeze(2)
        #     return Image.fromarray(np_array, mode='L')
        # else:
        return Image.fromarray(np_array, mode='RGB')

    def save_reconstruction(
        self,
        reconstruction: torch.Tensor,
        video: str,
        idx: int,
        original_path: Path,
    ) -> Path:
        """Save a single reconstruction maintaining directory structure."""
        # Create output subdirectory matching source structure
        output_subdir = self.output_dir / video
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Get original filename
        original_filename = original_path.name
        output_path = output_subdir / original_filename

        # Convert tensor to image and save
        image = self.tensor_to_image(reconstruction)
        image.save(output_path)

        return output_path

    def save_latents(
        self,
        latents: torch.Tensor,
        video: str,
        idx: int,
        original_path: Path,
    ) -> Path:
        """Save latent representations as numpy arrays."""
        # Create latents subdirectory
        latents_dir = self.output_dir / 'latents' / video
        latents_dir.mkdir(parents=True, exist_ok=True)

        # Save as .npy file
        original_stem = original_path.stem
        latents_path = latents_dir / f'{original_stem}.npy'

        # Convert to numpy and save
        latents_np = latents.detach().cpu().numpy()
        np.save(latents_path, latents_np)

        return latents_path

    def process_batch_predictions(
        self,
        predictions: dict,
        batch_metadata: dict,
        save_reconstructions: bool = True,
        save_latents: bool = False
    ) -> dict[str, list]:
        """Process a batch of predictions and save them."""
        reconstructions = predictions['reconstructions']
        latents = predictions['latents']

        batch_size = reconstructions.shape[0]

        saved_files = {
            'reconstructions': [],
            'latents': [],
            'metadata': []
        }

        for i in range(batch_size):
            video = batch_metadata['video'][i]
            idx = batch_metadata['idx'][i].item()

            # Get original image path for this item
            original_path = batch_metadata['image_paths'][i]

            # Initialize metadata entry
            metadata_entry = {
                'original_path': str(original_path),
                'video': video,
                'idx': idx
            }

            # Save reconstruction if requested
            if save_reconstructions:
                recon_path = self.save_reconstruction(
                    reconstructions[i], video, idx, Path(original_path),
                )
                saved_files['reconstructions'].append(str(recon_path))
                metadata_entry['reconstruction_path'] = str(recon_path)

            # Save latents if requested
            if save_latents:
                latents_path = self.save_latents(latents[i], video, idx, Path(original_path))
                saved_files['latents'].append(str(latents_path))
                metadata_entry['latents_path'] = str(latents_path)

            saved_files['metadata'].append(metadata_entry)
            self.metadata.append(metadata_entry)

        return saved_files

    def process_predictions(
        self,
        predictions: list,
        save_reconstructions: bool = True,
        save_latents: bool = False,
    ) -> dict[str, Any]:
        """Process all predictions from trainer.predict() and save results.

        Parameters
        ----------
        predictions: List of prediction dictionaries from trainer.predict()
        save_reconstructions: Whether to save reconstruction images
        save_latents: Whether to save latent representations

        Returns
        -------
        Dictionary with summary of saved files and metadata

        """
        all_saved_files = {
            'reconstructions': [],
            'latents': [],
            'metadata': []
        }

        for batch_predictions in predictions:
            # Extract metadata from predictions
            batch_metadata = batch_predictions['metadata']

            # Process this batch
            saved_files = self.process_batch_predictions(
                batch_predictions,
                batch_metadata,
                save_reconstructions=save_reconstructions,
                save_latents=save_latents
            )

            # Accumulate results
            all_saved_files['reconstructions'].extend(saved_files['reconstructions'])
            all_saved_files['latents'].extend(saved_files['latents'])
            all_saved_files['metadata'].extend(saved_files['metadata'])

        # Save metadata summary
        metadata_path = self.save_metadata_summary()

        # Create results summary
        results = {
            'output_dir': str(self.output_dir),
            'num_images_processed': len(all_saved_files['metadata']),
            'metadata_file': str(metadata_path)
        }

        if save_reconstructions:
            results['reconstructions_saved'] = len(all_saved_files['reconstructions'])
            results['reconstructions_dir'] = str(self.output_dir)

        if save_latents:
            results['latents_saved'] = len(all_saved_files['latents'])
            results['latents_dir'] = str(self.output_dir / "latents")

        _logger.info(f"Processed {results['num_images_processed']} images")
        if save_reconstructions:
            _logger.info(f"Saved {results['reconstructions_saved']} reconstructions")
        if save_latents:
            _logger.info(f"Saved {results['latents_saved']} latent representations")
        _logger.info(f'Results saved to: {self.output_dir}')
        _logger.info(f'Metadata saved to: {metadata_path}')

        return results

    def save_metadata_summary(self) -> Path:
        """Save complete metadata summary to YAML."""
        metadata_path = self.output_dir / 'prediction_metadata.yaml'
        with open(metadata_path, 'w') as f:
            yaml.safe_dump(self.metadata, f)
        return metadata_path


class VideoPredictionHandler:
    """Handles saving predictions for video processing."""

    def __init__(self, output_dir: str | Path, video_file: str | Path) -> None:
        """Initialize the video prediction handler.

        Parameters
        ----------
        output_dir: directory where results will be saved
        video_file: absolute path to the source video file

        """
        self.output_dir = Path(output_dir)
        self.video_file = Path(video_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # get video properties for output video
        cap = cv2.VideoCapture(str(self.video_file))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Store metadata and latents
        self.metadata = {
            'video_file': str(self.video_file),
            'output_dir': str(self.output_dir),
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames
        }

        # Accumulate latents and reconstructions
        self.all_latents = []
        self.reconstruction_writer = None
        self.frames_processed = 0

        # for normalization
        self.mean = torch.Tensor(_IMAGENET_MEAN).view(1, 1, 3)
        self.std = torch.Tensor(_IMAGENET_STD).view(1, 1, 3)

    def tensor_to_numpy_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor (C, H, W) to OpenCV BGR format."""
        # handle different tensor formats
        if tensor.dim() == 4:  # (B, C, H, W) - take first batch item
            tensor = tensor[0]

        # convert from (C, H, W) to (H, W, C)
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)

        # ensure values are in [0, 255] range
        # This gets you back to [0, 1]
        tensor = tensor * self.std + self.mean
        # after getting to [0, 1], scale to [0, 255]
        tensor = torch.clamp(tensor, 0, 1)  # Ensure [0, 1] range
        tensor = tensor * 255.0

        # convert to uint8 numpy array
        np_array = tensor.detach().cpu().numpy().astype(np.uint8)

        # convert RGB to BGR for OpenCV
        # if np_array.shape[2] == 3:
        np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        # elif np_array.shape[2] == 1:
        #     # Convert grayscale to BGR
        #     np_array = cv2.cvtColor(np_array.squeeze(2), cv2.COLOR_GRAY2BGR)

        return np_array

    def _init_video_writer(self) -> None:
        """Initialize the video writer for saving reconstructions."""
        if self.reconstruction_writer is None:
            output_video_path = self.output_dir / f'{self.video_file.stem}_reconstruction.mp4'

            # Use mp4v codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]

            self.reconstruction_writer = cv2.VideoWriter(
                str(output_video_path),
                fourcc,
                self.fps,
                # (self.width, self.height)
                (224, 224),  # hard-code to model output size for now
            )

            if not self.reconstruction_writer.isOpened():
                raise ValueError(f'Failed to open video writer for {output_video_path}')

            self.metadata['reconstruction_video'] = str(output_video_path)

    def process_batch_predictions(
        self,
        predictions: dict,
        save_reconstructions: bool = True,
        save_latents: bool = True
    ) -> dict[str, Any]:
        """Process a batch of predictions."""

        latents = predictions['latents']
        batch_size = latents.shape[0]

        # process each frame in the batch
        for i in range(batch_size):
            # save latents - accumulate all latents
            if save_latents:
                # convert to numpy and store
                latent_np = latents[i].detach().cpu().numpy()
                self.all_latents.append(latent_np)

            # save reconstruction frame to video
            if save_reconstructions:
                reconstructions = predictions['reconstructions']

                if self.reconstruction_writer is None:
                    self._init_video_writer()
                if self.reconstruction_writer is None:
                    raise RuntimeError('reconstruction_writer was not initialized')

                # convert tensor to BGR numpy array
                frame_bgr = self.tensor_to_numpy_bgr(reconstructions[i])

                # # resize if necessary
                # if frame_bgr.shape[:2] != (self.height, self.width):
                #     frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

                # write frame to video
                self.reconstruction_writer.write(frame_bgr)

            self.frames_processed += 1

        return {
            'frames_processed': batch_size
        }

    def process_predictions(
        self,
        predictions: list,
        save_reconstructions: bool = True,
        save_latents: bool = True,
    ) -> dict[str, Any]:
        """Process all predictions from trainer.predict() and save results.

        Parameters
        ----------
        predictions: list of prediction dictionaries from trainer.predict()
        save_reconstructions: whether to save reconstruction video
        save_latents: whether to save latent representations

        Returns
        -------
        dictionary with summary of saved files and metadata

        """
        # process all batches
        for batch_predictions in predictions:
            self.process_batch_predictions(
                batch_predictions,
                save_reconstructions=save_reconstructions,
                save_latents=save_latents
            )

        # finalize outputs
        results = {
            'output_dir': str(self.output_dir),
            'video_file': str(self.video_file),
            'frames_processed': self.frames_processed,
        }

        # save concatenated latents
        if save_latents and self.all_latents:
            latents_array = np.stack(self.all_latents, axis=0)
            latents_path = self.output_dir / f'{self.video_file.stem}.npy'
            np.save(latents_path, latents_array)

            results['latents_file'] = str(latents_path)
            results['latents_shape'] = latents_array.shape
            self.metadata['latents_file'] = str(latents_path)
            self.metadata['latents_shape'] = list(latents_array.shape)

        else:
            results['latents_file'] = None
            results['latents_shape'] = None

        # close video writer
        if save_reconstructions and self.reconstruction_writer is not None:
            self.reconstruction_writer.release()
            results['reconstruction_video'] = self.metadata.get('reconstruction_video')
        else:
            results['reconstruction_video'] = None

        _logger.info(f'Processed {self.frames_processed} frames from {self.video_file.name}')
        if save_reconstructions:
            _logger.info(f'Saved reconstruction video: {results.get("reconstruction_video")}')
        if save_latents:
            _logger.info(
                f'Saved latents array {results.get("latents_shape")} to: '
                f'{results.get("latents_file")}'
            )

        return results


def predict_images(
    model: BaseLightningModel,
    output_dir: str | Path,
    source_dir: str | Path,
    batch_size: int = 32,
    save_latents: bool = True,
    save_reconstructions: bool = True,
    num_channels: int = 3,
) -> dict[str, Any]:
    """Run inference on images using a trained model and save results.

    Processes all images in a directory (including nested subdirectories) through
    a trained PyTorch Lightning model, generating reconstructions and/or latent
    representations. Preserves the original directory structure in the output.

    Parameters
    ----------
    model: trained Beast model for inference
    output_dir: directory where results will be saved; creates subdirectories matching the source
        directory structure
    source_dir: directory containing input images; supports nested directory structures
    batch_size: number of images to process in each batch
    save_latents: whether to save latent representations as .npy files in a 'latents/' subdirectory
    save_reconstructions: whether to save reconstructed images as PNG files
    num_channels: number of image channels; 1 loads as grayscale then converts to RGB, 3 loads
        as RGB

    Returns
    -------
    Dictionary containing inference results with keys:
        - 'output_dir': Path to output directory
        - 'num_images_processed': Total number of images processed
        - 'metadata_file': Path to YAML metadata summary file
        - 'reconstructions_saved': Number of reconstructions saved (if enabled)
        - 'latents_saved': Number of latent files saved (if enabled)
        - 'reconstructions_dir': Path to reconstructions directory (if enabled)
        - 'latents_dir': Path to latents directory (if enabled)

    """

    output_dir = Path(output_dir)
    source_dir = Path(source_dir)

    # initialize prediction handler
    handler = ImagePredictionHandler(output_dir, source_dir)

    # dataset
    dataset = BaseDataset(
        data_dir=source_dir,
        imgaug_pipeline=None,
        num_channels=num_channels,
    )

    # dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )

    # configure model predict behavior before handing off to trainer
    model.return_reconstructions = save_reconstructions

    # run inference
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    predictions = trainer.predict(model, dataloaders=dataloader, return_predictions=True)
    if predictions is None:
        raise RuntimeError('trainer.predict() returned None')

    # process outputs
    results = handler.process_predictions(
        predictions,
        save_reconstructions=save_reconstructions,
        save_latents=save_latents,
    )

    return results


def combine_view_latents(
    latents_dir: str | Path,
    output_path: str | Path,
    views: tuple[str, str] = ('left', 'right'),
) -> Path:
    """Pair per-view latents saved by `predict_images` into a single (n_frames, 2, dim) array.

    `predict_images` has no notion of view and saves one `.npy` latent file per image, under
    `latents_dir/<view>/<frame_stem>.npy` (view is the image's parent directory name, e.g.
    'left' or 'right'). This function matches files across the two view subdirectories by
    frame stem and stacks each matched pair into a single per-view latent vector, so that
    ViT/ResNet latents can be consumed the same way as SABLE's (batch, 2, dim) latents.

    Parameters
    ----------
    latents_dir: directory containing one subdirectory per view (as written by
        `ImagePredictionHandler.save_latents`)
    output_path: path to the output .npz file
    views: names of the two view subdirectories to pair, in output order

    Returns
    -------
    path to the saved .npz file, containing 'z' (n_frames, 2, dim) and 'frame_ids' (n_frames,)

    """
    latents_dir = Path(latents_dir)
    output_path = Path(output_path)

    view_a_dir = latents_dir / views[0]
    view_b_dir = latents_dir / views[1]
    if not view_a_dir.is_dir() or not view_b_dir.is_dir():
        raise ValueError(f'expected latent subdirectories {view_a_dir} and {view_b_dir}')

    stems_a = {p.stem for p in view_a_dir.glob('*.npy')}
    stems_b = {p.stem for p in view_b_dir.glob('*.npy')}
    frame_ids = sorted(stems_a & stems_b)
    if not frame_ids:
        raise ValueError(f'no matching frame stems found between {view_a_dir} and {view_b_dir}')
    missing = (stems_a | stems_b) - set(frame_ids)
    if missing:
        _logger.warning(f'{len(missing)} frames missing a matching pair and will be skipped')

    paired_latents = np.stack([
        np.stack([
            np.load(view_a_dir / f'{frame_id}.npy'),
            np.load(view_b_dir / f'{frame_id}.npy'),
        ])
        for frame_id in frame_ids
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, z=paired_latents, frame_ids=np.array(frame_ids))
    _logger.info(f'Saved paired latents {paired_latents.shape} to: {output_path}')

    return output_path


def predict_video(
    model: BaseLightningModel,
    output_dir: str | Path,
    video_file: str | Path,
    batch_size: int = 32,
    save_latents: bool = True,
    save_reconstructions: bool = True,
) -> dict[str, Any]:
    """Run inference on video using a trained model and save results.

    Parameters
    ----------
    model: trained Beast model for inference
    output_dir: directory where results will be saved
    video_file: absolute path to video file (mp4 or avi)
    batch_size: number of images to process in each batch
    save_latents: whether to save latent representations as .npy files in a 'latents/' subdirectory
    save_reconstructions: whether to save reconstructed images as PNG files

    Returns
    -------
    Dictionary containing inference results with keys:
        - 'output_dir': path to output directory
        - 'video_file': path to source video file
        - 'frames_processed': total number of frames processed
        - 'reconstruction_video': path to reconstruction video (if enabled, else None)
        - 'latents_file': path to saved latents array (if enabled, else None)
        - 'latents_shape': shape of latents array (if enabled, else None)

    """

    output_dir = Path(output_dir)
    video_file = Path(video_file)

    # initialize prediction handler
    handler = VideoPredictionHandler(output_dir, video_file)

    # dataloader
    dataloader = VideoFrameIterator(
        video_file=video_file,
        batch_size=batch_size,
    )

    # configure model predict behavior before handing off to trainer
    model.return_reconstructions = save_reconstructions

    # run inference
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False)
    predictions = trainer.predict(model, dataloaders=dataloader, return_predictions=True)
    if predictions is None:
        raise RuntimeError('trainer.predict() returned None')

    # process outputs
    return handler.process_predictions(
        predictions,
        save_reconstructions=save_reconstructions,
        save_latents=save_latents,
    )


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    """Load and resize an image to a square tensor in [0, 1].

    Args:
        path: path to the image file.
        image_size: target height and width in pixels.

    Returns:
        float tensor of shape [3, image_size, image_size].
    """
    img = Image.open(path).convert('RGB')
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return tf(img)


@typechecked
def predict_sable(
    model: BaseLightningModel,
    view_images: list[list[str | Path]],
    image_size: int = 320,
    input_indices: list[int] | None = None,
    target_indices: list[int] | None = None,
    device: str = 'cuda',
) -> list[dict[str, Any]]:
    """Run Sable inference on one or more scenes, each with multiple views.

    Each element of ``view_images`` is a list of image paths representing the V
    views for one scene (batch item).  All scenes must have the same number of
    views.

    Args:
        model: trained Sable model.
        view_images: outer list length B (batch), inner list length V (views).
            Each path points to an RGB image for that view.
        image_size: images are resized to (image_size × image_size) before
            being passed to the model.
        input_view_indices: 0-based view indices to use as encoder inputs.  When
            None, all views are used as input.
        target_view_indices: 0-based view indices to render.  When None, all views
            are rendered.
        device: torch device string ('cuda' or 'cpu').

    Returns:
        list of dicts (one per batch item) with keys:
            - 'render': rendered images tensor [V_target, 3, H, W] in [0, 1].
            - 'render_video': interpolated video tensor [T, 3, H, W] or None.
            - 'c2w': predicted camera-to-world matrices [V, 4, 4].
            - 'depth_output': VDA depth maps [V_input, H, W].
    """
    if not view_images:
        return []

    num_views = len(view_images[0])
    if any(len(views) != num_views for views in view_images):
        raise ValueError('all scenes must have the same number of views')

    # build image batch [B, V, 3, H, W]
    batch_tensors = []
    for views in view_images:
        view_tensors = torch.stack(
            [_load_image_tensor(Path(p), image_size) for p in views],
            dim=0,
        )  # [V, 3, H, W]
        batch_tensors.append(view_tensors)
    images = torch.stack(batch_tensors, dim=0).to(device)  # [B, V, 3, H, W]

    batch_size = images.shape[0]

    # build index tensors
    if input_indices is None:
        idx_input_view = torch.arange(num_views, device=device).unsqueeze(0).expand(batch_size, -1)
    else:
        idx_input_view = torch.tensor(input_indices, device=device).unsqueeze(0).expand(batch_size, -1)

    if target_indices is None:
        idx_target_view = torch.arange(num_views, device=device).unsqueeze(0).expand(batch_size, -1)
    else:
        idx_target_view = torch.tensor(target_indices, device=device).unsqueeze(0).expand(batch_size, -1)

    batch_dict = {
        'image': images,
        'input_indices': idx_input_view,
        'target_indices': idx_target_view,
    }

    model = model.to(device).eval()
    with torch.no_grad():
        out = model.get_model_outputs(batch_dict)

    results = []
    for b_i in range(batch_size):
        render = out['render'][b_i].clamp(0.0, 1.0)         # [V_target, 3, H, W]
        render_video = (
            out['render_video'][b_i].clamp(0.0, 1.0)
            if out.get('render_video') is not None
            else None
        )
        results.append({
            'render': render,
            'render_video': render_video,
            'c2w': out['c2w'][b_i],
            'depth_output': out['depth_output'][b_i] if out['depth_output'] is not None else None,
        })

    return results


def _write_ply_ascii(path: Path, xyz: np.ndarray, rgb01: np.ndarray) -> None:
    """Write a point cloud as an ASCII PLY file without open3d.

    Args:
        path: output .ply file path.
        xyz: float array of shape [N, 3] with x, y, z coordinates.
        rgb01: float array of shape [N, 3] with RGB values in [0, 1].
    """
    rgb_u8 = (rgb01 * 255.0).round().clip(0, 255).astype(np.uint8)
    with open(path, 'w') as f:
        f.write(
            'ply\nformat ascii 1.0\nelement vertex %d\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'end_header\n' % len(xyz)
        )
        for i in range(len(xyz)):
            x, y, z = xyz[i]
            r, g, b = rgb_u8[i]
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n')


def _flatten_mask_for_points(
    mask: torch.Tensor | None,
    sample_idx: int,
    num_points: int,
) -> np.ndarray | None:
    """Flatten a per-sample foreground mask to align with a flattened point array.

    Args:
        mask: mask tensor with a leading batch dimension; either spatial
            ``[B, V, 1, H, W]`` (foreground/background aligned with pixel-aligned
            xyz/rgb) or patch-major ``[B, V, N]`` (aligned with a flat gaussian xyz
            ordering).  ``None`` if unavailable.
        sample_idx: batch index to select.
        num_points: expected number of points ``N`` after flattening; used to guard
            against ordering mismatches.

    Returns:
        float array of shape ``[N, 1]`` with values in ``{0, 1}`` (1 = foreground),
        or ``None`` if the mask is unavailable or its size doesn't match ``num_points``.
    """
    if mask is None or not torch.is_tensor(mask) or sample_idx >= mask.shape[0]:
        return None
    sample_mask = mask[sample_idx].detach().float().cpu()
    if sample_mask.dim() == 4:  # [V, 1, H, W] -> match permute(0, 2, 3, 1).reshape(-1, 1)
        sample_mask = sample_mask.permute(0, 2, 3, 1)
    mask01 = sample_mask.reshape(-1, 1).numpy()
    return mask01 if mask01.shape[0] == num_points else None


def _extract_pointcloud_xyz_rgb(
    result: dict,
    sample_idx: int,
    gs,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract point-cloud xyz/rgb for one batch sample from a Sable result dict.

    Uses pixel-aligned RGB from input images when ``result['pixelalign_xyz']`` and
    ``result['image']`` are both present and have matching point counts; otherwise
    falls back to opacity-as-grayscale coloring.  When a segmentation mask is present
    (``result['target_mask']`` or ``result['target_gaussian_mask']``), background
    points are recolored white so unregularized background geometry doesn't clutter
    the point cloud.

    Args:
        result: dict returned by ``model.get_model_outputs(batch)``.
        sample_idx: batch index to extract.
        gs: GaussianModel for this batch item, used as the opacity-grayscale fallback.

    Returns:
        tuple of ``(xyz, rgb01, used_pixel_colors)``: xyz is a float array of shape
        [N, 3]; rgb01 is a float array of shape [N, 3] with values in [0, 1];
        used_pixel_colors indicates whether per-pixel RGB (vs. opacity grayscale) was
        used.
    """
    pixelalign_xyz = result.get('pixelalign_xyz')
    images = result.get('image')
    use_pixel_colors = (
        pixelalign_xyz is not None
        and images is not None
        and torch.is_tensor(pixelalign_xyz)
        and torch.is_tensor(images)
        and sample_idx < pixelalign_xyz.shape[0]
        and sample_idx < images.shape[0]
    )
    target_mask = result.get('target_mask')
    target_gaussian_mask = result.get('target_gaussian_mask')

    rgb01 = None
    if use_pixel_colors:
        xyz = (
            pixelalign_xyz[sample_idx]
            .detach().float().cpu()
            .permute(0, 2, 3, 1)
            .reshape(-1, 3)
            .numpy()
        )
        xyz[:, [1, 2]] *= -1
        rgb_candidate = (
            images[sample_idx]
            .detach().float().cpu()
            .clamp(0, 1)
            .permute(0, 2, 3, 1)
            .reshape(-1, 3)
            .numpy()
        )
        if xyz.shape[0] == rgb_candidate.shape[0]:
            rgb01 = rgb_candidate
            mask01 = _flatten_mask_for_points(target_mask, sample_idx, xyz.shape[0])
            if mask01 is not None:
                rgb01 = rgb01 * mask01 + (1.0 - mask01)  # background -> white

    used_pixel_colors = rgb01 is not None

    if rgb01 is None:
        xyz = gs.get_xyz.detach().float().cpu().numpy()
        opacity = gs.get_opacity.detach().float().cpu().numpy().squeeze(-1)
        rgb01 = np.clip(np.stack([opacity, opacity, opacity], axis=-1), 0, 1)
        mask01 = _flatten_mask_for_points(target_gaussian_mask, sample_idx, xyz.shape[0])
        if mask01 is not None:
            rgb01 = rgb01 * mask01 + (1.0 - mask01)  # background -> white

    return xyz, rgb01, used_pixel_colors


def save_gaussian_pointclouds(
    result: dict,
    output_dir: str | Path,
    batch_idx: int,
    max_samples: int | None = None,
    session_ids: list[str] | None = None,
    sample_indices: list[int] | None = None,
) -> list[Path]:
    """Save Gaussian centers from one Sable batch output as PLY point clouds.

    Requires ``open3d`` for binary PLY output; falls back to ASCII PLY when not
    installed.

    Args:
        result: dict returned by ``model.get_model_outputs(batch)``.  Must contain
            ``'gaussians'`` (list of GaussianModel, one per batch item) and optionally
            ``'pixelalign_xyz'`` ([B, v_input, 3, H_r, W_r]), ``'image'``
            ([B, V, 3, H, W] in [0, 1]), ``'target_mask'`` ([B, V, 1, H, W]), and
            ``'target_gaussian_mask'`` ([B, V, hh*ww*ph*pw]).
        output_dir: root output directory; PLY files are written under
            ``output_dir / 'ply'``, or ``output_dir / 'ply' / session_ids[sample_idx]``
            when ``session_ids`` is given.
        batch_idx: used in the output filename
            ``pointcloud_batch{batch_idx:04d}_sample{sample_idx:02d}.ply``.
        max_samples: cap on the number of batch items to save.  ``None`` saves all.
        session_ids: one session ID per batch item; when given, files are grouped into
            a per-session subfolder instead of a flat ``ply`` directory.
        sample_indices: batch item indices to save; when given, all other items are
            skipped. ``None`` saves every item (subject to ``max_samples``).

    Returns:
        list of Path objects for the PLY files that were written.
    """
    gaussians_list = result.get('gaussians')
    if not gaussians_list:
        return []

    output_dir = Path(output_dir)
    ply_dir = output_dir / 'ply'

    try:
        import open3d as o3d
        has_o3d = True
    except ImportError:
        has_o3d = False

    saved = []
    for sample_idx, gs in enumerate(gaussians_list):
        if max_samples is not None and sample_idx >= max_samples:
            break
        if sample_indices is not None and sample_idx not in sample_indices:
            continue

        xyz, rgb01, used_pixel_colors = _extract_pointcloud_xyz_rgb(result, sample_idx, gs)
        if xyz.size == 0:
            continue

        sample_dir = ply_dir / session_ids[sample_idx] if session_ids is not None else ply_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_ply = sample_dir / f'pointcloud_batch{batch_idx:04d}_sample{sample_idx:02d}.ply'

        if has_o3d:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb01)
            o3d.io.write_point_cloud(str(out_ply), pcd)
        else:
            _write_ply_ascii(out_ply, xyz, rgb01)

        color_src = 'RGB from input views' if used_pixel_colors else 'opacity grayscale'
        log_step(f'Saved point cloud: {out_ply} ({xyz.shape[0]} points, {color_src})', level='info')
        saved.append(out_ply)

    return saved


def _dedupe_cameras_by_view_index(
    input_idx: torch.Tensor,
    c2w_input: np.ndarray,
    target_idx: torch.Tensor,
    c2w_target: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """Combine input/target camera poses into one array, one entry per unique view index.

    Input and target view index sets commonly overlap (e.g. full-context inference, where
    every view is used as both input and target), in which case ``c2w_input`` and
    ``c2w_target`` hold identical poses for the shared views. Keying by view index instead
    of blindly concatenating avoids drawing duplicate, overlapping frustums for those views.

    Args:
        input_idx: input view indices, shape [v_input].
        c2w_input: input camera-to-world matrices, shape [v_input, 4, 4].
        target_idx: target view indices, shape [v_target].
        c2w_target: target camera-to-world matrices, shape [v_target, 4, 4].

    Returns:
        tuple of (view_indices, c2ws): ``view_indices`` is the sorted list of unique view
        indices (length ``num_unique_views``); ``c2ws`` is the corresponding camera-to-world
        matrices, shape [num_unique_views, 4, 4].
    """
    c2w_by_view_idx = dict(zip(input_idx.tolist(), c2w_input))
    c2w_by_view_idx.update(zip(target_idx.tolist(), c2w_target))
    view_indices = sorted(c2w_by_view_idx)
    return view_indices, np.stack([c2w_by_view_idx[idx] for idx in view_indices], axis=0)


def save_camera_pointcloud_scene(
    result: dict,
    output_dir: str | Path,
    batch_idx: int,
    max_samples: int | None = None,
    screen_width: float | None = None,
    session_ids: list[str] | None = None,
    sample_indices: list[int] | None = None,
) -> list[Path]:
    """Save the predicted point cloud together with camera frustums as a .glb scene.

    Builds one ``trimesh.Scene`` per batch item containing the same point cloud as
    ``save_gaussian_pointclouds`` plus one cone-shaped camera-frustum mesh per unique
    predicted view (colored by view index via the ``hsv`` colormap; input and target
    views that share a view index are deduplicated, see
    ``_dedupe_cameras_by_view_index``), then exports the scene as a single glTF binary
    (``.glb``) file. The result can be opened directly in any local glTF viewer.

    When ``result['gt_c2w']`` ([B, V, 4, 4]) is present, this also overlays
    ground-truth camera frustums (fixed black color) alongside the predicted ones.
    Since predicted poses live in an arbitrary canonical frame (no shared scale/origin
    with GT), a best-fit similarity transform (rotation + isotropic scale + translation,
    via ``estimate_camera_similarity_transform``) is solved from predicted to GT camera
    poses at matching view indices, then applied to both the predicted point cloud and
    predicted frustums before drawing, so predicted and GT geometry end up in the same
    frame. Requires at least one corresponding camera (i.e. some overlap between
    predicted view indices and ``gt_c2w`` row indices); with none, the GT overlay is
    skipped, a warning is logged, and predicted-only geometry is drawn unaligned,
    exactly as when ``gt_c2w`` is absent (fully backward compatible).

    Args:
        result: dict returned by ``model.get_model_outputs(batch)``.  Must contain
            ``'gaussians'``, ``'c2w_input'`` ([B, v_input, 4, 4]), ``'c2w_target'``
            ([B, v_target, 4, 4]), ``'input_indices'`` ([B, v_input]), and
            ``'target_indices'`` ([B, v_target]); see ``_extract_pointcloud_xyz_rgb``
            for the point cloud fields it reads. Optionally ``'gt_c2w'`` ([B, V, 4, 4])
            for the ground-truth camera overlay described above.
        output_dir: root output directory; ``.glb`` files are written under
            ``output_dir / 'glb'``.
        batch_idx: used in the output filename
            ``scene_batch{batch_idx:04d}_sample{sample_idx:02d}.glb``.
        max_samples: cap on the number of batch items to save.  ``None`` saves all.
        screen_width: physical size of the drawn camera frustums, in the same units as
            the point cloud. ``None`` (default) auto-scales to 5% of the point cloud's
            bounding-box diagonal per sample, so frustums stay visibly proportioned
            regardless of the scene's absolute coordinate scale (Cheese3D point clouds
            span ~100s of units, unlike the ~1-2 unit scale a fixed absolute default
            was originally tuned for).
        session_ids: one session ID per batch item; when given, files are grouped into
            a per-session subfolder instead of a flat ``glb`` directory.
        sample_indices: batch item indices to save; when given, all other items are
            skipped. ``None`` saves every item (subject to ``max_samples``).

    Returns:
        list of Path objects for the .glb files that were written.
    """
    gaussians_list = result.get('gaussians')
    c2w_input = result.get('c2w_input')
    c2w_target = result.get('c2w_target')
    input_indices = result.get('input_indices')
    target_indices = result.get('target_indices')
    if (
        not gaussians_list
        or not torch.is_tensor(c2w_input)
        or not torch.is_tensor(c2w_target)
        or not torch.is_tensor(input_indices)
        or not torch.is_tensor(target_indices)
    ):
        return []

    gt_c2w = result.get('gt_c2w')
    has_gt_c2w = torch.is_tensor(gt_c2w)

    output_dir = Path(output_dir)
    glb_dir = output_dir / 'glb'

    cmap = matplotlib.colormaps['hsv']
    gt_color = np.array([0, 0, 0], dtype=np.uint8)

    saved = []
    for sample_idx, gs in enumerate(gaussians_list):
        if max_samples is not None and sample_idx >= max_samples:
            break
        if sample_indices is not None and sample_idx not in sample_indices:
            continue
        if sample_idx >= c2w_input.shape[0] or sample_idx >= c2w_target.shape[0]:
            continue

        xyz, rgb01, _ = _extract_pointcloud_xyz_rgb(result, sample_idx, gs)
        if xyz.size == 0:
            continue

        view_indices, c2ws = _dedupe_cameras_by_view_index(
            input_indices[sample_idx].detach().cpu(),
            c2w_input[sample_idx].detach().float().cpu().numpy(),
            target_indices[sample_idx].detach().cpu(),
            c2w_target[sample_idx].detach().float().cpu().numpy(),
        )
        num_cameras = len(view_indices)

        gt_c2w_np = None
        if has_gt_c2w and sample_idx < gt_c2w.shape[0]:
            gt_c2w_np = gt_c2w[sample_idx].detach().float().cpu().numpy()  # [V_gt, 4, 4]
            correspond = [
                (pos, vidx) for pos, vidx in enumerate(view_indices) if vidx < gt_c2w_np.shape[0]
            ]
            if correspond:
                pred_positions, gt_indices = zip(*correspond)
                transform = estimate_camera_similarity_transform(
                    c2ws[list(pred_positions)],
                    gt_c2w_np[list(gt_indices)],
                )
                yz_flip = np.array([1.0, -1.0, -1.0])
                xyz = (xyz * yz_flip) @ transform[:3, :3].T + transform[:3, 3] # undo flip, apply transform
                xyz = xyz * yz_flip # reapply flip
                c2ws = apply_similarity_transform_to_poses(transform, c2ws)
            else:
                log_step(
                    f'Skipping GT camera overlay for sample {sample_idx}: no view-index '
                    f'overlap between predicted views {view_indices} and gt_c2w rows '
                    f'(0..{gt_c2w_np.shape[0] - 1})',
                    level='warning',
                )
                gt_c2w_np = None

        sample_screen_width = screen_width
        if sample_screen_width is None:
            scene_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
            sample_screen_width = 0.05 * scene_diag if scene_diag > 0 else 0.1

        scene = trimesh.Scene()
        scene.add_geometry(trimesh.points.PointCloud(vertices=xyz, colors=rgb01))

        for view_idx, c2w in enumerate(c2ws):
            edge_color = (np.array(cmap(view_idx / num_cameras))[:3] * 255).astype(np.uint8)
            add_scene_cam(
                scene=scene,
                c2w=c2w,
                edge_color=edge_color,
                imsize=(256, 256),
                screen_width=sample_screen_width,
            )

        num_gt_cameras = 0
        if gt_c2w_np is not None:
            num_gt_cameras = gt_c2w_np.shape[0]
            for gt_view_idx in range(num_gt_cameras):
                add_scene_cam(
                    scene=scene,
                    c2w=gt_c2w_np[gt_view_idx],
                    edge_color=gt_color,
                    imsize=(256, 256),
                    screen_width=sample_screen_width,
                )

        sample_dir = glb_dir / session_ids[sample_idx] if session_ids is not None else glb_dir
        sample_dir.mkdir(parents=True, exist_ok=True)
        out_glb = sample_dir / f'scene_batch{batch_idx:04d}_sample{sample_idx:02d}.glb'
        scene.export(out_glb)

        log_step(
            f'Saved camera scene: {out_glb} ({num_cameras} predicted cameras, '
            f'{num_gt_cameras} GT cameras)',
            level='info',
        )
        saved.append(out_glb)

    return saved


# aliased so `infer_sable`'s `save_camera_pointcloud_scene` bool parameter can shadow
# the function name locally while still calling it
_save_camera_pointcloud_scene_fn = save_camera_pointcloud_scene


def _resolve_batch_size(training: dict, batch_size: int | None) -> int:
    """``batch_size`` if given, else ``training.batch_size_per_gpu`` (default ``1``)."""
    return int(batch_size) if batch_size is not None else int(training.get('batch_size_per_gpu', 1))


def _build_sable_dataset(config: dict, include_splits: list[str] | None = None) -> Any:
    """Construct the Sable inference dataset alone (no ``DataLoader``, no image loading).

    Args:
        config: full beast config dict (same as used for training).
        include_splits: IBL splits to load (e.g. ``['train', 'val']``). Defaults to
            ``'train'``, ``'val'``, and ``'test'``.

    Returns:
        the constructed dataset instance.
    """
    from beast.train_sable import _resolve_dataset_class

    if include_splits is None:
        include_splits = ['train', 'val', 'test']

    training = config['training']
    dataset_cls = _resolve_dataset_class(
        training.get('dataset_name', 'beast.data.sable_dataset.SABLEDataset')
    )
    dataset = dataset_cls(config, include_splits=include_splits)
    log_step(
        f'_build_sable_dataset: {len(dataset)} samples across splits {include_splits}',
        level='info',
    )
    return dataset


def _build_sable_inference_loader(
    config: dict,
    include_splits: list[str] | None = None,
    batch_size: int | None = None,
    start_row: int = 0,
    dataset: Any = None,
) -> tuple[Any, torch.utils.data.DataLoader]:
    """Build the ``(dataset, DataLoader)`` pair shared by Sable inference entrypoints.

    Args:
        config: full beast config dict (same as used for training).
        include_splits: IBL splits to load (e.g. ``['train', 'val']``).  Defaults to
            ``'train'``, ``'val'``, and ``'test'``.
        batch_size: overrides ``training.batch_size_per_gpu`` when given. Callers that key
            saved artifacts on ``batch_idx`` (e.g. ``extract_sable_latents``'s resume logic)
            must keep this the same across resumed runs against the same output directory,
            since a different batch size reshuffles which rows land in which ``batch_idx``.
        start_row: skip the dataset's first ``start_row`` rows (via a ``Subset``) so the
            ``DataLoader``/its workers never load or collate already-completed rows. Callers
            that key saved artifacts on ``batch_idx`` must offset the enumeration index by
            ``start_row // resolved_batch_size`` to recover the true ``batch_idx``.
        dataset: reuse an already-constructed dataset (e.g. one already inspected for resume
            bookkeeping) instead of building a new one.

    Returns:
        tuple ``(dataset, loader)`` — ``dataset`` is always the full (unsliced) dataset, even
        when ``loader`` iterates a ``start_row``-sliced ``Subset`` of it.
    """
    from beast.data.sable_dataset import collate_with_correspondence_padding

    if dataset is None:
        dataset = _build_sable_dataset(config, include_splits=include_splits)

    training = config['training']
    num_workers = int(training.get('num_workers', 4))
    resolved_batch_size = _resolve_batch_size(training, batch_size)

    loader_dataset = (
        torch.utils.data.Subset(dataset, range(start_row, len(dataset)))
        if start_row > 0
        else dataset
    )
    loader = torch.utils.data.DataLoader(
        loader_dataset,
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_correspondence_padding,
        drop_last=False,
    )
    return dataset, loader


def _select_samples_within_session_quota(
    session_ids: list[str],
    session_counts: dict[str, int],
    max_per_session: int,
) -> list[int]:
    """Pick batch-item indices whose session hasn't yet reached its output quota.

    Args:
        session_ids: one session ID per batch item.
        session_counts: running count of items already selected per session; updated
            in place for every index this call selects.
        max_per_session: max number of items allowed per session.

    Returns:
        list of batch-item indices whose session was still under quota.
    """
    selected = []
    for idx, session_id in enumerate(session_ids):
        if session_counts.get(session_id, 0) >= max_per_session:
            continue
        session_counts[session_id] = session_counts.get(session_id, 0) + 1
        selected.append(idx)
    return selected


def infer_sable(
    config: dict,
    model,
    output_dir: str | Path,
    save_pointclouds: bool = True,
    save_camera_pointcloud_scene: bool = False,
    save_visuals: bool = False,
    max_batches: int | None = None,
    include_splits: list[str] | None = None,
    max_files_per_session: int | None = None,
) -> dict:
    """Run Sable inference over an IBL dataset and optionally save PLY point clouds.

    Args:
        config: full beast config dict (same as used for training).
        model: trained Sable Lightning model instance.
        output_dir: root directory for outputs; PLY files go under ``output_dir/ply/``,
            optional camera-scene ``.glb`` files under ``output_dir/glb/``, and optional
            PNG visuals under ``output_dir/png/``.
        save_pointclouds: whether to save ``.ply`` files for each batch.
        save_camera_pointcloud_scene: whether to save ``.glb`` scenes (point cloud +
            camera frustums) for each batch.
        save_visuals: whether to save render-vs-target PNG grids for each batch.
        max_batches: stop after this many batches.  ``None`` runs the full dataset.
        include_splits: IBL splits to load (e.g. ``['train', 'val']``).  Defaults to
            ``'train'``, ``'val'``, and ``'test'``.
        max_files_per_session: cap on the number of PLY/GLB files saved per session.
            When set, PLY/GLB outputs are grouped into per-session subfolders
            (``output_dir/ply/{session_id}/`` and ``output_dir/glb/{session_id}/``) and
            batches whose items are all past quota skip the forward pass entirely.
            ``None`` (default) saves every item into the flat, unlimited layout.

    Returns:
        dict with keys:
            - ``'output_dir'``: str path of the output directory.
            - ``'num_batches'``: number of batches processed.
            - ``'ply_files'``: list of Path objects for all saved PLY files.
            - ``'camera_pointcloud_scene_glb_files'``: list of Path objects for all
              saved ``.glb`` scenes.
            - ``'vis_files'``: list of Path objects for all saved PNG grids.
    """
    from beast.models.model_utils.train_vis import save_training_visuals

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _dataset, loader = _build_sable_inference_loader(config, include_splits=include_splits)

    device = next(model.parameters()).device
    model.eval()

    all_ply: list[Path] = []
    all_glb: list[Path] = []
    all_vis: list[Path] = []
    num_batches = 0
    session_counts: dict[str, int] = {}

    target_sessions: set[str] | None = None
    if max_files_per_session is not None:
        configured_sessions = config.get('training', {}).get('session_names')
        if isinstance(configured_sessions, str):
            configured_sessions = [configured_sessions]
        if configured_sessions:
            target_sessions = set(configured_sessions)

    def _all_target_sessions_satisfied() -> bool:
        if target_sessions is None:
            return False
        satisfied = {
            sid for sid, count in session_counts.items() if count >= max_files_per_session
        }
        return target_sessions <= satisfied

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            session_ids = None
            sample_indices = None
            if max_files_per_session is not None:
                session_ids = [
                    session_id for session_id, _ in
                    (_parse_scene_name(scene_name) for scene_name in batch['scene_name'])
                ]
                sample_indices = _select_samples_within_session_quota(
                    session_ids, session_counts, max_files_per_session,
                )
                if not sample_indices:
                    if _all_target_sessions_satisfied():
                        break
                    continue

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            result = model.get_model_outputs(batch)

            if save_pointclouds:
                ply_paths = save_gaussian_pointclouds(
                    result, output_dir, batch_idx,
                    session_ids=session_ids, sample_indices=sample_indices,
                )
                all_ply.extend(ply_paths)

            if save_camera_pointcloud_scene:
                glb_paths = _save_camera_pointcloud_scene_fn(
                    result, output_dir, batch_idx,
                    session_ids=session_ids, sample_indices=sample_indices,
                )
                all_glb.extend(glb_paths)

            if save_visuals:
                vis_paths = save_training_visuals(
                    output_dir / 'png',
                    result=SimpleNamespace(**result),
                    batch=batch,
                    step=batch_idx,
                    session_ids=session_ids,
                    sample_indices=sample_indices,
                )
                all_vis.extend(vis_paths or [])

            num_batches += 1

            if max_files_per_session is not None and _all_target_sessions_satisfied():
                break

    log_step(
        f'infer_sable: processed {num_batches} batches, '
        f'{len(all_ply)} PLY files, {len(all_glb)} GLB scenes, {len(all_vis)} PNG grids',
        level='info',
    )

    return {
        'output_dir': str(output_dir),
        'num_batches': num_batches,
        'ply_files': all_ply,
        'camera_pointcloud_scene_glb_files': all_glb,
        'vis_files': all_vis,
    }


# maps a latent type name to the `data`/`config['model']` gate key read by
# `beast.models.model_utils.utils_latent.latent_tensor_export_if_requested`
_LATENT_GATE_KEYS: dict[str, str] = {
    'frame_z': 'return_frame_cls_tokens',
    'dino_z': 'return_dino_cls',
    'combined_z': 'return_combined_z',
    'img_tokens': 'return_img_tokens',
}

_SCENE_NAME_RE = re.compile(r'^(.+)_pair_(\d+)$')


def _parse_scene_name(scene_name: str) -> tuple[str, int]:
    """Split a Sable ``scene_name`` (``'{session_id}_pair_{pair_idx:06d}'``) into its parts.

    Args:
        scene_name: scene name as stored on each dataset record.

    Returns:
        tuple ``(session_id, pair_idx)``.

    Raises:
        ValueError: if ``scene_name`` doesn't match the expected ``'{session_id}_pair_{idx}'``
            pattern.
    """
    match = _SCENE_NAME_RE.match(scene_name)
    if not match:
        raise ValueError(f"scene_name {scene_name!r} does not match '<session_id>_pair_<idx>'")
    return match.group(1), int(match.group(2))


_BATCH_NPZ_REQUIRED_KEYS = (
    'z',
    'session_id',
    'pair_idx',
    'trial_split',
    'neural_trial_idx',
    'neural_bin_idx',
    'neural_interval_sec',
)


def _batch_output_path(
    output_dir: Path, latent_type: str, session_id: str, split_name: str, batch_idx: int,
) -> Path:
    """Path for a per-batch latent file:
    ``output_dir/{type}/{session}/{split}/{type}_batch{idx:04d}.npz``.
    """
    from beast.sable_encoding_decoding.img_token.trials_assembly import _session_subdir_key

    return (
        output_dir
        / latent_type
        / _session_subdir_key(session_id)
        / split_name
        / f'{latent_type}_batch{batch_idx:04d}.npz'
    )


def _num_batches_for(num_records: int, batch_size: int) -> int:
    """Number of ``batch_size``-sized (last one possibly smaller) batches over ``num_records``."""
    return (num_records + batch_size - 1) // batch_size if num_records else 0


def _batch_session_ids(records: list, batch_idx: int, batch_size: int) -> list[str]:
    """Distinct ``session_id``s covered by batch ``batch_idx``, in first-seen order.

    Reads directly off the dataset's (already-sorted, already-loaded) record list, so this
    never triggers ``__getitem__``/image loading — used to precompute resume state cheaply.
    """
    start = batch_idx * batch_size
    end = min(start + batch_size, len(records))
    seen: dict[str, None] = {}
    for i in range(start, end):
        seen.setdefault(records[i].session_id, None)
    return list(seen)


def _existing_batch_indices(directory: Path, file_prefix: str) -> set[int]:
    """Batch indices with an existing ``<file_prefix>_batch{idx:04d}.npz`` under ``directory``.

    A single directory listing rather than one open-and-parse per batch — see
    ``_resume_batch_start``.
    """
    if not directory.is_dir():
        return set()
    pattern = re.compile(rf'^{re.escape(file_prefix)}_batch(\d+)\.npz$')
    indices: set[int] = set()
    for p in directory.glob(f'{file_prefix}_batch*.npz'):
        match = pattern.match(p.name)
        if match:
            indices.add(int(match.group(1)))
    return indices


def _resume_batch_start(
    output_dir: Path,
    latent_types: list[str],
    session_ids_by_batch: list[list[str]],
    split_name: str,
) -> int:
    """Largest contiguous prefix of already-complete batches for one split.

    A ``(latent_type, session_id)`` is considered done for every batch either because a valid
    combined trials npz already exists for it (batches may have since been deleted — see
    ``extract_sable_latents``'s post-combine cleanup), or because its per-batch npz file exists
    on disk (checked via one directory listing per ``(latent_type, session_id)``, not one
    ``np.load`` per batch). Saves are atomic and processed strictly in increasing ``batch_idx``
    order within a split, so a complete batch implies every earlier batch is complete too —
    this never needs to look past the first incomplete batch. The boundary batch (the last one
    counted complete) is re-validated with ``_is_valid_batch_npz`` as a corruption guard.

    Args:
        output_dir: root directory for per-latent-type subdirectories.
        latent_types: latent types requested this run.
        session_ids_by_batch: ``session_ids_by_batch[b]`` is the list of distinct session ids
            covered by batch ``b`` (see ``_batch_session_ids``).
        split_name: IBL split these batches belong to.

    Returns:
        the first not-yet-complete ``batch_idx`` (i.e. how many leading batches to skip).
    """
    from beast.sable_encoding_decoding.img_token.trials_assembly import (
        _is_valid_trials_npz,
        _session_subdir_key,
    )

    # cache[(latent_type, session_id)] is `None` (session already fully combined — always
    # done) or a concrete set of present batch indices.
    cache: dict[tuple[str, str], set[int] | None] = {}

    def _present(latent_type: str, session_id: str) -> set[int] | None:
        key = (latent_type, session_id)
        if key not in cache:
            session_dir = output_dir / latent_type / _session_subdir_key(session_id)
            if _is_valid_trials_npz(session_dir / f'{latent_type}_trials.npz'):
                cache[key] = None
            else:
                cache[key] = _existing_batch_indices(session_dir / split_name, latent_type)
        return cache[key]

    def _done(latent_type: str, session_id: str, batch_idx: int) -> bool:
        present = _present(latent_type, session_id)
        return present is None or batch_idx in present

    k = 0
    for batch_idx, sessions in enumerate(session_ids_by_batch):
        if all(_done(lt, sid, batch_idx) for lt in latent_types for sid in sessions):
            k = batch_idx + 1
        else:
            break

    if k > 0:
        for lt in latent_types:
            for sid in session_ids_by_batch[k - 1]:
                if _present(lt, sid) is None:
                    continue  # already validated via its combined trials npz
                path = _batch_output_path(output_dir, lt, sid, split_name, k - 1)
                if not _is_valid_batch_npz(path):
                    k -= 1
                    break
            else:
                continue
            break

    return k


def _delete_session_batch_files(
    output_dir: Path, latent_type: str, session_id: str, splits: list[str],
) -> None:
    """Remove a session's per-batch npz files (and now-empty split dirs) after a good combine."""
    from beast.sable_encoding_decoding.img_token.trials_assembly import _session_subdir_key

    session_dir = output_dir / latent_type / _session_subdir_key(session_id)
    for split_name in splits:
        split_dir = session_dir / split_name
        if not split_dir.is_dir():
            continue
        for p in split_dir.glob(f'{latent_type}_batch*.npz'):
            p.unlink()
        try:
            split_dir.rmdir()
        except OSError:
            pass  # left non-empty (unexpected stray files) — not our call to remove it


def _is_valid_batch_npz(path: Path) -> bool:
    """True if ``path`` exists and loads as a batch npz with the expected keys and shape."""
    if not path.is_file():
        return False
    try:
        data = np.load(path, allow_pickle=True)
        if not all(key in data.files for key in _BATCH_NPZ_REQUIRED_KEYS):
            return False
        if data['z'].ndim != 3:
            return False
    except (OSError, ValueError, EOFError, KeyError, pickle.UnpicklingError):
        return False
    return True


def _save_latent_batch_npz(
    path: Path,
    *,
    z: np.ndarray,
    session_ids: list[str],
    pair_idxs: list[int],
    splits: list[str],
    neural_trial_idx: list[int],
    neural_bin_idx: list[int],
    neural_interval_sec: np.ndarray,
    aux: dict[str, np.ndarray] | None = None,
) -> None:
    """Atomically save one batch's latents plus row metadata as a single ``.npz``.

    Guards against a killed job leaving a file that looks complete but isn't, so a later
    resume never mistakes a partial write for a finished one. The row metadata (session,
    pair, split, neural trial/bin alignment) is exactly the schema
    ``beast.sable_encoding_decoding.img_token.trials_assembly.assemble_from_inference_batch_npz``
    reads. ``aux`` optionally carries extra per-row arrays (e.g. the ``img_tokens`` camera
    tensors keyed by ``trials_assembly.IMG_TOKEN_CAM_BATCH_KEYS``), saved as additional
    top-level ``.npz`` keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    aux_arrays = {k: np.asarray(v, dtype=np.float32) for k, v in (aux or {}).items()}
    # write via an open file handle so numpy doesn't append its own `.npz` suffix to tmp_path
    with open(tmp_path, 'wb') as f:
        np.savez(
            f,
            z=np.asarray(z, dtype=np.float32),
            session_id=np.asarray(session_ids, dtype=object),
            pair_idx=np.asarray(pair_idxs, dtype=np.int64),
            trial_split=np.asarray(splits, dtype=object),
            neural_trial_idx=np.asarray(neural_trial_idx, dtype=np.int64),
            neural_bin_idx=np.asarray(neural_bin_idx, dtype=np.int64),
            neural_interval_sec=np.asarray(neural_interval_sec, dtype=np.float64),
            **aux_arrays,
        )
    tmp_path.replace(path)


def _write_combined_trials_npz(
    output_path: Path,
    z_trials_time: np.ndarray,
    trial_split_labels: list[str] | None,
    trial_session_ids: list[str] | None,
    neural_trial_idx: np.ndarray | None,
    per_trial_iv: np.ndarray | None,
    meta: dict[str, Any],
    include_splits: str,
) -> Path:
    """Write one combined trials ``.npz`` spanning every session/split assembled so far.

    Unlike ``trials_assembly``'s own writers (which either drop session identity entirely
    or split output into one file per session subdirectory), this keeps everything as a
    single tensor per this pipeline's "one big combined tensor" contract, with ``session_id``
    stored as a per-trial metadata array (parallel to ``trial_split``) so the trial-to-session
    mapping is never lost.

    Args:
        output_path: destination ``.npz`` path.
        z_trials_time: ``[N_trials, T_bins, V, D]`` combined latent tensor.
        trial_split_labels: per-trial split label, or ``None`` if unavailable.
        trial_session_ids: per-trial session id, or ``None`` if unavailable.
        neural_trial_idx: per-trial neural trial index, or ``None`` if unavailable.
        per_trial_iv: per-trial ``[t_start, t_end]`` interval, or ``None`` if unavailable.
        meta: assembly metadata dict, stored as ``meta_json``.
        include_splits: comma-separated splits, used to key the per-split ``z_trials_time``.

    Returns:
        ``output_path``.
    """
    from beast.sable_encoding_decoding.img_token.trials_assembly import (
        _per_split_z_trials_kw,
        _stack_split_intervals_from_rows,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict[str, Any] = {'meta_json': json.dumps(meta)}
    if trial_split_labels is not None:
        save_kw.update(_per_split_z_trials_kw(z_trials_time, trial_split_labels, include_splits))
        save_kw['trial_split'] = np.array(trial_split_labels, dtype=object)
    else:
        save_kw['z_trials_time'] = z_trials_time
    if trial_session_ids is not None:
        save_kw['session_id'] = np.array(trial_session_ids, dtype=object)
    if neural_trial_idx is not None:
        save_kw['neural_trial_idx'] = neural_trial_idx
    if trial_split_labels is not None and per_trial_iv is not None:
        train_iv, val_iv, test_iv = _stack_split_intervals_from_rows(
            trial_split_labels, per_trial_iv,
        )
        save_kw['train_intervals'] = train_iv
        save_kw['val_intervals'] = val_iv
        save_kw['test_intervals'] = test_iv
    np.savez_compressed(output_path, **save_kw)
    return output_path


def extract_sable_latents(
    config: dict,
    model,
    output_dir: str | Path,
    latent_types: list[str] | None = None,
    max_batches: int | None = None,
    include_splits: list[str] | None = None,
    resume: bool = True,
    batch_size: int | None = None,
    time_bins: int | None = None,
) -> dict:
    """Extract and save per-batch, per-session Sable latent tensors for encoding/decoding.

    Sets the ``return_*`` gates already wired into ``beast.models.sable.Sable.forward`` (via
    ``latent_tensor_export_if_requested``), processes one on-disk split at a time (so every
    saved batch's rows share a single split), and saves one ``.npz`` per batch per session per
    requested latent type under
    ``output_dir/<latent_type>/<session_id>/<split>/<latent_type>_batch{idx:04d}.npz`` — a
    batch whose rows span a session boundary is split into one file per session. This is the
    exact layout ``beast.sable_encoding_decoding.img_token.trials_assembly``'s
    split-subdirectory batch-npz assembly path expects, and matches the multisession
    per-session-id convention used elsewhere (e.g.
    ``beast.sable_encoding_decoding.img_token.run_pca_and_save``) and documented in
    ``docs/sable/neural_encoding_decoding.md``.

    After every split has been processed, assembles each session's batches into one combined
    trials tensor at ``output_dir/<latent_type>/<session_id>/<latent_type>_trials.npz`` — except
    for ``img_tokens``, whose combine step is always skipped (see below). Once a session's
    combine succeeds and validates, its per-batch npz files are deleted (again, except for
    ``img_tokens``).

    ``img_tokens`` is never combined or cleaned up here: its per-batch shards are exactly what
    ``beast.sable_encoding_decoding.img_token.run_pca_and_save`` (Stage 2 of the neural
    encoding/decoding pipeline) reads directly off disk. Combining and deleting them would
    silently break that stage.

    Args:
        config: full beast config dict (same as used for training).
        model: trained Sable Lightning model instance.
        output_dir: root directory for per-latent-type subdirectories.
        latent_types: subset of ``['frame_z', 'dino_z', 'combined_z', 'img_tokens']``; ``None``
            exports all four (the ``--return-all-z`` analog).
        max_batches: stop after this many batches per split.  ``None`` runs the full dataset.
        include_splits: IBL splits to load and process, in order (e.g. ``['train', 'val']``).
            Defaults to ``'train'``, ``'val'``, and ``'test'``.
        resume: when ``True`` (default), skip a batch's forward pass (and the dataset row
            loading behind it) entirely once it's already saved or its session's combined
            trials file already exists, and skip re-combining a session whose combined trials
            file already exists and validates.
        batch_size: overrides ``training.batch_size_per_gpu``. Must stay the same across
            resumed runs against the same ``output_dir`` — see
            ``_build_sable_inference_loader``.
        time_bins: neural bins per trial, used to decide whether a trial is "complete" during
            the combine step. ``None`` derives it from the dataset's on-disk neural-alignment
            metadata (``SABLEDataset.max_neural_bin_idx``).

    Returns:
        dict with keys:
            - ``'output_dir'``: str path of the output directory.
            - ``'num_batches'``: number of batches for which a forward pass ran.
            - ``'num_batches_skipped'``: number of batches skipped via resume.
            - ``'saved_files'``: dict mapping latent type to list of saved Path objects
              (batch files already deleted by a prior successful combine are not listed).
            - ``'combined_trials_files'``: list of combined trials ``.npz`` Path objects, one
              per ``(latent_type, session_id)`` (excluding ``img_tokens``).

    Raises:
        ValueError: if ``latent_types`` contains an unknown latent type.
    """
    if latent_types is None:
        latent_types = list(_LATENT_GATE_KEYS)
    unknown = sorted(set(latent_types) - set(_LATENT_GATE_KEYS))
    if unknown:
        raise ValueError(
            f'unknown latent_types {unknown}; expected subset of {sorted(_LATENT_GATE_KEYS)}',
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_to_process = include_splits or ['train', 'val', 'test']
    resolved_batch_size = _resolve_batch_size(config['training'], batch_size)

    device = next(model.parameters()).device
    model.eval()

    num_batches = 0
    num_batches_skipped = 0
    saved_files: dict[str, list[Path]] = {latent_type: [] for latent_type in latent_types}
    max_bin_idx_seen: int | None = None

    with torch.no_grad():
        for split_name in splits_to_process:
            dataset = _build_sable_dataset(config, include_splits=[split_name])
            if len(dataset) == 0:
                log_step(
                    f'extract_sable_latents: split {split_name!r} has no records, skipping',
                    level='info',
                )
                continue
            split_max_bin = dataset.max_neural_bin_idx()
            if split_max_bin is not None:
                max_bin_idx_seen = max(max_bin_idx_seen or 0, split_max_bin)

            num_total_batches = _num_batches_for(len(dataset), resolved_batch_size)
            session_ids_by_batch = [
                _batch_session_ids(dataset._records, b, resolved_batch_size)
                for b in range(num_total_batches)
            ]
            resume_start = (
                _resume_batch_start(output_dir, latent_types, session_ids_by_batch, split_name)
                if resume
                else 0
            )
            effective_start = (
                min(resume_start, max_batches) if max_batches is not None else resume_start
            )
            log_step(
                f'extract_sable_latents: split {split_name!r} has {num_total_batches} batches '
                f'total (resuming from batch {effective_start})',
                level='info',
            )

            for b in range(effective_start):
                num_batches_skipped += 1
                for latent_type in latent_types:
                    for session_id in session_ids_by_batch[b]:
                        path = _batch_output_path(
                            output_dir, latent_type, session_id, split_name, b,
                        )
                        if path.is_file():
                            saved_files[latent_type].append(path)

            _, loader = _build_sable_inference_loader(
                config,
                include_splits=[split_name],
                batch_size=resolved_batch_size,
                start_row=effective_start * resolved_batch_size,
                dataset=dataset,
            )

            for local_idx, batch in enumerate(loader):
                batch_idx = effective_start + local_idx
                if max_batches is not None and batch_idx >= max_batches:
                    break

                scene_names = batch['scene_name']
                row_ids = [_parse_scene_name(scene_name) for scene_name in scene_names]
                session_ids = [session_id for session_id, _ in row_ids]
                pair_idxs = [pair_idx for _, pair_idx in row_ids]
                splits = list(batch['split'])
                neural_trial_idx = batch['neural_trial_idx'].cpu().numpy().tolist()
                neural_bin_idx = batch['neural_bin_idx'].cpu().numpy().tolist()
                neural_interval_sec = batch['neural_interval_sec'].cpu().numpy()

                batch = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                for latent_type in latent_types:
                    batch[_LATENT_GATE_KEYS[latent_type]] = True

                result = model.get_model_outputs(batch)

                # group row indices by session so a session-boundary batch is split into one
                # saved file per session, never mixing sessions within a file
                session_row_groups: dict[str, list[int]] = {}
                for row_idx, session_id in enumerate(session_ids):
                    session_row_groups.setdefault(session_id, []).append(row_idx)

                for latent_type in latent_types:
                    z_np = result[latent_type].detach().cpu().numpy()
                    aux = None
                    if latent_type == 'img_tokens':
                        from beast.sable_encoding_decoding.img_token.trials_assembly import (
                            IMG_TOKEN_CAM_BATCH_KEYS,
                        )

                        cam_key_to_result_attr = {
                            'c2w_target_out': 'c2w_target',
                            'fxfycxcy_target_out': 'fxfycxcy_target',
                            'c2w_input_out': 'c2w_input',
                            'fxfycxcy_input_out': 'fxfycxcy_input',
                        }
                        aux = {}
                        for cam_key in IMG_TOKEN_CAM_BATCH_KEYS:
                            cam_tensor = result[cam_key_to_result_attr[cam_key]]
                            aux[cam_key] = cam_tensor.detach().float().cpu().numpy()
                    for session_id, rows in session_row_groups.items():
                        path = _batch_output_path(
                            output_dir, latent_type, session_id, split_name, batch_idx,
                        )
                        _save_latent_batch_npz(
                            path,
                            z=z_np[rows],
                            session_ids=[session_ids[i] for i in rows],
                            pair_idxs=[pair_idxs[i] for i in rows],
                            splits=[splits[i] for i in rows],
                            neural_trial_idx=[neural_trial_idx[i] for i in rows],
                            neural_bin_idx=[neural_bin_idx[i] for i in rows],
                            neural_interval_sec=neural_interval_sec[rows],
                            aux={k: v[rows] for k, v in aux.items()} if aux else None,
                        )
                        saved_files[latent_type].append(path)

                        combine_note = (
                            ' (combined_z = cat([frame_z, dino_z]))'
                            if latent_type == 'combined_z' else ''
                        )
                        log_step(
                            f'extract_sable_latents: [{split_name}] batch '
                            f'{batch_idx + 1}/{num_total_batches} saved '
                            f'{latent_type}{combine_note} -> {path}',
                            level='info',
                        )

                num_batches += 1

    log_step(
        f'extract_sable_latents: processed {num_batches} batches '
        f'({num_batches_skipped} skipped via resume) across splits {splits_to_process}, '
        f'latent types {latent_types}',
        level='info',
    )

    resolved_time_bins = time_bins or max_bin_idx_seen
    combined_trials_files: list[Path] = []
    if resolved_time_bins is None:
        log_step(
            'extract_sable_latents: no neural-alignment metadata found in this run, '
            'skipping trial combine step',
            level='info',
        )
    else:
        sessions_this_run = {
            path.parent.parent.name for paths in saved_files.values() for path in paths
        }
        combined_trials_files = _combine_and_cleanup_sessions(
            output_dir=output_dir,
            latent_types=latent_types,
            splits_to_process=splits_to_process,
            resolved_time_bins=resolved_time_bins,
            resume=resume,
            sessions_this_run=sessions_this_run,
        )

    return {
        'output_dir': str(output_dir),
        'num_batches': num_batches,
        'num_batches_skipped': num_batches_skipped,
        'saved_files': saved_files,
        'combined_trials_files': combined_trials_files,
    }


def _combine_and_cleanup_sessions(
    output_dir: Path,
    latent_types: list[str],
    splits_to_process: list[str],
    resolved_time_bins: int,
    resume: bool,
    sessions_this_run: set[str],
) -> list[Path]:
    """Combine this run's sessions' batches into a trials npz, then delete the batches.

    ``img_tokens`` is always skipped (see ``extract_sable_latents``'s docstring): its raw
    per-batch shards are what ``img_token.run_pca_and_save`` consumes directly.

    Args:
        output_dir: root directory for per-latent-type subdirectories.
        latent_types: latent types requested this run.
        splits_to_process: splits that were processed this run (only these splits' batch dirs
            are cleaned up).
        resolved_time_bins: neural bins per trial, passed through to the assembler.
        resume: when ``True``, a session already having a valid combined trials file is left
            untouched (not re-assembled, not re-deleted).
        sessions_this_run: session subdirectory names this run actually wrote (or found
            already-written) batches for. Other sessions under ``output_dir`` are left
            untouched even if present on disk — they belong to other, possibly concurrent,
            runs sharing this ``output_dir`` (e.g. one SLURM job per session), and combining or
            deleting their files here would race with that other run.

    Returns:
        list of combined trials ``.npz`` paths, one per discovered ``(latent_type, session)``.
    """
    from beast.sable_encoding_decoding.img_token.trials_assembly import (
        _is_valid_trials_npz,
        assemble_z_trials_time_from_inference_batches,
    )

    include_splits_str = ','.join(splits_to_process)
    combined_trials_files: list[Path] = []

    for latent_type in latent_types:
        if latent_type == 'img_tokens':
            log_step(
                'extract_sable_latents: skipping combine for img_tokens — '
                'img_token.run_pca_and_save reads its raw per-batch shards directly',
                level='info',
            )
            continue

        latent_dir = output_dir / latent_type
        if not latent_dir.is_dir():
            continue

        for session_dir in sorted(
            p for p in latent_dir.iterdir() if p.is_dir() and p.name in sessions_this_run
        ):
            trials_path = session_dir / f'{latent_type}_trials.npz'
            if resume and _is_valid_trials_npz(trials_path):
                combined_trials_files.append(trials_path)
                continue

            combine_note = (
                ' (combined_z = cat([frame_z, dino_z]))' if latent_type == 'combined_z' else ''
            )
            log_step(
                f'extract_sable_latents: combining {latent_type!r}{combine_note} for session '
                f'{session_dir.name!r} into trials tensor',
                level='info',
            )
            assembly = assemble_z_trials_time_from_inference_batches(
                input_dir=session_dir,
                include_splits=include_splits_str,
                time_bins=resolved_time_bins,
                file_prefix=latent_type,
                split_subdirs=True,
            )
            written_path = _write_combined_trials_npz(
                output_path=trials_path,
                z_trials_time=assembly.z_trials_time,
                trial_split_labels=assembly.trial_split_labels,
                trial_session_ids=assembly.trial_session_ids,
                neural_trial_idx=assembly.neural_trial_idx,
                per_trial_iv=assembly.per_trial_iv,
                meta=assembly.meta,
                include_splits=include_splits_str,
            )
            log_step(
                f'extract_sable_latents: saved combined {latent_type!r} trials '
                f'({assembly.z_trials_time.shape}) -> {written_path}',
                level='info',
            )
            combined_trials_files.append(written_path)

            if _is_valid_trials_npz(written_path):
                _delete_session_batch_files(
                    output_dir, latent_type, session_dir.name, splits_to_process,
                )

    return combined_trials_files
