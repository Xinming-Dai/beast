"""Inference handlers for saving model predictions on images and videos."""

import logging
from pathlib import Path
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
            ``output_dir / 'ply'``.
        batch_idx: used in the output filename
            ``pointcloud_batch{batch_idx:04d}_sample{sample_idx:02d}.ply``.
        max_samples: cap on the number of batch items to save.  ``None`` saves all.

    Returns:
        list of Path objects for the PLY files that were written.
    """
    gaussians_list = result.get('gaussians')
    if not gaussians_list:
        return []

    output_dir = Path(output_dir)
    ply_dir = output_dir / 'ply'
    ply_dir.mkdir(parents=True, exist_ok=True)

    try:
        import open3d as o3d
        has_o3d = True
    except ImportError:
        has_o3d = False

    saved = []
    for sample_idx, gs in enumerate(gaussians_list):
        if max_samples is not None and sample_idx >= max_samples:
            break

        xyz, rgb01, used_pixel_colors = _extract_pointcloud_xyz_rgb(result, sample_idx, gs)
        if xyz.size == 0:
            continue

        out_ply = ply_dir / f'pointcloud_batch{batch_idx:04d}_sample{sample_idx:02d}.ply'

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
    glb_dir.mkdir(parents=True, exist_ok=True)

    cmap = matplotlib.colormaps['hsv']
    gt_color = np.array([0, 0, 0], dtype=np.uint8)

    saved = []
    for sample_idx, gs in enumerate(gaussians_list):
        if max_samples is not None and sample_idx >= max_samples:
            break
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

        out_glb = glb_dir / f'scene_batch{batch_idx:04d}_sample{sample_idx:02d}.glb'
        scene.export(out_glb)

        log_step(
            f'Saved camera scene: {out_glb} ({num_cameras} predicted cameras, '
            f'{num_gt_cameras} GT cameras)',
            level='info',
        )
        saved.append(out_glb)

    return saved


def save_front_view_nvs(
    result: dict,
    output_dir: str | Path,
    batch_idx: int,
    max_samples: int | None = None,
) -> list[Path]:
    """Save held-out front-camera novel-view-synthesis renders as side-by-side PNGs.

    For each batch item, writes a two-panel image with the ground-truth front-camera
    frame on the left and the model's rendered front view on the right. The render is
    produced by ``Sable.render_front_view_nvs`` (predicted gaussians viewed from the GT
    front pose aligned into the predicted frame); the front image is never fed to the
    model, so this is a true novel-view-synthesis comparison. No-ops (returns ``[]``)
    when ``result`` lacks ``front_render`` (front NVS disabled).

    Args:
        result: dict form of the Sable forward output. Reads ``front_render``
            ([B, 1, 3, H, W]) and, when present, ``front_gt_image`` ([B, 3, H, W]).
        output_dir: root output directory; PNGs are written under
            ``output_dir / 'front_nvs'``.
        batch_idx: used in the output filename
            ``front_nvs_batch{batch_idx:04d}_sample{sample_idx:02d}.png``.
        max_samples: cap on the number of batch items to save. ``None`` saves all.

    Returns:
        list of Path objects for the PNG files that were written.
    """
    front_render = result.get('front_render')
    if not torch.is_tensor(front_render):
        return []
    front_gt_image = result.get('front_gt_image')

    out_dir = Path(output_dir) / 'front_nvs'
    out_dir.mkdir(parents=True, exist_ok=True)

    def _to_uint8(image: torch.Tensor) -> np.ndarray:
        """Convert a ``[3, H, W]`` float tensor in ``[0, 1]`` to ``[H, W, 3]`` uint8."""
        arr = image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return (arr * 255.0 + 0.5).astype(np.uint8)

    num_samples = int(front_render.shape[0])
    if max_samples is not None:
        num_samples = min(num_samples, int(max_samples))

    saved = []
    for sample_idx in range(num_samples):
        render_img = _to_uint8(front_render[sample_idx, 0])  # [H, W, 3]
        panels = [render_img]
        labels = ['front NVS render']
        if torch.is_tensor(front_gt_image) and sample_idx < front_gt_image.shape[0]:
            gt_img = _to_uint8(front_gt_image[sample_idx])   # [H, W, 3]
            panels = [gt_img, render_img]
            labels = ['front GT', 'front NVS render']

        combined = np.concatenate(panels, axis=1)  # [H, W*len(panels), 3]
        out_png = out_dir / f'front_nvs_batch{batch_idx:04d}_sample{sample_idx:02d}.png'
        Image.fromarray(combined).save(out_png)
        log_step(
            f'Saved front-view NVS ({" | ".join(labels)}): {out_png}',
            level='info',
        )
        saved.append(out_png)

    return saved


# aliased so `infer_sable`'s `save_camera_pointcloud_scene` bool parameter can shadow
# the function name locally while still calling it
_save_camera_pointcloud_scene_fn = save_camera_pointcloud_scene


def infer_sable(
    config: dict,
    model,
    output_dir: str | Path,
    save_pointclouds: bool = True,
    save_camera_pointcloud_scene: bool = False,
    save_visuals: bool = False,
    max_batches: int | None = None,
    include_splits: list[str] | None = None,
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
            both ``'train'`` and ``'val'``.

    Returns:
        dict with keys:
            - ``'output_dir'``: str path of the output directory.
            - ``'num_batches'``: number of batches processed.
            - ``'ply_files'``: list of Path objects for all saved PLY files.
            - ``'camera_pointcloud_scene_glb_files'``: list of Path objects for all
              saved ``.glb`` scenes.
            - ``'vis_files'``: list of Path objects for all saved PNG grids.
    """
    from beast.data.sable_dataset import collate_with_correspondence_padding
    from beast.models.model_utils.train_vis import save_training_visuals
    from beast.train_sable import _resolve_dataset_class

    if include_splits is None:
        include_splits = ['train', 'val']

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training = config['training']
    dataset_cls = _resolve_dataset_class(
        training.get('dataset_name', 'beast.data.sable_dataset.SABLEDataset')
    )
    dataset = dataset_cls(config, include_splits=include_splits)
    log_step(f'infer_sable: {len(dataset)} samples across splits {include_splits}', level='info')

    num_workers = int(training.get('num_workers', 4))
    batch_size = int(training.get('batch_size_per_gpu', 1))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_correspondence_padding,
        drop_last=False,
    )

    device = next(model.parameters()).device
    model.eval()

    all_ply: list[Path] = []
    all_glb: list[Path] = []
    all_vis: list[Path] = []
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            result = model.get_model_outputs(batch)

            if save_pointclouds:
                ply_paths = save_gaussian_pointclouds(result, output_dir, batch_idx)
                all_ply.extend(ply_paths)

            if save_camera_pointcloud_scene:
                glb_paths = _save_camera_pointcloud_scene_fn(result, output_dir, batch_idx)
                all_glb.extend(glb_paths)

            if save_visuals:
                vis_paths = save_training_visuals(
                    output_dir / 'png',
                    result=result,
                    batch=batch,
                    step=batch_idx,
                )
                all_vis.extend(vis_paths or [])

            num_batches += 1

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
