"""Dataset objects store images and augmentation pipeline."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import imgaug.augmenters.size as _iaa_size
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from beast.data.types import ExampleDict
from beast.logging import log_step

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _patched_prevent(axis_size: int, crop_start: int, crop_end: int) -> tuple[int, ...]:
    """Monkey patch to fix imaug 0.4.2 compatability issue with numpy 2.x"""
    result = _iaa_size._prevent_zero_sizes_after_crops_(
        np.array([axis_size], dtype=np.int32),
        np.array([crop_start], dtype=np.int32),
        np.array([crop_end], dtype=np.int32),
    )
    return tuple(int(np.asarray(v).flat[0]) for v in result)


#  monkey patch to fix imaug 0.4.2 compatability issue with numpy 2.x
_iaa_size._prevent_zero_size_after_crop_ = _patched_prevent


class BaseDataset(torch.utils.data.Dataset):
    """Base dataset that contains images."""

    def __init__(
        self,
        data_dir: str | Path,
        imgaug_pipeline: Callable | None,
        num_channels: int = 3,
        session_names: list[str] | str | None = None,
        cameras: list[str] | str | None = None,
        segmentation_root: str | Path | None = None,
        mask_session_id: str | None = None,
        mask_camera_role: str | None = None,
    ) -> None:
        """Initialize a dataset for autoencoder models.

        Parameters
        ----------
        data_dir: absolute path to data directory
        imgaug_pipeline: imgaug transform pipeline to apply to images
        num_channels: number of output channels; 1 loads as grayscale then converts to RGB,
            3 loads directly as RGB
        session_names: if provided, restrict images to paths whose directory parts contain
            one of these session IDs as a substring; accepts a single string or a list of
            strings; all images under data_dir are used when null
        cameras: if provided, restrict images to paths whose directory parts contain one of
            these camera names as a substring; accepts a single string or a list of strings;
            applied together with session_names (an image must match both filters when both
            are set); all camera views are used when null
        segmentation_root: if provided, root directory containing a single-channel mask PNG
            for every image; each item's mask is loaded and returned under the 'mask' key.
            Resolved one of two ways: by default, mirroring the image's path relative to
            data_dir (segmentation_root / image_path.relative_to(data_dir)); when
            mask_session_id and mask_camera_role are also given, via each image's eval-layout
            frame_index_mapping.json instead (see mask_camera_role)
        mask_session_id: session id segment of the eval-layout mask path
            (segmentation_root/segmentation_masks/{mask_session_id}/{mask_camera_role}/
            mask{source_frame_index:08d}.png); required together with mask_camera_role
        mask_camera_role: 'left' or 'right' — selects the
            f'{mask_camera_role}_source_frame_index' field from each image's parent
            directory's frame_index_mapping.json (an eval-layout sidecar mapping
            interval{N}timebin{M}.png filenames to source frame indices, written by
            beast.preprocess.cheese3d.extract_cheese3d_eval_frames); required together with
            mask_session_id. Matches the convention Sable's IBLTwoViewDataset uses to align
            eval-layout frames with beast.preprocess.sable.precompute_sam3_masks_eval output

        """
        if isinstance(session_names, str):
            session_names = [session_names]
        if isinstance(cameras, str):
            cameras = [cameras]
        if num_channels not in (1, 3):
            raise ValueError(f'num_channels must be 1 or 3, got {num_channels}')
        if (mask_session_id is None) != (mask_camera_role is None):
            raise ValueError('mask_session_id and mask_camera_role must be set together')
        if mask_camera_role is not None and mask_camera_role not in ('left', 'right'):
            raise ValueError(f"mask_camera_role must be 'left' or 'right', got {mask_camera_role}")
        self.num_channels = num_channels
        log_step(f"BaseDataset.__init__ called with data_dir: {data_dir}", level='debug')
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f'{self.data_dir} is not a directory')
        log_step(f"Data directory exists: {self.data_dir}", level='debug')

        self.imgaug_pipeline = imgaug_pipeline
        self.segmentation_root = Path(segmentation_root) if segmentation_root else None
        self.mask_session_id = mask_session_id
        self.mask_camera_role = mask_camera_role
        self._frame_index_mapping_cache: dict[Path, dict] = {}
        # collect ALL png files in data_dir
        scan_start = time.time()
        try:
            log_step(
                f"Starting to scan for PNG files in {self.data_dir}"
                ' (this may take a while for large directories)...',
                level='debug',
            )
            self.image_list = sorted(list(self.data_dir.rglob('*.png')))
            scan_duration = time.time() - scan_start
            log_step(
                f"Finished scanning. Found {len(self.image_list)} PNG files"
                f' in {scan_duration:.2f} seconds',
                level='debug',
            )
        except Exception as e:
            log_step(f"ERROR during file scanning: {e}", level='error')
            raise
        if session_names:
            self.image_list = [
                img_path for img_path in self.image_list
                if any(
                    session_name in part
                    for part in img_path.parts
                    for session_name in session_names
                )
            ]
            log_step(
                f"Filtered to {len(self.image_list)} PNG files matching sessions"
                f' {session_names}',
                level='debug',
            )
        if cameras:
            self.image_list = [
                img_path for img_path in self.image_list
                if any(
                    camera in part
                    for part in img_path.parts
                    for camera in cameras
                )
            ]
            log_step(
                f"Filtered to {len(self.image_list)} PNG files matching cameras"
                f' {cameras}',
                level='debug',
            )
        if len(self.image_list) == 0:
            raise ValueError(f'{self.data_dir} does not contain image data in png format')
        log_step(
            f"BaseDataset initialization complete with {len(self.image_list)} images",
            level='debug',
        )

        # send image to tensor, resize to canonical dimensions, and normalize
        self.image_size = 224
        pytorch_transform_list = [
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
        self.pytorch_transform = transforms.Compose(pytorch_transform_list)

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        return len(self.image_list)

    def __getitem__(self, idx: int | list) -> ExampleDict | list[ExampleDict]:
        """Get item(s) from dataset.

        Parameters
        ----------
        idx: single index or list  indices

        Returns
        -------
        Single ExampleDict or list of ExampleDict objects

        """
        # Handle batch of indices
        if isinstance(idx, list):
            return [self._get_single_item(i) for i in idx]
        else:
            # Handle single index
            return self._get_single_item(idx)

    def _load_mask(self, mask_path: Path) -> torch.Tensor:
        """Load a binary segmentation mask and resize to the dataset's canonical image size.

        Parameters
        ----------
        mask_path: path to a single-channel mask PNG with values in {0, 255}

        Returns
        -------
        float32 tensor of shape (1, image_size, image_size) with values in {0, 1}

        """
        if not mask_path.is_file():
            raise FileNotFoundError(f'segmentation mask not found: {mask_path}')
        arr = np.asarray(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = torch.from_numpy(arr > 0).to(torch.float32).unsqueeze(0).unsqueeze(0)
        if mask.shape[-2] != self.image_size or mask.shape[-1] != self.image_size:
            mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode='nearest')
        return mask.squeeze(0)  # shape (1, image_size, image_size)

    def _resolve_mask_path(self, img_path: Path) -> Path:
        """Return the mask path for one image, per the configured resolution mode.

        Parameters
        ----------
        img_path: path to the source image

        Returns
        -------
        path to the corresponding mask PNG (not guaranteed to exist)

        """
        if self.mask_camera_role is None:
            return self.segmentation_root / img_path.relative_to(self.data_dir)

        split_dir = img_path.parent
        mapping = self._frame_index_mapping_cache.get(split_dir)
        if mapping is None:
            mapping_path = split_dir / 'frame_index_mapping.json'
            if not mapping_path.is_file():
                raise FileNotFoundError(f'frame_index_mapping.json not found in {split_dir}')
            with mapping_path.open(encoding='utf-8') as f:
                mapping = json.load(f)
            self._frame_index_mapping_cache[split_dir] = mapping

        record = mapping.get(img_path.name)
        if record is None:
            mapping_path = split_dir / 'frame_index_mapping.json'
            raise KeyError(f'{img_path.name} not found in {mapping_path}')
        source_frame_index = int(record[f'{self.mask_camera_role}_source_frame_index'])

        return (
            self.segmentation_root / 'segmentation_masks' / self.mask_session_id
            / self.mask_camera_role / f'mask{source_frame_index:08d}.png'
        )

    def _get_single_item(self, idx: int) -> ExampleDict:
        """Get a single item from the dataset."""
        img_path = self.image_list[idx]

        # read image from file and apply transformations (if any)
        if self.num_channels == 1:
            image = Image.open(img_path).convert('L').convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        if self.imgaug_pipeline is not None:
            # expands add batch dim for imgaug
            transformed_images = self.imgaug_pipeline(
                images=np.expand_dims(np.asarray(image), axis=0)
            )
            # get rid of the batch dim
            transformed_images = transformed_images[0]
        else:
            transformed_images = image

        transformed_tensor = cast(torch.Tensor, self.pytorch_transform(transformed_images))

        example = ExampleDict(
            image=transformed_tensor,  # shape (3, img_height, img_width)
            video=img_path.parts[-2],
            idx=idx,
            image_path=str(img_path),
        )
        if self.segmentation_root is not None:
            example['mask'] = self._load_mask(self._resolve_mask_path(img_path))

        return example
