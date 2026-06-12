"""Data utilities for splitting multi-view batches into input and target subsets."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import default_collate


class SplitData(nn.Module):
    """Split a fixed-size multi-view batch into input and target subsets."""

    def __init__(self, config: dict) -> None:
        """Initialize.

        Args:
            config: full training config dict; reads config['training'] for view counts.
        """
        super().__init__()
        self.config = config
        self.total_views = int(self.config['training']['num_views'])
        self.num_input_views = int(self.config['training']['num_input_views'])
        self.num_target_views = int(self.config['training']['num_target_views'])
        self.input_pattern, self.target_pattern = self._build_indices(
            total_views=self.total_views,
            num_input_views=self.num_input_views,
            num_target_views=self.num_target_views,
        )

    @torch.no_grad()
    def forward(
        self,
        data_batch: dict,
        random_index: bool = True,
    ) -> tuple[SimpleNamespace, SimpleNamespace, torch.Tensor, torch.Tensor]:
        """Split batch into input and target dicts.

        Args:
            data_batch: batch dict with tensor values of shape [B, V, ...].
            random_index: whether to sample random view indices.

        Returns:
            tuple of (input_data, target_data, input_pattern, target_pattern).
        """
        input_dict, target_dict = {}, {}
        device = data_batch['image'].device
        batch_size = data_batch['image'].shape[0]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)

        if 'context_indices' in data_batch and 'target_indices' in data_batch:
            input_pattern = data_batch['context_indices'].to(device=device, dtype=torch.long)
            target_pattern = data_batch['target_indices'].to(device=device, dtype=torch.long)
        else:
            if random_index:
                input_pattern, target_pattern = self.get_random_index(
                    batch_size, data_batch['image'].shape[1], device,
                )
            else:
                input_pattern = self.input_pattern.to(device).unsqueeze(0).repeat(batch_size, 1)
                target_pattern = self.target_pattern.to(device).unsqueeze(0).repeat(batch_size, 1)

        for key, value in data_batch.items():
            if key in {'scene_name', 'context_indices', 'target_indices'}:
                continue
            input_dict[key] = value[batch_idx, input_pattern, ...]
            target_dict[key] = value[batch_idx, target_pattern, ...]

        return SimpleNamespace(**input_dict), SimpleNamespace(**target_dict), input_pattern, target_pattern

    def _build_indices(
        self,
        total_views: int,
        num_input_views: int,
        num_target_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_input_views == total_views and num_target_views == total_views:
            indices = torch.arange(total_views, dtype=torch.long)
            return indices, indices.clone()

        if num_input_views + num_target_views != total_views:
            raise ValueError(
                'Unsupported view allocation: expected either disjoint '
                '`num_input_views + num_target_views == num_views` or full-overlap '
                '`num_input_views == num_target_views == num_views`, '
                'or ibl_training_regime=nvs with input_view_indices / target_view_indices '
                'set via config (context/target indices are then built in the dataset class).'
            )

        # Disjoint split: use config-driven indices when available (NVS regime),
        # otherwise fall back to GCD-based interleaving.
        training_cfg = self.config.get('training', {})
        regime = training_cfg.get('ibl_training_regime', 'two_input_reconstruction')

        if regime == 'nvs':
            input_view_indices = training_cfg.get('input_view_indices')
            target_view_indices = training_cfg.get('target_view_indices')
            if input_view_indices is not None and target_view_indices is not None:
                return (
                    torch.tensor(input_view_indices, dtype=torch.long),
                    torch.tensor(target_view_indices, dtype=torch.long),
                )
            # Fall through to GCD-based if not configured — not recommended for NVS

        group_count = math.gcd(num_input_views, num_target_views)
        group_size = total_views // group_count
        input_per_group = num_input_views // group_count
        target_per_group = num_target_views // group_count

        input_indices = []
        target_indices = []
        for group_idx in range(group_count):
            start = group_idx * group_size
            block = list(range(start, start + group_size))
            input_indices.extend(block[:input_per_group])
            target_indices.extend(block[input_per_group: input_per_group + target_per_group])

        input_indices = torch.tensor(sorted(input_indices), dtype=torch.long)
        target_indices = torch.tensor(sorted(target_indices), dtype=torch.long)
        return input_indices, target_indices

    def get_random_index(
        self,
        batch_size: int,
        num_views: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random input/target view index patterns.

        Args:
            batch_size: number of items in the batch.
            num_views: total number of views available.
            device: target device for output tensors.

        Returns:
            tuple of (input_indices [B, n_in], target_indices [B, n_tgt]).
        """
        num_input_views = self.config['training']['num_input_views']
        num_target_views = self.config['training']['num_target_views']
        random_shuffle = self.config['training']['view_selector'].get('shuffle', False)

        if num_input_views == num_views and num_target_views == num_views:
            indices = torch.arange(
                num_views, dtype=torch.long, device=device,
            ).unsqueeze(0).repeat(batch_size, 1)
            return indices, indices.clone()

        assert (
            num_input_views + num_target_views == self.config['training']['num_views']
        ), 'Mismatch in total views allocation.'

        input_indices = torch.zeros(
            (batch_size, num_input_views), dtype=torch.long, device=device,
        )
        target_indices = torch.zeros(
            (batch_size, num_target_views), dtype=torch.long, device=device,
        )

        for batch_idx in range(batch_size):
            input_indices[batch_idx, 0] = 0
            input_indices[batch_idx, -1] = num_views - 1

            remaining = torch.arange(1, num_views - 1, device=device)
            middle_count = num_input_views - 2
            middle = remaining[torch.randperm(len(remaining))[:middle_count]]
            middle, _ = middle.sort()
            input_indices[batch_idx, 1:-1] = middle

            target = torch.tensor(
                [
                    view_idx for view_idx in range(num_views)
                    if view_idx not in input_indices[batch_idx]
                ],
                dtype=torch.long,
                device=device,
            )
            target, _ = target.sort()
            target_indices[batch_idx] = target

            if random_shuffle:
                input_indices[batch_idx] = input_indices[batch_idx][
                    torch.randperm(num_input_views)
                ]
                target_indices[batch_idx] = target_indices[batch_idx][
                    torch.randperm(num_target_views)
                ]

        return input_indices, target_indices


def pad_correspondence_fields_to_batch_max(batch: list[dict]) -> list[dict]:
    """Pad correspondence tensors to max length in batch so default_collate can stack.

    Args:
        batch: list of dicts, each containing correspondence tensors.

    Returns:
        list of dicts, each containing padded correspondence tensors.
    """
    if not batch:
        return batch
    keys = ('leftcamera_xy', 'rightcamera_xy', 'confidence')
    if not all(k in batch[0] for k in keys):
        return batch
    max_n = max(int(sample['leftcamera_xy'].shape[0]) for sample in batch)
    out: list[dict] = []
    for sample in batch:
        sample = dict(sample)
        n = int(sample['leftcamera_xy'].shape[0])
        if n < max_n:
            pad = max_n - n
            lt = sample['leftcamera_xy']
            rt = sample['rightcamera_xy']
            cf = sample['confidence']
            sample['leftcamera_xy'] = torch.cat(
                [lt, torch.full((pad, 2), -1.0, dtype=lt.dtype, device=lt.device)], dim=0,
            )
            sample['rightcamera_xy'] = torch.cat(
                [rt, torch.full((pad, 2), -1.0, dtype=rt.dtype, device=rt.device)], dim=0,
            )
            sample['confidence'] = torch.cat(
                [cf, torch.zeros(pad, dtype=cf.dtype, device=cf.device)], dim=0,
            )
        out.append(sample)
    return out


def normalize_optional_pose_fields(batch: list[dict]) -> list[dict]:
    """Ensure optional pose supervision keys are consistent across a batch.

    Some samples may not have pose supervision available. When any sample has
    ``pose``, inject a zero placeholder plus ``pose_valid=False`` for the rest.

    Args:
        batch: list of sample dicts.

    Returns:
        list of sample dicts with consistent pose keys.
    """
    if not batch:
        return batch
    if not all(isinstance(sample, dict) for sample in batch):
        return batch

    pose_template = None
    for sample in batch:
        pose = sample.get('pose')
        if isinstance(pose, torch.Tensor):
            pose_template = pose
            break

    if pose_template is None:
        return batch

    pose_shape = tuple(pose_template.shape)
    pose_valid_shape = pose_shape[:-1]
    out: list[dict] = []
    for sample in batch:
        sample = dict(sample)
        pose = sample.get('pose')
        if isinstance(pose, torch.Tensor):
            if 'pose_valid' not in sample:
                sample['pose_valid'] = torch.ones(
                    pose.shape[:-1],
                    dtype=torch.bool,
                    device=pose.device,
                )
        else:
            sample['pose'] = torch.zeros(
                pose_shape,
                dtype=pose_template.dtype,
                device=pose_template.device,
            )
            sample['pose_valid'] = torch.zeros(
                pose_valid_shape,
                dtype=torch.bool,
                device=pose_template.device,
            )
        out.append(sample)
    return out


def collate_with_correspondence_padding(batch: list[Any]):
    """Collate a batch with correspondence padding and optional pose normalisation."""
    batch = normalize_optional_pose_fields(batch)
    batch = pad_correspondence_fields_to_batch_max(batch)
    return default_collate(batch)
