"""Trial-indexed ground-truth frame loading for estimated-token PSNR/SSIM metrics.

Beast's pipeline never precomputes a target-image `.npz` for neurally-estimated
(step3-unprojected) tokens, unlike combined-npz mode's `--target-images-npz`. Instead, the raw
frame each `(trial_split, neural_trial_idx, neural_bin_idx)` came from can be recovered directly
from the eval-layout camera input directory's `<split>/frame_index_mapping.json` — the same file
`beast.inference._list_eval_layout_split_stems` reads when building step0's shards, so a trial
estimated by step3 can be matched back to its original left/right camera frames on disk.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from beast.inference import _load_image_tensor


def load_frame_index_mapping(input_dir: Path, split: str) -> dict[int, dict[int, Path]]:
    """Invert one camera's `<split>/frame_index_mapping.json` into a trial/bin -> path lookup.

    Args:
        input_dir: eval-layout camera directory passed as `beast predict --input` (holds
            `<split>/frame_index_mapping.json` plus the raw frame image files themselves).
        split: split name (`'train'`, `'val'`, or `'test'`).

    Returns:
        `{neural_trial_idx: {neural_bin_idx: frame_path}}`. Empty if `split` has no mapping file.

    Raises:
        ValueError: if two frames in the mapping share the same `(neural_trial_idx,
            neural_bin_idx)`.
    """
    mapping_path = Path(input_dir) / split / 'frame_index_mapping.json'
    if not mapping_path.is_file():
        return {}
    with mapping_path.open(encoding='utf-8') as f:
        mapping = json.load(f)

    out: dict[int, dict[int, Path]] = {}
    for filename, record in mapping.items():
        tid = int(record['neural_trial_idx'])
        bid = int(record['neural_bin_idx'])
        bins = out.setdefault(tid, {})
        if bid in bins:
            raise ValueError(
                f'{mapping_path}: duplicate neural_bin_idx={bid} for neural_trial_idx={tid}',
            )
        bins[bid] = mapping_path.parent / filename
    return out


def load_source_frame_index_mapping(
    input_dir: Path, split: str, role_key: str,
) -> dict[int, dict[int, int]]:
    """Invert one camera's `<split>/frame_index_mapping.json` into a trial/bin -> source index.

    Reads the `f'{role_key}_source_frame_index'` field written by
    `beast.preprocess.cheese3d.extract_cheese3d_eval_frames._write_role_mapping`, which is the
    frame index precomputed segmentation masks are named after (see
    `beast.preprocess.sable.precompute_sam3_masks_eval`).

    Args:
        input_dir: eval-layout camera directory holding `<split>/frame_index_mapping.json`.
        split: split name (`'train'`, `'val'`, or `'test'`).
        role_key: `'left'` or `'right'` — selects the JSON field to read.

    Returns:
        `{neural_trial_idx: {neural_bin_idx: source_frame_index}}`. Empty if `split` has no
        mapping file.
    """
    mapping_path = Path(input_dir) / split / 'frame_index_mapping.json'
    if not mapping_path.is_file():
        return {}
    with mapping_path.open(encoding='utf-8') as f:
        mapping = json.load(f)

    out: dict[int, dict[int, int]] = {}
    for record in mapping.values():
        tid = int(record['neural_trial_idx'])
        bid = int(record['neural_bin_idx'])
        out.setdefault(tid, {})[bid] = int(record[f'{role_key}_source_frame_index'])
    return out


def _load_mask_tensor(path: Path, image_size: int) -> torch.Tensor:
    """Load a binary segmentation mask PNG and resize it to `image_size x image_size`.

    Mirrors `SABLEDataset._load_mask`'s threshold/resize semantics, so masking behaves
    identically to Sable's decode/render pipeline.

    Args:
        path: path to a single-channel mask PNG with values in `{0, 255}`.
        image_size: side length to resize the mask to.

    Returns:
        float32 tensor `[1, image_size, image_size]` with values in `{0, 1}`.
    """
    with Image.open(path) as img:
        arr = np.asarray(img.convert('L'), dtype=np.float32)
    mask = torch.from_numpy(arr > 0).to(torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    if mask.shape[-2] != image_size or mask.shape[-1] != image_size:
        mask = F.interpolate(mask, size=(image_size, image_size), mode='nearest')
    return mask.squeeze(0)  # [1, S, S]


def load_target_masks_for_trials(
    trial_split_labels: list[str],
    neural_trial_idx: np.ndarray,
    time_bins: int,
    mask_index_left: dict[str, dict[int, dict[int, int]]],
    mask_index_right: dict[str, dict[int, dict[int, int]]],
    segmentation_root: Path,
    eid: str,
    image_size: int,
) -> torch.Tensor:
    """Load segmentation masks for a block of estimated trials.

    Mask paths follow `beast.preprocess.sable.precompute_sam3_masks_eval`'s convention:
    `{segmentation_root}/segmentation_masks/{eid}/{left,right}/mask{source_frame_index:08d}.png`.

    Args:
        trial_split_labels: per-trial split label, length `K`.
        neural_trial_idx: per-trial neural trial id, shape `[K]`.
        time_bins: number of neural bins per trial (`T`).
        mask_index_left: per-split `load_source_frame_index_mapping` output for the left camera.
        mask_index_right: per-split `load_source_frame_index_mapping` output for the right camera.
        segmentation_root: root directory precomputed masks were written under.
        eid: session id (mask subdirectory name).
        image_size: side length to resize masks to.

    Returns:
        Float tensor shaped `[K, T, 2, 1, image_size, image_size]` with values in `{0, 1}` (view
        order: left, right).

    Raises:
        KeyError: if any `(split, neural_trial_idx, neural_bin_idx)` has no matching source frame
            index in either camera's mask index.
    """
    k = len(trial_split_labels)
    out = torch.empty((k, time_bins, 2, 1, image_size, image_size), dtype=torch.float32)
    missing: list[tuple[str, int, int]] = []
    segmentation_root = Path(segmentation_root)
    for i in range(k):
        split = str(trial_split_labels[i]).lower()
        tid = int(neural_trial_idx[i])
        for bid in range(time_bins):
            for view, (cam, mask_index) in enumerate(
                (('left', mask_index_left), ('right', mask_index_right)),
            ):
                source_frame_index = mask_index.get(split, {}).get(tid, {}).get(bid)
                if source_frame_index is None:
                    missing.append((split, tid, bid))
                    continue
                mask_path = (
                    segmentation_root
                    / 'segmentation_masks'
                    / eid
                    / cam
                    / f'mask{source_frame_index:08d}.png'
                )
                out[i, bid, view] = _load_mask_tensor(mask_path, image_size)
    if missing:
        raise KeyError(
            f'{len(missing)} (split, neural_trial_idx, neural_bin_idx) row(s) have no matching '
            f'source frame index for masking; first few: {missing[:5]}',
        )
    return out


def load_target_images_for_trials(
    trial_split_labels: list[str],
    neural_trial_idx: np.ndarray,
    time_bins: int,
    mapping_left: dict[str, dict[int, dict[int, Path]]],
    mapping_right: dict[str, dict[int, dict[int, Path]]],
    image_size: int,
) -> torch.Tensor:
    """Load ground-truth left/right frames for a block of estimated trials.

    Args:
        trial_split_labels: per-trial split label, length `K`.
        neural_trial_idx: per-trial neural trial id, shape `[K]`.
        time_bins: number of neural bins per trial (`T`).
        mapping_left: per-split `load_frame_index_mapping` output for the left camera.
        mapping_right: per-split `load_frame_index_mapping` output for the right camera.
        image_size: side length passed to `beast.inference._load_image_tensor`.

    Returns:
        Float tensor shaped `[K, T, 2, 3, image_size, image_size]` in `[0, 1]` (view order:
        left, right).

    Raises:
        KeyError: if any `(split, neural_trial_idx, neural_bin_idx)` has no matching frame in
            either camera's mapping.
    """
    k = len(trial_split_labels)
    out = torch.empty((k, time_bins, 2, 3, image_size, image_size), dtype=torch.float32)
    missing: list[tuple[str, int, int]] = []
    for i in range(k):
        split = str(trial_split_labels[i]).lower()
        tid = int(neural_trial_idx[i])
        for bid in range(time_bins):
            for view, mapping in ((0, mapping_left), (1, mapping_right)):
                path = mapping.get(split, {}).get(tid, {}).get(bid)
                if path is None:
                    missing.append((split, tid, bid))
                    continue
                out[i, bid, view] = _load_image_tensor(path, image_size)
    if missing:
        raise KeyError(
            f'{len(missing)} (split, neural_trial_idx, neural_bin_idx) row(s) have no matching '
            f'target frame; first few: {missing[:5]}',
        )
    return out
