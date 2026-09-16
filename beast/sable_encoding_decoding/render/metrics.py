"""PSNR/SSIM evaluation for rendered vs. target images.

Requires ``pip install 'torchmetrics[image]'``.

SSIM and PSNR evaluation checklist:

- Ensure tensors are in shape `(B, 3, H, W)`.
- Normalize all images to `[0, 1]`, then use a fixed setting `data_range = 1.0`.
- SSIM with an 11x11 Gaussian window requires `H, W >= 11`.

Metrics `.npz` schema (`K` trials, `T` time bins, `V` views):

- `psnr`: `[K, T, V]`
- `ssim`: `[K, T, V]`
- `average_psnr`: scalar mean of `psnr`
- `average_ssim`: scalar mean of `ssim`
- `se_psnr`: scalar standard error of `psnr` (`nanstd / sqrt(n)`)
- `se_ssim`: scalar standard error of `ssim` (`nanstd / sqrt(n)`)
- `neural_trial_idx`: `[K]`
- `neural_bin_idx`: `[K, T]`
- `trial_split`: `[K]`
- `source_files`: `[K]`

Ordinary-inference `.npz` schema (flat `N = samples * views` records, no trial/bin/neural
metadata, produced by `save_inference_psnr_ssim_metrics_npz`):

- `session_id`: `[N]`
- `scene_name`: `[N]`
- `view_name`: `[N]`
- `psnr`: `[N]`
- `ssim`: `[N]`
- `average_psnr`: scalar mean of `psnr`
- `average_ssim`: scalar mean of `ssim`
- `sd_psnr`: scalar standard deviation of `psnr` (`nanstd`)
- `sd_ssim`: scalar standard deviation of `ssim` (`nanstd`)
- `se_psnr`: scalar standard error of `psnr` (`nanstd / sqrt(n)`)
- `se_ssim`: scalar standard error of `ssim` (`nanstd / sqrt(n)`)

Ordinary inference emits the `K/T/V` schema instead when given a neural-data root
(`beast predict --neural-input-dir`): `NeuralTrialMetricsAccumulator` sizes `K` and `T` from
each session's aligned neural `.npz`, so the metrics line up index-for-index with the neural
trials, and writes one file per session.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def _flatten_image_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    metric_name: str,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Flatten tensors shaped `[B, V, C, H, W]` to `[B*V, C, H, W]`."""
    if pred.shape != target.shape:
        raise ValueError(
            f'{metric_name} expects matching shapes; got pred={pred.shape} target={target.shape}',
        )
    if pred.ndim != 5:
        raise ValueError(f'{metric_name} expects [B,V,C,H,W]; got {pred.shape}')

    bsz, views, channels, height, width = pred.shape
    return (
        pred.reshape(bsz * views, channels, height, width),
        target.reshape(bsz * views, channels, height, width),
        bsz,
        views,
    )


def resize_image_batch(
    x: torch.Tensor,
    image_size: int,
    *,
    mode: str = 'bilinear',
) -> torch.Tensor:
    """Resize a `[B, V, C, H, W]` batch to `image_size x image_size`.

    Used to score renders produced at one resolution (e.g. beast/resnet's 224x224) at another
    (e.g. SABLE's 320x320) so PSNR/SSIM are comparable across models. Bilinear resizing is
    antialiased, matching the PIL-style resampling applied to ground-truth frames; use
    `mode='nearest'` for binary masks.

    Args:
        x: tensor shaped `[B, V, C, H, W]`.
        image_size: target side length.
        mode: `F.interpolate` mode, `'bilinear'` (antialiased) or `'nearest'`.

    Returns:
        float32 tensor shaped `[B, V, C, image_size, image_size]`; `x` itself (dtype unchanged)
        when already that size.

    Raises:
        ValueError: if `x` is not rank 5.
    """
    if x.ndim != 5:
        raise ValueError(f'resize_image_batch expects [B,V,C,H,W]; got {tuple(x.shape)}')
    bsz, views, channels, height, width = x.shape
    if height == image_size and width == image_size:
        return x
    # interpolate in float32: antialiased resampling is not supported for every half dtype
    flat = x.reshape(bsz * views, channels, height, width).float()
    kwargs: dict[str, bool] = {'antialias': True, 'align_corners': False}
    if mode == 'nearest':
        kwargs = {}
    resized = F.interpolate(flat, size=(image_size, image_size), mode=mode, **kwargs)
    return resized.reshape(bsz, views, channels, image_size, image_size)


def _psnr_per_image(
    pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0,
) -> torch.Tensor:
    """Compute PSNR for tensors shaped `[B, V, C, H, W]`, returning `[B, V]`."""
    pred, target, bsz, views = _flatten_image_batch(pred, target, metric_name='PSNR')

    metric = PeakSignalNoiseRatio(
        data_range=data_range, dim=(1, 2, 3), reduction='none',
    ).to(pred.device)
    psnr = metric(pred.detach().float(), target.detach().float())
    return psnr.reshape(bsz, views)


def _ssim_per_image(
    pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0,
) -> torch.Tensor:
    """Compute SSIM for tensors shaped `[B, V, C, H, W]`, returning `[B, V]`."""
    pred, target, bsz, views = _flatten_image_batch(pred, target, metric_name='SSIM')

    if min(pred.shape[-2:]) < 11:
        raise ValueError(
            f'SSIM with an 11x11 Gaussian window requires H,W >= 11; got {pred.shape[-2:]}',
        )

    metric = StructuralSimilarityIndexMeasure(
        data_range=data_range,
        gaussian_kernel=True,
        kernel_size=11,
        reduction='none',
    ).to(pred.device)
    ssim = metric(pred.detach().float(), target.detach().float())
    return ssim.reshape(bsz, views)


def apply_segmentation_mask(
    render: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out background pixels in a render/target pair using a foreground mask.

    Args:
        render: predicted image tensor, e.g. `[B, V, 3, H, W]`.
        target: ground-truth image tensor, same shape as `render`.
        mask: foreground mask (`1` = keep, `0` = zero out), broadcastable to `render`'s
            shape (e.g. `[B, V, 1, H, W]`).

    Returns:
        `(masked_render, masked_target)`, each the same shape/dtype as the inputs.
    """
    mask = mask.to(device=render.device, dtype=render.dtype)
    return render * mask, target * mask


def _image_metrics_by_view(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PSNR and SSIM as numpy arrays shaped `[B, V]`."""
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float()

    data_range = 1.0
    psnr = _psnr_per_image(pred, target, data_range=data_range)
    ssim = _ssim_per_image(pred, target, data_range=data_range)
    return (
        psnr.detach().cpu().numpy().astype(np.float32),
        ssim.detach().cpu().numpy().astype(np.float32),
    )


def load_neural_trial_idx(path: Path, *, k_trials: int, t_bins: int) -> np.ndarray:
    """Return one neural trial id per K row when available, else a stable placeholder."""
    with np.load(path, allow_pickle=True) as d:
        if 'neural_trial_idx' not in d.files:
            return np.full((k_trials,), -1, dtype=np.int64)
        trial_idx = np.asarray(d['neural_trial_idx'], dtype=np.int64).reshape(-1)

    if len(trial_idx) == k_trials:
        return trial_idx
    if len(trial_idx) == k_trials * t_bins:
        return trial_idx.reshape(k_trials, t_bins)[:, 0]
    raise ValueError(
        f'{path}: neural_trial_idx length {len(trial_idx)} cannot index z shape '
        f'({k_trials}, {t_bins}, ...)',
    )


def load_neural_bin_idx(path: Path, *, k_trials: int, t_bins: int) -> np.ndarray:
    """Return neural bin ids shaped `[K, T]` when available, else `[0, ..., T-1]`."""
    with np.load(path, allow_pickle=True) as d:
        if 'neural_bin_idx' not in d.files:
            return np.tile(np.arange(t_bins, dtype=np.int64), (k_trials, 1))
        bin_idx = np.asarray(d['neural_bin_idx'], dtype=np.int64).reshape(-1)

    if len(bin_idx) == k_trials * t_bins:
        return bin_idx.reshape(k_trials, t_bins)
    raise ValueError(
        f'{path}: neural_bin_idx length {len(bin_idx)} cannot index z shape '
        f'({k_trials}, {t_bins}, ...)',
    )


def load_trial_split(path: Path, *, k_trials: int, t_bins: int) -> np.ndarray:
    """Return one split label per K row when available, else a stable placeholder."""
    with np.load(path, allow_pickle=True) as d:
        if 'trial_split' not in d.files:
            return np.full((k_trials,), 'unknown', dtype=str)
        split = np.asarray(
            [str(x).lower() for x in np.asarray(d['trial_split'], dtype=object).reshape(-1)],
        )

    if len(split) == k_trials:
        return split.astype(str)
    if len(split) == k_trials * t_bins:
        split_by_time = split.reshape(k_trials, t_bins)
        first_split = split_by_time[:, 0]
        if not np.all(split_by_time == first_split[:, None]):
            raise ValueError(f'{path}: trial_split varies across time bins for the same trial')
        return first_split.astype(str)
    raise ValueError(
        f'{path}: trial_split length {len(split)} cannot index z shape '
        f'({k_trials}, {t_bins}, ...)',
    )


def resolve_metrics_npz_path(metrics_npz: Path | None, out_dir: Path) -> Path:
    """Return the explicit metrics path or the shared default under `out_dir`."""
    return (
        Path(metrics_npz).resolve()
        if metrics_npz is not None
        else Path(out_dir) / 'psnr_ssim_metrics.npz'
    )


def collect_psnr_ssim_metrics_block(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_npz: Path,
    *,
    k_trials: int,
    t_bins: int,
    neural_trial_idx: np.ndarray | None = None,
    neural_bin_idx: np.ndarray | None = None,
    trial_split: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute one canonical metrics block for a token file.

    `neural_trial_idx`/`neural_bin_idx`/`trial_split`, when given, are used directly instead of
    being read from `source_npz` — for callers (e.g. estimated-mode decoding) that already have
    this metadata in memory and have no single npz file to read it back from.

    Returns:
        PSNR/SSIM shaped `[K, T, V]`, neural trial ids shaped `[K]`, neural bin ids shaped
        `[K, T]`, split labels shaped `[K]`, and one source-file row per K trial.
    """
    psnr_flat, ssim_flat = _image_metrics_by_view(pred, target)
    metric_views = psnr_flat.shape[1]
    source_npz = Path(source_npz)
    return (
        psnr_flat.reshape(k_trials, t_bins, metric_views).astype(np.float32),
        ssim_flat.reshape(k_trials, t_bins, metric_views).astype(np.float32),
        (
            neural_trial_idx
            if neural_trial_idx is not None
            else load_neural_trial_idx(source_npz, k_trials=k_trials, t_bins=t_bins)
        ),
        (
            neural_bin_idx
            if neural_bin_idx is not None
            else load_neural_bin_idx(source_npz, k_trials=k_trials, t_bins=t_bins)
        ),
        (
            trial_split
            if trial_split is not None
            else load_trial_split(source_npz, k_trials=k_trials, t_bins=t_bins)
        ),
        [str(source_npz)] * k_trials,
    )


def reassemble_flat_row_metrics(
    psnr_blocks: list[np.ndarray],
    ssim_blocks: list[np.ndarray],
    neural_trial_blocks: list[np.ndarray],
    neural_bin_blocks: list[np.ndarray],
    trial_split_blocks: list[np.ndarray],
    *,
    k_trials: int,
    t_bins: int,
    views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape per-row (single-view) metric chunks collected over a flat K*T*V decode loop back
    into the canonical `[K, T, V]` block.

    Callers that decode each `(trial, bin, view)` row independently (one row per single-camera
    decode) accumulate `psnr`/`ssim` chunks shaped `[chunk, 1, 1]` — via
    `collect_psnr_ssim_metrics_block(..., k_trials=chunk, t_bins=1)` — and matching per-row
    `neural_trial_idx`/`neural_bin_idx`/`trial_split` chunks, over a loop that flattens `(K, T,
    V)` into one batch axis in row-major order. This concatenates those chunks back into that
    flat `K*T*V` axis and reshapes to `[K, T, V]` (metrics) / `[K]` / `[K, T]` / `[K]`
    (metadata), recovering the structure `.reshape(k*t*v, ...)` collapsed.

    Returns:
        `(psnr, ssim, neural_trial_idx, neural_bin_idx, trial_split)` shaped `[K, T, V]`, `[K,
        T, V]`, `[K]`, `[K, T]`, `[K]`.
    """
    psnr = np.concatenate(psnr_blocks, axis=0).reshape(k_trials, t_bins, views)
    ssim = np.concatenate(ssim_blocks, axis=0).reshape(k_trials, t_bins, views)
    trial_idx = np.concatenate(neural_trial_blocks, axis=0).reshape(
        k_trials, t_bins, views,
    )[:, 0, 0]
    bin_idx = np.concatenate(neural_bin_blocks, axis=0).reshape(k_trials, t_bins, views)[:, :, 0]
    trial_split = np.concatenate(trial_split_blocks, axis=0).reshape(
        k_trials, t_bins, views,
    )[:, 0, 0]
    return psnr, ssim, trial_idx, bin_idx, trial_split


def save_psnr_ssim_metrics_npz(
    metrics_npz: Path,
    *,
    psnr_blocks: list[np.ndarray],
    ssim_blocks: list[np.ndarray],
    neural_trial_blocks: list[np.ndarray],
    neural_bin_blocks: list[np.ndarray],
    trial_split_blocks: list[np.ndarray],
    source_file_rows: list[str],
    view_names: tuple[str, ...] = ('left', 'right'),
) -> dict[str, np.ndarray]:
    """Save PSNR/SSIM metrics with the shared compressed NPZ structure.

    Raises:
        RuntimeError: if `psnr_blocks` is empty (nothing to save).
    """
    if not psnr_blocks:
        raise RuntimeError('No metrics were collected.')

    psnr = np.concatenate(psnr_blocks, axis=0).astype(np.float32)
    ssim = np.concatenate(ssim_blocks, axis=0).astype(np.float32)
    neural_trial_idx = np.concatenate(neural_trial_blocks, axis=0).astype(np.int64)
    neural_bin_idx = np.concatenate(neural_bin_blocks, axis=0).astype(np.int64)
    trial_split = np.concatenate(trial_split_blocks, axis=0).astype(str)
    n_psnr = np.sum(~np.isnan(psnr))
    n_ssim = np.sum(~np.isnan(ssim))
    se_psnr = np.nanstd(psnr) / np.sqrt(n_psnr) if n_psnr > 0 else np.nan
    se_ssim = np.nanstd(ssim) / np.sqrt(n_ssim) if n_ssim > 0 else np.nan
    arrays = {
        'psnr': psnr,
        'ssim': ssim,
        'average_psnr': np.asarray(np.nanmean(psnr), dtype=np.float32),
        'average_ssim': np.asarray(np.nanmean(ssim), dtype=np.float32),
        'se_psnr': np.asarray(se_psnr, dtype=np.float32),
        'se_ssim': np.asarray(se_ssim, dtype=np.float32),
        'neural_trial_idx': neural_trial_idx,
        'neural_bin_idx': neural_bin_idx,
        'trial_split': trial_split,
        'source_files': np.asarray(source_file_rows, dtype=str),
        'view_names': np.asarray(view_names, dtype=str),
    }

    metrics_npz = Path(metrics_npz)
    metrics_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metrics_npz, **arrays)
    return arrays


def combine_view_metrics_npz(
    view_npz_paths: dict[str, Path],
    metrics_npz: Path,
) -> dict[str, np.ndarray]:
    """Merge per-view flat inference metrics npz files into one combined npz.

    Each input npz is expected to have the ordinary-inference schema written by
    `save_inference_psnr_ssim_metrics_npz` (one flat record per sample). This overwrites each
    input's `view_name` field with its key in `view_npz_paths` (ordinary inference has no view
    concept, so `view_name` is otherwise a placeholder), then concatenates all views' records
    and recomputes the aggregate stats.

    Args:
        view_npz_paths: mapping of view label (e.g. `'left'`, `'right'`) to that view's
            `psnr_ssim_metrics.npz` path. A view whose path does not exist is skipped.
        metrics_npz: output path for the combined npz.

    Returns:
        dict of the arrays written (same keys as the saved `.npz`).

    Raises:
        RuntimeError: if no listed view npz exists (nothing to combine).
    """
    session_ids: list[str] = []
    scene_names: list[str] = []
    view_names: list[str] = []
    psnr_parts: list[np.ndarray] = []
    ssim_parts: list[np.ndarray] = []

    for view_name, view_npz in view_npz_paths.items():
        view_npz = Path(view_npz)
        if not view_npz.exists():
            continue
        with np.load(view_npz, allow_pickle=True) as d:
            n_records = len(d['psnr'])
            session_ids.extend(d['session_id'].tolist())
            scene_names.extend(d['scene_name'].tolist())
            view_names.extend([view_name] * n_records)
            psnr_parts.append(d['psnr'])
            ssim_parts.append(d['ssim'])

    if not psnr_parts:
        raise RuntimeError(f'No metrics to combine; none of {list(view_npz_paths)} exist.')

    return save_inference_psnr_ssim_metrics_npz(
        metrics_npz,
        session_ids=session_ids,
        scene_names=scene_names,
        view_names=view_names,
        psnr=np.concatenate(psnr_parts, axis=0),
        ssim=np.concatenate(ssim_parts, axis=0),
    )


def save_inference_psnr_ssim_metrics_npz(
    metrics_npz: Path,
    *,
    session_ids: list[str],
    scene_names: list[str],
    view_names: list[str],
    psnr: np.ndarray,
    ssim: np.ndarray,
) -> dict[str, np.ndarray]:
    """Save per-sample PSNR/SSIM records from ordinary `beast predict` inference.

    Unlike `save_psnr_ssim_metrics_npz` (`K`-trial/`T`-bin neural-decode schema), this saves
    one flat record per `(batch item, view)` pair, since ordinary inference batches carry no
    trial/bin/neural metadata.

    Args:
        metrics_npz: output path (parent dirs created if missing).
        session_ids: one session id per record, length `N`.
        scene_names: one scene name per record, length `N`.
        view_names: one view label per record (e.g. `'view00'`), length `N`.
        psnr: `[N]` float32 array.
        ssim: `[N]` float32 array.

    Returns:
        dict of the arrays written (same keys as the saved `.npz`).

    Raises:
        RuntimeError: if `psnr` is empty (nothing to save).
    """
    if psnr.size == 0:
        raise RuntimeError('No metrics were collected.')

    n_psnr = np.sum(~np.isnan(psnr))
    n_ssim = np.sum(~np.isnan(ssim))
    sd_psnr = np.nanstd(psnr) if n_psnr > 0 else np.nan
    sd_ssim = np.nanstd(ssim) if n_ssim > 0 else np.nan
    se_psnr = sd_psnr / np.sqrt(n_psnr) if n_psnr > 0 else np.nan
    se_ssim = sd_ssim / np.sqrt(n_ssim) if n_ssim > 0 else np.nan
    arrays = {
        'session_id': np.asarray(session_ids, dtype=str),
        'scene_name': np.asarray(scene_names, dtype=str),
        'view_name': np.asarray(view_names, dtype=str),
        'psnr': psnr.astype(np.float32),
        'ssim': ssim.astype(np.float32),
        'average_psnr': np.asarray(np.nanmean(psnr), dtype=np.float32),
        'average_ssim': np.asarray(np.nanmean(ssim), dtype=np.float32),
        'sd_psnr': np.asarray(sd_psnr, dtype=np.float32),
        'sd_ssim': np.asarray(sd_ssim, dtype=np.float32),
        'se_psnr': np.asarray(se_psnr, dtype=np.float32),
        'se_ssim': np.asarray(se_ssim, dtype=np.float32),
    }

    metrics_npz = Path(metrics_npz)
    metrics_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metrics_npz, **arrays)
    return arrays
