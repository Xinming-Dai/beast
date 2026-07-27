"""PSNR/SSIM/temporal-consistency evaluation for rendered vs. target images.

Requires ``pip install 'torchmetrics[image]'``.

SSIM and PSNR evaluation checklist:

- Ensure tensors are in shape `(B, 3, H, W)`.
- Normalize all images to `[0, 1]`, then use a fixed setting `data_range = 1.0`.
- SSIM with an 11x11 Gaussian window requires `H, W >= 11`.

Metrics `.npz` schema (`K` trials, `T` time bins, `V` views):

- `psnr`: `[K, T, V]`
- `ssim`: `[K, T, V]`
- `temporal_delta_l1`: `[K, T-1, V]`
- `pred_motion_energy`: `[K, T-1, V]`
- `target_motion_energy`: `[K, T-1, V]`
- `motion_energy_ratio`: `[K, T-1, V]` (NaN where the target pair has no motion)
- `motion_energy_corr`: `[K, V]` (Pearson r of motion energy over time)
- `neural_trial_idx`: `[K]`
- `neural_bin_idx`: `[K, T]`
- `trial_split`: `[K]`
- `source_files`: `[K]`
"""

from pathlib import Path

import numpy as np
import torch
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Compute one canonical metrics block for a token file.

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
        load_neural_trial_idx(source_npz, k_trials=k_trials, t_bins=t_bins),
        load_neural_bin_idx(source_npz, k_trials=k_trials, t_bins=t_bins),
        load_trial_split(source_npz, k_trials=k_trials, t_bins=t_bins),
        [str(source_npz)] * k_trials,
    )


def _as_image_sequence(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    k_trials: int,
    t_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Reshape a flat `[K*T, V, C, H, W]` decoded render into `[K, T, V, C, H, W]`.

    Args:
        pred: tensor shaped `[K*T, V, C, H, W]`.
        target: tensor shaped `[K*T, V, C, H, W]` with the same shape as `pred`.
        k_trials: number of trials (`K`).
        t_bins: number of time bins per trial (`T`).

    Returns:
        Tuple `(pred_seq, target_seq, views)` reshaped to `[K, T, V, C, H, W]`. The
        intermediate `[K*T, V, C, H, W]` shape is preserved on the trial axis using a single
        reshape since the decoded layout is already a contiguous K-major, T-minor block.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f'pred/target shape mismatch: pred={pred.shape} target={target.shape}',
        )
    if pred.ndim != 5:
        raise ValueError(f'expected [K*T,V,C,H,W]; got {pred.shape}')
    if pred.shape[0] != k_trials * t_bins:
        raise ValueError(
            f'pred.shape[0]={pred.shape[0]} != k_trials*t_bins={k_trials*t_bins}',
        )
    views = pred.shape[1]
    channels, height, width = pred.shape[2], pred.shape[3], pred.shape[4]
    pred_seq = pred.reshape(k_trials, t_bins, views, channels, height, width)
    target_seq = target.reshape(k_trials, t_bins, views, channels, height, width)
    return pred_seq, target_seq, views


def collect_temporal_metrics_block(
    pred: torch.Tensor,
    target: torch.Tensor,
    source_npz: Path,
    *,
    k_trials: int,
    t_bins: int,
    motion_eps: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Compute frame-to-frame temporal metrics on rendered vs. target images.

    The decoded render layout is `[K*T, V, C, H, W]` (a single TCN pass flattens K and T into
    one leading dim). This function reshapes the pair to `[K, T, V, C, H, W]` via
    `_as_image_sequence`, then returns the metrics described in the module docstring.

    Args:
        pred: decoded render tensor shaped `[K*T, V, C, H, W]`.
        target: ground-truth image tensor shaped `[K*T, V, C, H, W]`.
        source_npz: path to the source token `.npz`; used to read `neural_bin_idx` so non-
            contiguous time bins are masked out as NaN.
        k_trials: number of trials per shard (`K`).
        t_bins: number of time bins per trial (`T`).
        motion_eps: minimum target motion energy (per view, in [0, 1] image units) below
            which a frame pair is excluded from `motion_energy_ratio` (avoids divide-by-zero
            explosions on near-static pairs).

    Returns:
        dict with keys `temporal_delta_l1`, `pred_motion_energy`, `target_motion_energy`,
        `motion_energy_ratio`, `motion_energy_corr`, all as float32 numpy arrays.
    """
    if pred.shape[0] != k_trials * t_bins:
        raise ValueError(
            f'pred.shape[0]={pred.shape[0]} does not match k_trials*t_bins={k_trials*t_bins}',
        )
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float()
    pred_seq, target_seq, views = _as_image_sequence(
        pred, target, k_trials=k_trials, t_bins=t_bins,
    )

    pred_seq = pred_seq.reshape(k_trials, t_bins, views, *pred_seq.shape[3:])
    target_seq = target_seq.reshape(k_trials, t_bins, views, *target_seq.shape[3:])

    pred_delta = pred_seq[:, 1:] - pred_seq[:, :-1]
    target_delta = target_seq[:, 1:] - target_seq[:, :-1]

    pred_motion = pred_delta.abs().mean(dim=(-3, -2, -1))
    target_motion = target_delta.abs().mean(dim=(-3, -2, -1))
    temporal_delta_l1 = (pred_delta - target_delta).abs().mean(dim=(-3, -2, -1))

    ratio = np.full(pred_motion.shape, np.nan, dtype=np.float32)
    valid_per_pair = (target_motion.cpu().numpy() > motion_eps).any(axis=-1)
    valid_per_pair &= np.all(np.isfinite(target_motion.cpu().numpy()) & np.isfinite(pred_motion.cpu().numpy()), axis=-1)
    ratio_3d = np.full(pred_motion.shape, np.nan, dtype=np.float32)
    valid_3d = (target_motion.cpu().numpy() > motion_eps)
    ratio_3d[valid_3d] = (pred_motion[valid_3d] / target_motion[valid_3d]).cpu().numpy()

    bin_idx = load_neural_bin_idx(Path(source_npz), k_trials=k_trials, t_bins=t_bins)
    contiguous = (bin_idx[:, 1:] - bin_idx[:, :-1]) == 1
    mask = contiguous & valid_per_pair
    mask_3d = mask[:, :, None]
    temporal_delta_l1_np = np.where(mask_3d, temporal_delta_l1.cpu().numpy(), np.nan)
    pred_motion_np = np.where(mask_3d, pred_motion.cpu().numpy(), np.nan)
    target_motion_np = np.where(mask_3d, target_motion.cpu().numpy(), np.nan)
    ratio_np = np.where(mask_3d, ratio_3d, np.nan)

    pred_motion_t = pred_motion.permute(0, 2, 1)
    target_motion_t = target_motion.permute(0, 2, 1)
    corr = _pearson_corr_per_trial_view(pred_motion_t, target_motion_t, mask)
    return {
        'temporal_delta_l1': temporal_delta_l1_np.astype(np.float32),
        'pred_motion_energy': pred_motion_np.astype(np.float32),
        'target_motion_energy': target_motion_np.astype(np.float32),
        'motion_energy_ratio': ratio_np.astype(np.float32),
        'motion_energy_corr': corr.astype(np.float32),
    }


def _pearson_corr_per_trial_view(
    pred_motion: torch.Tensor,
    target_motion: torch.Tensor,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute Pearson r per (trial, view) over the T-1 motion-energy time series.

    Args:
        pred_motion: tensor shaped `[K, V, T-1]` of predicted motion energy.
        target_motion: tensor shaped `[K, V, T-1]` of target motion energy.
        mask: numpy array shaped `[K, T-1]`; pairs with `False` are dropped from the
            correlation. NaN-valued entries in either series are also dropped.

    Returns:
        Numpy array shaped `[K, V]` with NaN where fewer than 2 valid pairs remain.
    """
    pred_np = pred_motion.detach().cpu().numpy()
    target_np = target_motion.detach().cpu().numpy()
    out = np.full(pred_np.shape[:2], np.nan, dtype=np.float32)
    for k in range(pred_np.shape[0]):
        for v in range(pred_np.shape[1]):
            sel = mask[k] & ~np.isnan(pred_np[k, v]) & ~np.isnan(target_np[k, v])
            if sel.sum() < 2:
                continue
            x = pred_np[k, v, sel]
            y = target_np[k, v, sel]
            x = x - x.mean()
            y = y - y.mean()
            denom = np.sqrt((x * x).sum() * (y * y).sum())
            if denom <= 0.0:
                continue
            out[k, v] = float((x * y).sum() / denom)
    return out


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
    temporal_blocks: dict[str, list[np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Save PSNR/SSIM metrics with the shared compressed NPZ structure.

    Args:
        metrics_npz: target `.npz` path.
        psnr_blocks: list of per-source PSNR arrays shaped `[K, T, V]`.
        ssim_blocks: list of per-source SSIM arrays shaped `[K, T, V]`.
        neural_trial_blocks: list of per-source neural trial id arrays shaped `[K]`.
        neural_bin_blocks: list of per-source neural bin id arrays shaped `[K, T]`.
        trial_split_blocks: list of per-source split-label arrays shaped `[K]`.
        source_file_rows: flat list with one entry per concatenated trial.
        view_names: ordered view labels.
        temporal_blocks: optional dict mapping temporal metric name to a list of per-source
            arrays shaped `[K, T-1, V]` (or `[K, V]` for `motion_energy_corr`). When provided
            each list must have the same length as `psnr_blocks`.

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
    arrays = {
        'psnr': psnr,
        'ssim': ssim,
        'average_psnr': np.asarray(np.nanmean(psnr), dtype=np.float32),
        'average_ssim': np.asarray(np.nanmean(ssim), dtype=np.float32),
        'neural_trial_idx': neural_trial_idx,
        'neural_bin_idx': neural_bin_idx,
        'trial_split': trial_split,
        'source_files': np.asarray(source_file_rows, dtype=str),
        'view_names': np.asarray(view_names, dtype=str),
    }
    if temporal_blocks:
        for key, blocks in temporal_blocks.items():
            if not blocks:
                continue
            arrays[key] = np.concatenate(blocks, axis=0).astype(np.float32)

    metrics_npz = Path(metrics_npz)
    metrics_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metrics_npz, **arrays)
    return arrays
