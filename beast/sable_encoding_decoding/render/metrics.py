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

    Used to score renders produced at one resolution (e.g. SABLE's 320x320) at another
    (e.g. the 224x224 used by the beast/resnet baselines) so PSNR/SSIM are comparable across
    models. Bilinear resizing is antialiased, matching the PIL-style downsampling the baselines
    apply to their ground-truth frames; use `mode='nearest'` for binary masks.

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


def neural_aligned_npz_path(neural_input_dir: Path, session_id: str) -> Path:
    """Return the aligned neural `.npz` of one session under a neural-data root.

    The root is laid out as `{neural_input_dir}/{session_id}/{session_id}_aligned.npz`, the
    same layout the neural encoding/decoding scripts read via `--neural_input_dir`.

    Args:
        neural_input_dir: neural-data root directory.
        session_id: session/EID name.

    Returns:
        Path of the session's aligned neural `.npz` (not checked for existence).
    """
    return Path(neural_input_dir) / session_id / f'{session_id}_aligned.npz'


def load_neural_split_layout(
    neural_aligned_npz: Path,
    split: str,
) -> tuple[int, int, np.ndarray | None]:
    """Return `(K, T, intervals)` for one split of an aligned neural `.npz`.

    Args:
        neural_aligned_npz: aligned neural file carrying `{split}_spikes` shaped `[K, T, N]`
            and optionally `{split}_intervals` shaped `[K, 2]`.
        split: split name (`'train'`, `'val'`, or `'test'`).

    Returns:
        Trials `K`, bins per trial `T`, and the `[K, 2]` trial intervals in seconds, or
        `None` for the intervals when the file has none.

    Raises:
        FileNotFoundError: if `neural_aligned_npz` does not exist.
        KeyError: if `{split}_spikes` is missing.
        ValueError: if `{split}_spikes` is not rank 3 or the intervals do not have `K` rows.
    """
    neural_aligned_npz = Path(neural_aligned_npz)
    if not neural_aligned_npz.is_file():
        raise FileNotFoundError(f'aligned neural .npz not found: {neural_aligned_npz}')
    spikes_key = f'{split}_spikes'
    intervals_key = f'{split}_intervals'
    with np.load(neural_aligned_npz, allow_pickle=True) as d:
        if spikes_key not in d.files:
            raise KeyError(
                f'{neural_aligned_npz}: missing {spikes_key!r}; got {sorted(d.files)}',
            )
        spikes_shape = tuple(d[spikes_key].shape)
        intervals = (
            np.asarray(d[intervals_key], dtype=np.float64) if intervals_key in d.files else None
        )
    if len(spikes_shape) != 3:
        raise ValueError(
            f'{neural_aligned_npz}: {spikes_key} must be [K, T, N]; got shape {spikes_shape}',
        )
    k_trials, t_bins = int(spikes_shape[0]), int(spikes_shape[1])
    if intervals is not None and intervals.shape != (k_trials, 2):
        raise ValueError(
            f'{neural_aligned_npz}: {intervals_key} must be [K={k_trials}, 2]; got shape '
            f'{intervals.shape}',
        )
    return k_trials, t_bins, intervals


def _view_names_for(num_views: int) -> tuple[str, ...]:
    """Return `('left', 'right')` for two views, else generic `view{idx:02d}` labels."""
    if num_views == 2:
        return ('left', 'right')
    return tuple(f'view{idx:02d}' for idx in range(num_views))


@dataclass
class _NeuralMetricsBlock:
    """Dense `[K, T, V]` PSNR/SSIM for one `(session, split)`, NaN where never scored."""

    source_file: str
    intervals: np.ndarray | None
    psnr: np.ndarray
    ssim: np.ndarray

    @property
    def k_trials(self) -> int:
        """Number of neural trials `K`."""
        return int(self.psnr.shape[0])

    @property
    def t_bins(self) -> int:
        """Number of neural bins per trial `T`."""
        return int(self.psnr.shape[1])

    @property
    def num_views(self) -> int:
        """Number of scored views `V`."""
        return int(self.psnr.shape[2])


class NeuralTrialMetricsAccumulator:
    """Accumulate per-view PSNR/SSIM into dense `[K, T, V]` blocks aligned to neural trials.

    `K` and `T` come from each session's aligned neural `.npz` (see
    `load_neural_split_layout`), so every block lines up index-for-index with that split's
    `{split}_spikes`; `(trial, bin)` cells that are never scored stay NaN. Rows are placed by
    their `(session_id, split, neural_trial_idx, neural_bin_idx)` identity, never by
    dataloader position.
    """

    def __init__(self, neural_input_dir: Path, *, interval_atol: float = 1e-3) -> None:
        """Initialize.

        Args:
            neural_input_dir: neural-data root, see `neural_aligned_npz_path`.
            interval_atol: absolute tolerance (seconds) when checking a row's
                `neural_interval_sec` against the neural file's trial intervals.
        """
        self._neural_input_dir = Path(neural_input_dir)
        self._interval_atol = float(interval_atol)
        self._blocks: dict[tuple[str, str], _NeuralMetricsBlock] = {}

    def _block(self, session_id: str, split: str, num_views: int) -> _NeuralMetricsBlock:
        """Return the block for `(session_id, split)`, allocating it from the neural file."""
        key = (session_id, split)
        block = self._blocks.get(key)
        if block is None:
            npz_path = neural_aligned_npz_path(self._neural_input_dir, session_id)
            k_trials, t_bins, intervals = load_neural_split_layout(npz_path, split)
            block = _NeuralMetricsBlock(
                source_file=str(npz_path),
                intervals=intervals,
                psnr=np.full((k_trials, t_bins, num_views), np.nan, dtype=np.float32),
                ssim=np.full((k_trials, t_bins, num_views), np.nan, dtype=np.float32),
            )
            self._blocks[key] = block
        elif block.num_views != num_views:
            raise ValueError(
                f'session {session_id!r} split {split!r}: got {num_views} views but earlier '
                f'rows had {block.num_views}',
            )
        return block

    def add(
        self,
        *,
        session_id: str,
        split: str,
        neural_trial_idx: int,
        neural_bin_idx: int,
        psnr: np.ndarray,
        ssim: np.ndarray,
        neural_interval_sec: np.ndarray | None = None,
    ) -> None:
        """Record one scored frame's per-view PSNR/SSIM at its neural `(trial, bin)` cell.

        Args:
            session_id: session/EID the frame belongs to.
            split: split label of the frame (`'train'`, `'val'`, or `'test'`).
            neural_trial_idx: trial index local to `split`, i.e. the row of `{split}_spikes`.
            neural_bin_idx: bin index within the trial.
            psnr: per-view PSNR, shape `[V]`.
            ssim: per-view SSIM, shape `[V]`.
            neural_interval_sec: optional `[2]` trial interval carried by the frame, checked
                against the neural file's `{split}_intervals` when both are available.

        Raises:
            ValueError: if the row carries no neural metadata (negative ids), the ids fall
                outside `[K, T]`, the cell was already scored, the view count changed, or the
                interval disagrees with the neural file.
        """
        psnr = np.asarray(psnr, dtype=np.float32).reshape(-1)
        ssim = np.asarray(ssim, dtype=np.float32).reshape(-1)
        if psnr.shape != ssim.shape:
            raise ValueError(f'psnr has {psnr.shape[0]} views but ssim has {ssim.shape[0]}')
        split = str(split).lower()
        trial = int(neural_trial_idx)
        bin_i = int(neural_bin_idx)
        if trial < 0 or bin_i < 0:
            raise ValueError(
                f'session {session_id!r}: frame carries no neural trial/bin metadata '
                f'(neural_trial_idx={trial}, neural_bin_idx={bin_i}); the dataset must use the '
                'eval layout whose frame_index_mapping.json carries the neural fields',
            )
        block = self._block(session_id, split, psnr.shape[0])
        if trial >= block.k_trials or bin_i >= block.t_bins:
            raise ValueError(
                f'session {session_id!r} split {split!r}: (neural_trial_idx={trial}, '
                f'neural_bin_idx={bin_i}) is outside the neural layout K={block.k_trials}, '
                f'T={block.t_bins} from {block.source_file}',
            )
        if not np.all(np.isnan(block.psnr[trial, bin_i])):
            raise ValueError(
                f'session {session_id!r} split {split!r}: (neural_trial_idx={trial}, '
                f'neural_bin_idx={bin_i}) was scored twice',
            )
        if neural_interval_sec is not None and block.intervals is not None:
            got = np.asarray(neural_interval_sec, dtype=np.float64).reshape(-1)
            want = block.intervals[trial]
            if got.shape == want.shape and not np.allclose(
                got, want, atol=self._interval_atol, rtol=0.0,
            ):
                raise ValueError(
                    f'session {session_id!r} split {split!r} neural_trial_idx={trial}: frame '
                    f'interval {got.tolist()} != neural interval {want.tolist()} from '
                    f'{block.source_file}; the frames and neural data are misaligned',
                )
        block.psnr[trial, bin_i] = psnr
        block.ssim[trial, bin_i] = ssim

    def num_scored(self) -> int:
        """Return how many `(session, split, trial, bin)` cells have been scored."""
        return int(sum(
            np.sum(~np.isnan(block.psnr[:, :, 0])) for block in self._blocks.values()
        ))

    def num_unscored(self) -> int:
        """Return how many `(session, split, trial, bin)` cells are still NaN."""
        return int(sum(
            np.sum(np.isnan(block.psnr[:, :, 0])) for block in self._blocks.values()
        ))

    def overall_average(self) -> tuple[float, float]:
        """Return `(mean PSNR, mean SSIM)` over every scored cell and view, NaN if none."""
        if self.num_scored() == 0:
            return float('nan'), float('nan')
        psnr = np.concatenate([block.psnr.reshape(-1) for block in self._blocks.values()])
        ssim = np.concatenate([block.ssim.reshape(-1) for block in self._blocks.values()])
        return float(np.nanmean(psnr)), float(np.nanmean(ssim))

    def save(
        self,
        output_dir: Path,
        *,
        splits_order: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        """Write one `{output_dir}/{session_id}/psnr_ssim_metrics.npz` per session.

        Splits of the same session are stacked along `K` in `splits_order` (splits not listed
        follow in first-seen order); `trial_split` records which split each `K` row came from,
        and `source_files` names the aligned neural `.npz` behind each row.

        Args:
            output_dir: inference output root.
            splits_order: preferred split order along `K`, e.g. the `--splits` argument.

        Returns:
            `{session_id: metrics path}` for every session with at least one block.

        Raises:
            ValueError: if a session's splits disagree on `T` or on the number of views.
        """
        output_dir = Path(output_dir)
        preferred = [str(s).lower() for s in (splits_order or [])]
        by_session: dict[str, dict[str, _NeuralMetricsBlock]] = {}
        for (session_id, split), block in self._blocks.items():
            by_session.setdefault(session_id, {})[split] = block

        saved: dict[str, Path] = {}
        for session_id in sorted(by_session):
            blocks = by_session[session_id]
            ordered = [s for s in preferred if s in blocks]
            ordered += [s for s in blocks if s not in ordered]
            t_bins = {blocks[s].t_bins for s in ordered}
            num_views = {blocks[s].num_views for s in ordered}
            if len(t_bins) != 1 or len(num_views) != 1:
                raise ValueError(
                    f'session {session_id!r}: splits {ordered} disagree on T={sorted(t_bins)} '
                    f'or V={sorted(num_views)}; cannot stack them along K',
                )
            path = output_dir / session_id / 'psnr_ssim_metrics.npz'
            save_psnr_ssim_metrics_npz(
                path,
                psnr_blocks=[blocks[s].psnr for s in ordered],
                ssim_blocks=[blocks[s].ssim for s in ordered],
                neural_trial_blocks=[
                    np.arange(blocks[s].k_trials, dtype=np.int64) for s in ordered
                ],
                neural_bin_blocks=[
                    np.tile(np.arange(blocks[s].t_bins, dtype=np.int64), (blocks[s].k_trials, 1))
                    for s in ordered
                ],
                trial_split_blocks=[np.full(blocks[s].k_trials, s, dtype=str) for s in ordered],
                source_file_rows=[
                    blocks[s].source_file for s in ordered for _ in range(blocks[s].k_trials)
                ],
                view_names=_view_names_for(next(iter(num_views))),
            )
            saved[session_id] = path
        return saved


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
