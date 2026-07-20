"""Resumable decoding: per-token metrics shards, combined metrics, render completion markers."""

import argparse
import re
from pathlib import Path

import numpy as np

from beast.sable_encoding_decoding.render.metrics import save_psnr_ssim_metrics_npz

_NEURAL_TRIAL_STEM_RE = re.compile(r'neuraltrial(\d+)', re.IGNORECASE)


def parse_neural_trial_index_arg(expr: str) -> frozenset[int]:
    """Parse a comma-separated `--neural-trial-index` argument into a set of ints.

    Args:
        expr: comma-separated integer list, e.g. `'0,4,10'`.

    Returns:
        Frozenset of parsed neural trial ids.

    Raises:
        argparse.ArgumentTypeError: if `expr` is blank or contains a non-integer token.
    """
    parts = [x.strip() for x in expr.split(',') if x.strip()]
    if not parts:
        raise argparse.ArgumentTypeError('--neural-trial-index must list at least one integer')
    try:
        return frozenset(int(x) for x in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'invalid integer in --neural-trial-index: {expr!r}',
        ) from exc


def parse_neural_trial_range_arg(expr: str) -> tuple[int, int]:
    """Parse a `--neural-trial-range` argument into an inclusive `(lo, hi)` range.

    Args:
        expr: either one integer `'N'` (meaning `0..N`) or two comma-separated integers
            `'LO,HI'`.

    Returns:
        Tuple `(lo, hi)`, inclusive.

    Raises:
        argparse.ArgumentTypeError: if `expr` has the wrong number of tokens, tokens are not
            integers, or `lo > hi`.
    """
    parts = [x.strip() for x in expr.split(',') if x.strip()]
    try:
        if len(parts) == 1:
            return 0, int(parts[0])
        if len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
            if lo > hi:
                raise argparse.ArgumentTypeError(
                    f'--neural-trial-range expects lo<=hi; got {lo},{hi}',
                )
            return lo, hi
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'invalid integer(s) in --neural-trial-range: {expr!r}',
        ) from exc
    raise argparse.ArgumentTypeError(
        '--neural-trial-range expects one number (0..n) or two (lo,hi inclusive)',
    )


def read_scalar_neural_trial_id_from_token_npz(path: Path) -> int:
    """Read the single `neural_trial_idx` scalar stored in a token `.npz` file.

    Args:
        path: path to a `img_tokens*.npz` file expected to carry one trial id.

    Returns:
        The scalar neural trial id.

    Raises:
        KeyError: if `neural_trial_idx` is missing from the file.
        ValueError: if `neural_trial_idx` is not a single unique value, or the filename's
            `neuraltrialXXXX` id disagrees with the stored value.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=True) as d:
        if 'neural_trial_idx' not in d.files:
            raise KeyError(f'{path}: missing neural_trial_idx; got {d.files}')
        arr = np.asarray(d['neural_trial_idx'], dtype=np.int64).reshape(-1)
    unq = np.unique(arr)
    if len(unq) != 1:
        raise ValueError(
            f'{path}: neural_trial_idx must be a single trial id when filtering; unique={unq}',
        )
    tid = int(unq[0])
    m = _NEURAL_TRIAL_STEM_RE.search(path.stem)
    if m is not None:
        stem_id = int(m.group(1))
        if stem_id != tid:
            raise ValueError(
                f'{path}: filename neuraltrial id {stem_id} != neural_trial_idx {tid}',
            )
    return tid


def filter_img_tokens_npz_paths_by_neural_trial(
    paths: list[Path],
    *,
    allowed_indices: frozenset[int] | None = None,
    inclusive_range: tuple[int, int] | None = None,
) -> list[Path]:
    """Filter token `.npz` paths by their scalar `neural_trial_idx`.

    Args:
        paths: candidate `.npz` paths, each expected to carry one trial id.
        allowed_indices: explicit set of allowed ids; mutually exclusive with
            `inclusive_range`.
        inclusive_range: inclusive `(lo, hi)` range of allowed ids; mutually exclusive with
            `allowed_indices`.

    Returns:
        Subset of `paths` whose neural trial id matches the filter.

    Raises:
        ValueError: if neither or both of `allowed_indices`/`inclusive_range` are given, or if
            no path matches the filter.
    """
    if (allowed_indices is None) == (inclusive_range is None):
        raise ValueError('exactly one of allowed_indices and inclusive_range must be set')
    lo, hi = (0, 0)
    if inclusive_range is not None:
        lo, hi = inclusive_range
    out: list[Path] = []
    for p in paths:
        tid = read_scalar_neural_trial_id_from_token_npz(p)
        if allowed_indices is not None:
            if tid not in allowed_indices:
                continue
        else:
            if tid < lo or tid > hi:
                continue
        out.append(p)
    if not out:
        if allowed_indices is not None:
            crit = f'neural_trial_idx in {sorted(allowed_indices)}'
        else:
            crit = f'neural_trial_idx in [{lo}, {hi}] inclusive'
        raise ValueError(f'No token .npz matched {crit} among {len(paths)} path(s)')
    return out


METRICS_SHARDS_SUBDIR = 'metrics_shards'


def metrics_shards_dir(out_dir: Path) -> Path:
    """Return the `metrics_shards/` subdirectory path under `out_dir`."""
    return Path(out_dir).resolve() / METRICS_SHARDS_SUBDIR


def metrics_shard_path(out_dir: Path, source_npz: Path) -> Path:
    """One metrics `.npz` path per token latent file (same stem as `source_npz`)."""
    return metrics_shards_dir(out_dir) / f'{Path(source_npz).stem}_psnr_ssim.npz'


def save_single_token_metrics_npz(
    shard_path: Path,
    *,
    psnr_block: np.ndarray,
    ssim_block: np.ndarray,
    neural_trial_idx: np.ndarray,
    neural_bin_idx: np.ndarray,
    trial_split: np.ndarray,
    source_files: list[str],
    view_names: tuple[str, ...] = ('left', 'right'),
) -> dict[str, np.ndarray]:
    """Save one token file's metrics using the same schema as the combined NPZ."""
    return save_psnr_ssim_metrics_npz(
        shard_path,
        psnr_blocks=[psnr_block],
        ssim_blocks=[ssim_block],
        neural_trial_blocks=[neural_trial_idx],
        neural_bin_blocks=[neural_bin_idx],
        trial_split_blocks=[trial_split],
        source_file_rows=list(source_files),
        view_names=view_names,
    )


def _load_one_metrics_shard(
    shard_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], tuple[str, ...]]:
    """Load one metrics shard `.npz` into its component arrays."""
    with np.load(shard_path, allow_pickle=True) as d:
        psnr = np.asarray(d['psnr'], dtype=np.float32)
        ssim = np.asarray(d['ssim'], dtype=np.float32)
        neural_trial_idx = np.asarray(d['neural_trial_idx'], dtype=np.int64)
        neural_bin_idx = np.asarray(d['neural_bin_idx'], dtype=np.int64)
        trial_split = np.asarray(d['trial_split'], dtype=str)
        source_files = [str(x) for x in np.asarray(d['source_files'], dtype=object).reshape(-1)]
        view_names = tuple(str(x) for x in np.asarray(d['view_names'], dtype=str).reshape(-1))
    return psnr, ssim, neural_trial_idx, neural_bin_idx, trial_split, source_files, view_names


def gather_metrics_shard_blocks(
    out_dir: Path,
    npz_paths: list[Path],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    tuple[str, ...] | None,
    list[int],
]:
    """Load per-file shards in `npz_paths` order.

    Returns:
        Block lists, `view_names`, and missing indices (see the tuple layout above).
    """
    out_dir = Path(out_dir).resolve()
    psnr_blocks: list[np.ndarray] = []
    ssim_blocks: list[np.ndarray] = []
    neural_trial_blocks: list[np.ndarray] = []
    neural_bin_blocks: list[np.ndarray] = []
    trial_split_blocks: list[np.ndarray] = []
    source_file_rows: list[str] = []
    view_names: tuple[str, ...] | None = None
    missing: list[int] = []

    for i, src in enumerate(npz_paths):
        shard = metrics_shard_path(out_dir, src)
        if not shard.is_file():
            missing.append(i)
            continue
        (
            psnr,
            ssim,
            nt,
            nb,
            ts,
            sf,
            vn,
        ) = _load_one_metrics_shard(shard)
        if view_names is None:
            view_names = vn
        elif vn != view_names:
            raise ValueError(f'view_names mismatch: {shard} has {vn}, expected {view_names}')
        psnr_blocks.append(psnr)
        ssim_blocks.append(ssim)
        neural_trial_blocks.append(nt)
        neural_bin_blocks.append(nb)
        trial_split_blocks.append(ts)
        source_file_rows.extend(sf)

    return (
        psnr_blocks,
        ssim_blocks,
        neural_trial_blocks,
        neural_bin_blocks,
        trial_split_blocks,
        source_file_rows,
        view_names if view_names is not None else ('left', 'right'),
        missing,
    )


def combine_metrics_shards_to_combined_npz(
    out_dir: Path,
    npz_paths: list[Path],
    metrics_npz: Path,
    *,
    allow_missing: bool = True,
) -> tuple[dict[str, np.ndarray] | None, list[int]]:
    """Merge per-token shards into `metrics_npz`.

    Returns:
        Tuple `(arrays dict or None, missing indices)`.

    Raises:
        FileNotFoundError: if shards are missing and `allow_missing` is `False`.
    """
    (
        psnr_blocks,
        ssim_blocks,
        neural_trial_blocks,
        neural_bin_blocks,
        trial_split_blocks,
        source_file_rows,
        view_names,
        missing,
    ) = gather_metrics_shard_blocks(out_dir, npz_paths)

    if missing and not allow_missing:
        raise FileNotFoundError(
            f'Missing metrics shard(s) for {len(missing)} source file(s): '
            f"indices {missing[:20]}{'...' if len(missing) > 20 else ''}",
        )

    if not psnr_blocks:
        return None, missing

    merged = save_psnr_ssim_metrics_npz(
        metrics_npz,
        psnr_blocks=psnr_blocks,
        ssim_blocks=ssim_blocks,
        neural_trial_blocks=neural_trial_blocks,
        neural_bin_blocks=neural_bin_blocks,
        trial_split_blocks=trial_split_blocks,
        source_file_rows=source_file_rows,
        view_names=view_names,
    )
    return merged, missing


def delete_metrics_shards_for_sources(out_dir: Path, npz_paths: list[Path]) -> int:
    """Remove shard files tied to `npz_paths` under `metrics_shards/`.

    Returns:
        Count of shard files removed.
    """
    root = metrics_shards_dir(out_dir)
    removed = 0
    for src in npz_paths:
        shard = metrics_shard_path(out_dir, src)
        if shard.is_file():
            shard.unlink()
            removed += 1
    if root.is_dir():
        try:
            if not any(root.iterdir()):
                root.rmdir()
        except OSError:
            pass
    return removed


def render_done_marker_path(out_dir: Path, batch_idx: int) -> Path:
    """Return the render-completion marker path for one batch."""
    return Path(out_dir).resolve() / f'batch_{batch_idx:04d}' / '.decode_saved_img_tokens_complete'


def is_render_complete(marker_path: Path) -> bool:
    """Return whether the render-completion marker file exists."""
    return Path(marker_path).is_file()


def write_render_done_marker(marker_path: Path, source_npz: Path) -> None:
    """Write a render-completion marker recording the resolved source `.npz` path."""
    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f'{Path(source_npz).resolve()}\n', encoding='utf-8')
