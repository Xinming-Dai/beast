"""Shared helpers: load img_tokens from inference `.npz` and align inference dataloader batches."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np


def load_img_tokens_trials_npz(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a combined-format img_tokens `.npz` and return `(z, meta)`.

    The `.npz` is expected to contain per-split arrays written by the PCA/combine pipeline:
    `train_z_trials_time`, `val_z_trials_time`, `test_z_trials_time` each of shape
    `(K_split, T, L, D)`. All present (non-empty) splits are concatenated along axis 0, yielding
    `(K_total, T, L, D)`.

    Args:
        path: path to the trials `.npz`.

    Returns:
        Tuple `(z, meta)`:

        - `z`: float32 array, shape `(K_total, T, L, D)`.
        - `meta`: dict with keys `path`, `keys`, `trial_split` (`list[str]`), and optionally
          `neural_trial_idx` (int64 array) and `meta_json`.

    Raises:
        KeyError: if no non-empty `{split}_z_trials_time` array is present.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        meta_json_raw = d['meta_json'] if 'meta_json' in keys else None

        blocks: list[np.ndarray] = []
        split_labels: list[str] = []
        for split in ('train', 'val', 'test'):
            key = f'{split}_z_trials_time'
            if key not in keys:
                continue
            arr = np.asarray(d[key], dtype=np.float32)
            if arr.shape[0] == 0:
                continue
            blocks.append(arr)
            split_labels.extend([split] * int(arr.shape[0]))

        if not blocks:
            raise KeyError(
                f'{path}: no non-empty train/val/test_z_trials_time arrays; got {sorted(keys)}',
            )

        z = np.concatenate(blocks, axis=0) if len(blocks) > 1 else blocks[0]

        neural_trial_idx: np.ndarray | None = None
        if 'neural_trial_idx' in keys:
            neural_trial_idx = np.asarray(d['neural_trial_idx'], dtype=np.int64)

    meta: dict[str, Any] = {
        'path': str(path),
        'keys': sorted(keys),
        'trial_split': split_labels,
    }
    if neural_trial_idx is not None:
        meta['neural_trial_idx'] = neural_trial_idx
    if meta_json_raw is not None:
        meta['meta_json'] = str(meta_json_raw.tolist())
    return z, meta


# inference / estimated img_token saves use various stems; keep discovery in one place.
_IMG_TOKENS_NPZ_PATTERN = 'img_tokens*.npz'
# sidecars / non-latent npz that still match img_tokens*.npz
_EXCLUDED_IMG_TOKENS_NAME_PREFIXES = ('img_tokens_camera',)


def _is_img_tokens_latent_npz(path: Path) -> bool:
    """True unless `path`'s name is a known camera-sidecar prefix."""
    name = path.name.lower()
    return not any(name.startswith(p) for p in _EXCLUDED_IMG_TOKENS_NAME_PREFIXES)


def sorted_img_tokens_npz_paths(root: Path, *, recursive: bool = True) -> list[Path]:
    """Sorted `.npz` paths whose names match `img_tokens*.npz` under `root`.

    Covers e.g. `img_tokens_batch*.npz`, `img_tokens_estimated_batch*.npz`,
    `img_tokens_estimated_neuraltrial*.npz`. Excludes `img_tokens_camera*.npz` (camera sidecars).

    Args:
        root: directory to search.
        recursive: if `True`, search `root` and all subdirectories; otherwise only immediate
            children of `root`.

    Returns:
        Sorted list of matching paths, or an empty list if `root` is not a directory.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    if recursive:
        cands = sorted(root.glob(f'**/{_IMG_TOKENS_NPZ_PATTERN}'))
    else:
        cands = sorted(root.glob(_IMG_TOKENS_NPZ_PATTERN))
    return [p for p in cands if _is_img_tokens_latent_npz(p)]


def load_z_from_npz_file(path: Path) -> np.ndarray:
    """Load img_tokens array from a single `.npz` path (`z`, `z_trials`, or PCA combine format).

    Args:
        path: path to the `.npz` file.

    Returns:
        Array of shape `[K, T, L, D]` (a leading `[K, L, D]` array is expanded to `T=1`).

    Raises:
        KeyError: if none of `'z'`, `'z_trials'`, or the per-split combine keys are present.
        ValueError: if the loaded array is not rank 3 or rank 4.
    """
    path = Path(path).resolve()
    with np.load(path, allow_pickle=True) as d:
        keys = set(d.files)
        if 'z' in keys:
            z = np.asarray(d['z'], dtype=np.float32)
        elif 'z_trials' in keys:
            z = np.asarray(d['z_trials'], dtype=np.float32)
        elif all(f'{s}_z_trials_time' in keys for s in ('train', 'val', 'test')):
            z, _ = load_img_tokens_trials_npz(path)
        else:
            raise KeyError(
                f"{path}: need 'z', 'z_trials', or train/val/test_z_trials_time; "
                f'got {sorted(keys)}',
            )

    if z.ndim == 3:
        z = z[:, np.newaxis, :, :]
    if z.ndim != 4:
        raise ValueError(f'img_tokens expected [K,T,L,D] (or [K,L,D]); got shape {z.shape}')
    return z


def load_z_array(path: Path) -> np.ndarray:
    """Load an img_tokens array from either a direct `.npz` file or a directory to search.

    Args:
        path: `.npz` file path, or a directory containing `img_tokens*.npz` files (the first
            match, sorted, is used).

    Returns:
        Array of shape `[K, T, L, D]`.

    Raises:
        FileNotFoundError: if `path` is a directory with no matching `.npz` files.
    """
    path = path.resolve()
    if path.is_dir():
        cands = sorted_img_tokens_npz_paths(path, recursive=True)
        if not cands:
            raise FileNotFoundError(
                f'No {_IMG_TOKENS_NPZ_PATTERN} under {path} '
                '(expected names like img_tokens_batch*, img_tokens_estimated_*, …)',
            )
        path = cands[0]

    return load_z_from_npz_file(path)


def iter_dataloader_batch(dataloader: Any, batch_index: int) -> Any:
    """Return the batch at `batch_index` from `dataloader`, iterating from the start.

    Args:
        dataloader: an iterable of batches.
        batch_index: 0-based index of the batch to return.

    Returns:
        The batch at `batch_index`.

    Raises:
        IndexError: if `dataloader` yields fewer than `batch_index + 1` batches.
    """
    for i, batch in enumerate(dataloader):
        if i == batch_index:
            return batch
    raise IndexError(f'Dataloader ended before batch_index={batch_index}')


def apply_dataloader_overrides(config: Any, args: argparse.Namespace) -> None:
    """Apply CLI overrides for dataset path, batch size, workers, and IBL-specific options.

    Args:
        config: mutable training config object with `.training` / `.model` attributes.
        args: parsed CLI namespace; only attributes that are set (not `None`/falsy, per field)
            are applied.
    """
    if args.dataset_path:
        config.training.dataset_path = args.dataset_path
    if getattr(args, 'batch_size', None) is not None:
        config.training.batch_size_per_gpu = args.batch_size
    if getattr(args, 'num_workers', None) is not None:
        config.training.num_workers = args.num_workers
    if getattr(args, 'include_splits', None) is not None:
        config.training.ibl_precache_splits = args.include_splits
    if getattr(args, 'eid', None) is not None:
        config.training.ibl_inference_session_eids = args.eid
    if getattr(args, 'vda_cache_root', None) is not None:
        config.model.setdefault('vda', {})
        config.model['vda']['cache_root'] = args.vda_cache_root
    if getattr(args, 'correspondence_cache_root', None):
        config.model.setdefault('merge_pcd', {})
        config.model['merge_pcd'].setdefault('use_correspondences', {})
        config.model['merge_pcd']['use_correspondences']['cache_root'] = (
            args.correspondence_cache_root
        )
    if getattr(args, 'ibl_precache_valid_index', None) is not None:
        config.training.ibl_precache_valid_index = args.ibl_precache_valid_index
