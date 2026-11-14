"""TCN neural encoding from PCA-compressed patch embeddings, using one cross-validation fold.

Each fold directory holds `features_pca128.npy` (trials x time bins x patches x PCs) and
`pca_model.npz` (with a `fold_id` array marking the trials held out of that fold's PCA fit). The
first `num_pcs` components of every patch are flattened into one feature vector per time bin, the
`fold_id == fold` trials form the test set, and the remaining trials are split into train and val
for the learning-rate search in `train_cnn_encoder_with_tune`.

Runnable as `python -m beast.sable_encoding_decoding.neural.patch_tcn`.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import ray

from beast.sable_encoding_decoding.neural.encoder import train_cnn_encoder_with_tune
from beast.sable_encoding_decoding.neural.run_encoding_decoding import (
    _resolve_ray_session_temp_dir,
)
from beast.sable_encoding_decoding.neural.utils import set_seed

_CHUNK_TRIALS = 20


def load_patch_features(features_path: str | Path, num_pcs: int) -> np.ndarray:
    """Load the first `num_pcs` PCs of every patch and flatten patches and PCs.

    The file is memory-mapped and read in trial chunks, so the full PCA array (about 10 GB for
    128 PCs) never has to fit in memory.

    Args:
        features_path: path to `features_pca128.npy`, shape (trials, time, patches, pcs).
        num_pcs: number of leading PCs to keep along the last axis.

    Returns:
        Float32 array of shape (trials, time, patches * num_pcs).

    Raises:
        ValueError: if the file is not 4D or has fewer than `num_pcs` PCs.
    """
    features = np.load(features_path, mmap_mode='r')
    if features.ndim != 4:
        raise ValueError(f'expected 4D features, got shape {features.shape}')
    n_trials, n_time, n_patches, n_pcs = features.shape
    if not 0 < num_pcs <= n_pcs:
        raise ValueError(f'num_pcs must be in [1, {n_pcs}], got {num_pcs}')
    out = np.empty((n_trials, n_time, n_patches * num_pcs), dtype=np.float32)
    for start in range(0, n_trials, _CHUNK_TRIALS):
        chunk = features[start:start + _CHUNK_TRIALS, ..., :num_pcs]
        out[start:start + _CHUNK_TRIALS] = chunk.reshape(len(chunk), n_time, -1)
    return out


def load_spikes(neural_npz_path: str | Path) -> np.ndarray:
    """Stack the train, val and test spikes in the original neural trial order.

    Args:
        neural_npz_path: path to `<eid>_aligned.npz`.

    Returns:
        Array of shape (trials, time, neurons), rows ordered train, val, test.
    """
    with np.load(neural_npz_path) as data:
        return np.concatenate(
            [data['train_spikes'], data['val_spikes'], data['test_spikes']], axis=0,
        )


def split_trials(
    fold_id: np.ndarray,
    fold: int,
    num_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split trial rows into train, val and test following one cross-validation fold.

    Args:
        fold_id: held-out fold of every trial, shape (trials,).
        fold: fold whose trials form the test set.
        num_val: number of non-test trials assigned to the validation set.
        seed: seed of the permutation that picks the validation trials.

    Returns:
        Sorted row indices `(idx_train, idx_val, idx_test)`.

    Raises:
        ValueError: if `fold` is absent from `fold_id` or `num_val` leaves no training trials.
    """
    idx_test = np.flatnonzero(fold_id == fold)
    if len(idx_test) == 0:
        raise ValueError(f'fold {fold} not found in fold_id (values: {np.unique(fold_id)})')
    idx_rest = np.flatnonzero(fold_id != fold)
    if not 0 < num_val < len(idx_rest):
        raise ValueError(f'num_val must be in [1, {len(idx_rest) - 1}], got {num_val}')
    idx_perm = np.random.default_rng(seed).permutation(idx_rest)
    return np.sort(idx_perm[num_val:]), np.sort(idx_perm[:num_val]), idx_test


def build_data_dict(
    eid: str,
    features: np.ndarray,
    spikes: np.ndarray,
    idx_splits: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Arrange features and spikes into the `[train, val, test]` layout of the CNN encoder.

    Args:
        eid: session id.
        features: array of shape (trials, time, features).
        spikes: array of shape (trials, time, neurons).
        idx_splits: row indices `(idx_train, idx_val, idx_test)`.

    Returns:
        Mapping `eid -> {'X': [train, val, test], 'y': [train, val, test]}`.

    Raises:
        ValueError: if `features` and `spikes` disagree on trials or time bins.
    """
    if features.shape[:2] != spikes.shape[:2]:
        raise ValueError(
            f'features {features.shape[:2]} and spikes {spikes.shape[:2]} must share '
            '(trials, time)',
        )
    return {
        eid: {
            'X': [features[idx] for idx in idx_splits],
            'y': [spikes[idx] for idx in idx_splits],
        },
    }


def run_patch_tcn(
    eid: str,
    neural_input_dir: str | Path,
    pca_dir: str | Path,
    out_dir: str | Path,
    num_pcs: int,
    fold: int,
    num_samples: int,
    num_val: int,
    seed: int,
) -> Path:
    """Tune and evaluate the TCN on one fold of PCA-compressed patch features.

    Args:
        eid: session id.
        neural_input_dir: root of the neural data, holding `<eid>/<eid>_aligned.npz`.
        pca_dir: representation directory holding `fold_<k>/` subdirectories.
        out_dir: output root; results go to `<out_dir>/<eid>/`.
        num_pcs: number of leading PCs kept per patch.
        fold: fold whose held-out trials form the test set.
        num_samples: number of Ray Tune trials.
        num_val: number of validation trials taken from the non-test trials.
        seed: seed for the train/val split and the TCN.

    Returns:
        Path of the saved `.npy` result file.
    """
    fold_dir = Path(pca_dir) / f'fold_{fold}'
    with np.load(fold_dir / 'pca_model.npz') as pca_model:
        fold_id = pca_model['fold_id']
    features = load_patch_features(fold_dir / 'features_pca128.npy', num_pcs)
    spikes = load_spikes(Path(neural_input_dir) / eid / f'{eid}_aligned.npz')
    if len(fold_id) != len(features) or len(spikes) != len(features):
        raise ValueError(
            f'trial counts differ: fold_id {len(fold_id)}, features {len(features)}, '
            f'spikes {len(spikes)}',
        )

    idx_splits = split_trials(fold_id, fold, num_val, seed)
    data_dict = build_data_dict(eid, features, spikes, idx_splits)
    for name, x, y in zip(
        ('Train', 'Val', 'Test'),
        data_dict[eid]['X'],
        data_dict[eid]['y'],
        strict=True,
    ):
        print(f'{name} X shape: {x.shape}, Y shape: {y.shape}')

    session_out_dir = (Path(out_dir) / eid).resolve()
    session_out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    result = train_cnn_encoder_with_tune(
        data_dict,
        num_samples=num_samples,
        tune_storage_path=str(session_out_dir / f'ray_tune_k{num_pcs}_fold{fold}'),
        seed=seed,
    )
    print(f"Patch TCN {eid} fold {fold} Test BPS: {result[eid]['bps']} R2: {result[eid]['r2']}")

    save_path = session_out_dir / f'patch_tcn_k{num_pcs}_fold{fold}.npy'
    meta = {
        'num_pcs': num_pcs,
        'fold': fold,
        'seed': seed,
        'idx_train': idx_splits[0],
        'idx_val': idx_splits[1],
        'idx_test': idx_splits[2],
    }
    np.save(save_path, {eid: {'cnn': result[eid]}, 'meta': meta})
    print(f'Saved results to: {save_path}')
    return save_path


def get_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--eid', required=True, help='session id')
    parser.add_argument('--neural_input_dir', required=True, help='root of the neural data')
    parser.add_argument(
        '--pca_dir',
        required=True,
        help='representation directory holding fold_<k>/ (e.g. .../session_78/full)',
    )
    parser.add_argument('--out_dir', required=True, help='output root (patch_TCN directory)')
    parser.add_argument('--num_pcs', type=int, default=6, help='leading PCs kept per patch')
    parser.add_argument('--fold', type=int, default=0, help='fold used as the test set')
    parser.add_argument('--num_samples', type=int, default=30, help='Ray Tune trials')
    parser.add_argument('--num_val', type=int, default=40, help='validation trials')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the patch TCN pipeline for one session and fold."""
    args = get_args()
    ray_temp = _resolve_ray_session_temp_dir()
    print(f'Ray session temp dir (_temp_dir): {ray_temp}')
    ray.init(num_cpus=2, num_gpus=1, _temp_dir=ray_temp)
    run_patch_tcn(
        eid=args.eid,
        neural_input_dir=args.neural_input_dir,
        pca_dir=args.pca_dir,
        out_dir=args.out_dir,
        num_pcs=args.num_pcs,
        fold=args.fold,
        num_samples=args.num_samples,
        num_val=args.num_val,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
