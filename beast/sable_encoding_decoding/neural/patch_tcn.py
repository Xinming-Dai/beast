"""TCN neural encoding from PCA-compressed patch embeddings, using one cross-validation fold.

Each fold directory holds `features_pca128.npy` (trials x time bins x patches x PCs) and
`pca_model.npz` (with a `fold_id` array marking the trials held out of that fold's PCA fit). The
first `num_pcs` components of every patch are flattened into one feature vector per time bin, the
`fold_id == fold` trials form the test set, and the remaining trials are split into train and val
for the hyperparameter search.

Three modes are available. `legacy` searches only the learning rate and evaluates the best
model on the test trials. `tune` searches lr, weight decay and dropout on train/val only and
saves the best config as a JSON file, without touching the test trials. `eval` trains with a
saved config, applied unchanged, and evaluates on the test trials; the config is that JSON file
or a YAML file such as `configs/sable/encoding/patch_tcn_shared.yaml`.

Runnable as `python -m beast.sable_encoding_decoding.neural.patch_tcn`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import ray
import yaml

from beast.sable_encoding_decoding.neural.encoder import (
    train_cnn_encoder,
    train_cnn_encoder_with_tune,
    tune_cnn_encoder,
    validate_tcn_config,
)
from beast.sable_encoding_decoding.neural.run_encoding_decoding import (
    _resolve_ray_session_temp_dir,
)
from beast.sable_encoding_decoding.neural.utils import set_seed

_CHUNK_TRIALS = 20
_MODES = ('legacy', 'tune', 'eval')


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


def save_locked_config(
    config_json: str | Path,
    config: dict[str, Any],
    val_metrics: dict[str, float],
    meta: dict[str, Any],
) -> Path:
    """Write a tuned config to JSON so it can be applied unchanged to other runs.

    Args:
        config_json: output path; parent directories are created.
        config: hyperparameters accepted by `train_cnn_encoder`.
        val_metrics: validation `bps` and `r2` of the config, for the record.
        meta: provenance (session, PCs, fold, seed, ...).

    Returns:
        The path written.
    """
    config_json = Path(config_json)
    config_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {'config': config, 'val_metrics': val_metrics, 'meta': meta}
    config_json.write_text(json.dumps(payload, indent=2) + '\n')
    return config_json


def load_locked_config(config_json: str | Path) -> dict[str, Any]:
    """Read a TCN hyperparameter config from a YAML or JSON file.

    Accepts the file written by `save_locked_config` (hyperparameters under a `config` entry) and
    a flat mapping of hyperparameters, such as `configs/sable/encoding/patch_tcn_shared.yaml`.
    JSON is read as YAML.

    Args:
        config_json: path of the YAML or JSON file.

    Returns:
        Hyperparameters accepted by `train_cnn_encoder`.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a mapping or the hyperparameters are invalid, see
            `validate_tcn_config`.
    """
    config_json = Path(config_json)
    if not config_json.is_file():
        raise FileNotFoundError(f'locked config not found: {config_json}')
    payload = yaml.safe_load(config_json.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f'{config_json} must contain a mapping of hyperparameters')
    config = payload['config'] if 'config' in payload else payload
    try:
        validate_tcn_config(config)
    except ValueError as err:
        raise ValueError(f'invalid config in {config_json}: {err}') from err
    return config


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
    mode: str = 'legacy',
    config_json: str | Path | None = None,
) -> Path:
    """Tune and/or evaluate the TCN on one fold of PCA-compressed patch features.

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
        mode: `legacy` (lr search, then test evaluation), `tune` (lr/wd/dropout search on
            train/val only, writes `config_json`) or `eval` (train with the config in
            `config_json` and evaluate on test).
        config_json: output path in `tune` mode, input path in `eval` mode.

    Returns:
        Path of the saved `.npy` result file, or of `config_json` in `tune` mode.

    Raises:
        ValueError: if `mode` is unknown, or `config_json` is missing in `tune`/`eval` mode.
    """
    if mode not in _MODES:
        raise ValueError(f'mode must be one of {_MODES}, got {mode!r}')
    if mode != 'legacy' and config_json is None:
        raise ValueError(f'config_json is required in {mode!r} mode')
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
    config = None
    if mode == 'tune':
        config, val_metrics = tune_cnn_encoder(
            data_dict,
            num_samples=num_samples,
            tune_storage_path=str(session_out_dir / f'ray_tune_k{num_pcs}_fold{fold}'),
            seed=seed,
            preset='reg',
        )
        meta = {
            'eid': eid,
            'num_pcs': num_pcs,
            'fold': fold,
            'seed': seed,
            'num_val': num_val,
            'num_samples': num_samples,
        }
        save_path = save_locked_config(config_json, config, val_metrics, meta)
        print(f'Saved locked config to: {save_path}')
        return save_path
    if mode == 'eval':
        config = load_locked_config(config_json)
        print(f'Evaluating with locked config: {config}')
        # drop the validation split, the locked config is trained on train only
        data_dict_eval = {
            eid: {
                'X': [data_dict[eid]['X'][0], data_dict[eid]['X'][2]],
                'y': [data_dict[eid]['y'][0], data_dict[eid]['y'][2]],
            },
        }
        result = train_cnn_encoder(config, data_dict_eval, seed=seed)
    else:
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
        'mode': mode,
        'config': config,
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
    parser.add_argument(
        '--mode',
        choices=_MODES,
        default='legacy',
        help='legacy: lr search + test; tune: lr/wd/dropout search, no test; eval: locked config',
    )
    parser.add_argument(
        '--config',
        '--config_json',
        dest='config_json',
        default=None,
        help='TCN config file (YAML or JSON): written (JSON) in tune mode, read in eval mode',
    )
    args = parser.parse_args()
    if args.mode != 'legacy' and args.config_json is None:
        parser.error(f'--config is required with --mode {args.mode}')
    return args


def main() -> None:
    """Run the patch TCN pipeline for one session and fold."""
    args = get_args()
    if args.mode != 'eval':
        # only Ray Tune searches need Ray; the locked-config evaluation trains directly
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
        mode=args.mode,
        config_json=args.config_json,
    )


if __name__ == '__main__':
    main()
