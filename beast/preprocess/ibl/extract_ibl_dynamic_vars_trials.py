"""Build a beast-compatible ``dynamic_vars_z_trials.npz`` from IBL ``DYNAMIC_VARS``.

``DYNAMIC_VARS`` (wheel speed, licks, whisker motion energy, nose speed, paw speed — all
keypoint-derived IBL behavior traces) are extracted by an external script (see
``https://github.com/yzhang511/beast/blob/behavior_traces_and_pca_latents/beast/extract_neural_data.py``)
and saved as ``train_<var>`` / ``val_<var>`` / ``test_<var>`` arrays directly inside each
``<eid>_aligned.npz`` file, already binned into the same trials as ``train/val/test_spikes``.

Unlike the Cheese3D keypoint case, no CSV loading or frame/trial re-alignment is needed here:
this module only concatenates the ``DYNAMIC_VARS`` arrays along the feature axis and reshapes
them into the ``train/val/test_z_trials_time`` contract that
``beast.sable_encoding_decoding.neural.run_encoding_decoding`` expects for any ``--latent_kind``.

Writes ``<output-dir>/dynamic_vars_z/<eid>/dynamic_vars_z_trials.npz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beast.logging import log_step

# keypoint-derived IBL behavior traces (hyphenated names, as produced by the external
# extract_neural_data.py); the npz keys use underscores instead of hyphens
DYNAMIC_VARS = (
    'wheel-speed',
    'licks',
    'left-whisker-motion-energy',
    'right-whisker-motion-energy',
    'left-nose-speed',
    'right-nose-speed',
    'left-paw-speed',
    'right-paw-speed',
)

_SPLITS = ('train', 'val', 'test')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; `None` uses `sys.argv`.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build dynamic_vars_z_trials.npz from IBL DYNAMIC_VARS behavior traces '
        'already stored inside <eid>_aligned.npz.',
    )
    parser.add_argument('--eid', type=str, required=True, help='session id')
    parser.add_argument(
        '--neural-input-dir',
        type=str,
        required=True,
        help='root dir containing <eid>/<eid>_aligned.npz',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='root dir to write dynamic_vars_z/<eid>/dynamic_vars_z_trials.npz',
    )
    return parser.parse_args(argv)


def _npz_key(split_name: str, dynamic_var: str) -> str:
    """Npz key for one split/variable, e.g. `('train', 'wheel-speed')` -> `'train_wheel_speed'`.

    Args:
        split_name: `'train'`, `'val'`, or `'test'`.
        dynamic_var: one entry of `DYNAMIC_VARS` (hyphenated).

    Returns:
        The corresponding key in `<eid>_aligned.npz`.
    """
    return f"{split_name}_{dynamic_var.replace('-', '_')}"


def build_split_z_trials_time(neural_data_dict, split_name: str) -> np.ndarray:
    """Concatenate one split's `DYNAMIC_VARS` into a `[K, T, 1, D]` array.

    Args:
        neural_data_dict: loaded `<eid>_aligned.npz` (an `np.lib.npyio.NpzFile`).
        split_name: `'train'`, `'val'`, or `'test'`.

    Returns:
        Array of shape `[K, T, 1, D]`, `D = sum` of each `DYNAMIC_VARS` entry's feature dim.

    Raises:
        KeyError: if a `DYNAMIC_VARS` key is missing from `neural_data_dict`.
    """
    missing = [
        var for var in DYNAMIC_VARS if _npz_key(split_name, var) not in neural_data_dict
    ]
    if missing:
        raise KeyError(f'{split_name}: missing DYNAMIC_VARS keys {missing} in aligned npz')
    parts = [
        np.asarray(neural_data_dict[_npz_key(split_name, var)], dtype=np.float32)
        for var in DYNAMIC_VARS
    ]
    combined = np.concatenate(parts, axis=-1)  # [K, T, D]
    return combined[:, :, None, :]  # [K, T, 1, D]


def main(argv: list[str] | None = None) -> None:
    """Build `dynamic_vars_z_trials.npz` for one IBL session's `DYNAMIC_VARS` traces."""
    args = parse_args(argv)

    neural_aligned_npz = Path(args.neural_input_dir) / args.eid / f'{args.eid}_aligned.npz'
    log_step(f'loading {neural_aligned_npz}')
    neural_data_dict = np.load(neural_aligned_npz, allow_pickle=True)

    output_dir = Path(args.output_dir) / 'dynamic_vars_z' / args.eid
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_kwargs: dict[str, np.ndarray] = {}
    split_sizes: dict[str, int] = {}
    neural_trial_idx_parts: list[np.ndarray] = []
    for split_name in _SPLITS:
        z_trials_time = build_split_z_trials_time(neural_data_dict, split_name)
        npz_kwargs[f'{split_name}_z_trials_time'] = z_trials_time
        npz_kwargs[f'{split_name}_intervals'] = np.asarray(
            neural_data_dict[f'{split_name}_intervals'], dtype=np.float64,
        )
        n_split = z_trials_time.shape[0]
        split_sizes[split_name] = n_split
        # local index into this split's own *_intervals, as required by
        # run_encoding_decoding._maybe_assert_latent_intervals_match_neural
        neural_trial_idx_parts.append(np.arange(n_split, dtype=np.int64))
        log_step(f'{split_name}: z_trials_time shape={z_trials_time.shape}')

    npz_kwargs['neural_trial_idx'] = np.concatenate(neural_trial_idx_parts)

    trials_npz_path = output_dir / 'dynamic_vars_z_trials.npz'
    np.savez(trials_npz_path, **npz_kwargs)
    log_step(f'wrote {trials_npz_path}')

    params = {
        'eid': args.eid,
        'neural_aligned_npz': str(neural_aligned_npz),
        'dynamic_vars': list(DYNAMIC_VARS),
        'split_sizes': split_sizes,
    }
    params_path = output_dir / 'params.json'
    with params_path.open('w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    log_step(f'wrote {params_path}')


if __name__ == '__main__':
    main()
