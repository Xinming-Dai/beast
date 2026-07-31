"""One-off patch: add a `trial_split` key to an existing resnet `frame_z_trials.npz`.

`beast.sable_encoding_decoding.img_token.run_pca_and_save`'s combined-trials-npz loader
requires a `trial_split` label array, which older `frame_z_trials.npz` files written by
`beast.inference.combine_eval_layout_latents` (before that function started saving one) do not
have. The label is fully derivable from the existing per-split `*_z_trials_time` arrays'
row counts, so this avoids re-running `beast predict` / `combine-eval-layout-latents`.
"""

import argparse
from pathlib import Path

import numpy as np


def add_trial_split_key(trials_npz: Path) -> None:
    """Add a `trial_split` key to `trials_npz` in place, derived from split row counts.

    Args:
        trials_npz: path to a `frame_z_trials.npz` written by `combine_eval_layout_latents`.

    Raises:
        KeyError: if `trials_npz` is missing any of `train_z_trials_time`, `val_z_trials_time`,
            or `test_z_trials_time`.
    """
    data = dict(np.load(trials_npz, allow_pickle=True))
    if 'trial_split' in data:
        print(f'{trials_npz}: trial_split already present, nothing to do.')
        return

    splits = ('train', 'val', 'test')
    for split in splits:
        key = f'{split}_z_trials_time'
        if key not in data:
            raise KeyError(f'{trials_npz}: missing required key {key!r}; got {sorted(data)}')

    trial_split = np.asarray(
        [split for split in splits for _ in range(data[f'{split}_z_trials_time'].shape[0])],
        dtype=object,
    )
    data['trial_split'] = trial_split

    np.savez_compressed(trials_npz, **data)
    print(f'{trials_npz}: added trial_split with {len(trial_split)} labels {tuple(splits)}.')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the `trial_split` patch script.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, falls back to `sys.argv`.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--trials-npz',
        type=Path,
        required=True,
        help='path to the frame_z_trials.npz file to patch in place',
    )
    return ap.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    add_trial_split_key(args.trials_npz.resolve())


if __name__ == '__main__':
    main()
