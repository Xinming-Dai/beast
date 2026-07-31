"""Split a combined `frame_z_trials.npz` into one file per split.

`beast.sable_encoding_decoding.img_token.run_pca_and_save`'s `--combined-trials-{train,val,test}
-npz` flags each expect a file containing *only* that split's rows (it does not filter by a
`trial_split` label after loading — whatever rows are in the file are treated as belonging to
the role the flag was passed under). Resnet's `frame_z_trials.npz` (from
`beast.inference.combine_eval_layout_latents`) instead stores all three splits, keyed as
`train_z_trials_time` / `val_z_trials_time` / `test_z_trials_time`, in one file. This script
splits it into three single-split files so each can be passed to its matching
`--combined-trials-*-npz` flag.
"""

import argparse
from pathlib import Path

import numpy as np


def split_trials_by_split(trials_npz: Path, out_dir: Path) -> dict[str, Path]:
    """Write one single-split combined-trials `.npz` per split found in `trials_npz`.

    Args:
        trials_npz: path to a `frame_z_trials.npz` written by `combine_eval_layout_latents`.
        out_dir: directory to write `frame_z_trials_{split}.npz` files into.

    Returns:
        Mapping from split name to the written file path, for splits with a non-empty
        `<split>_z_trials_time` array.

    Raises:
        KeyError: if `trials_npz` is missing `neural_trial_idx` or every `*_z_trials_time` key.
    """
    data = np.load(trials_npz, allow_pickle=True)
    if 'neural_trial_idx' not in data.files:
        raise KeyError(f'{trials_npz}: missing required key "neural_trial_idx"')

    splits = ('train', 'val', 'test')
    if not any(f'{split}_z_trials_time' in data.files for split in splits):
        raise KeyError(
            f'{trials_npz}: expected at least one of '
            f'{[f"{s}_z_trials_time" for s in splits]}; got {sorted(data.files)}',
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    row_start = 0
    for split in splits:
        key = f'{split}_z_trials_time'
        if key not in data.files:
            continue
        n_rows = data[key].shape[0]
        if n_rows == 0:
            row_start += n_rows
            continue

        out_path = out_dir / f'frame_z_trials_{split}.npz'
        save_kw = {
            key: data[key],
            'neural_trial_idx': data['neural_trial_idx'][row_start:row_start + n_rows],
            'trial_split': np.asarray([split] * n_rows, dtype=object),
        }
        # `assembly_from_combined_trials_npz` unconditionally reads train/val/test_intervals
        # (indexed by the per-split-local `neural_trial_idx`), regardless of which
        # `*_z_trials_time` keys are present, so every single-split file needs all three,
        # empty for the splits it doesn't carry.
        for other_split in splits:
            intervals_key = f'{other_split}_intervals'
            if intervals_key in data.files:
                save_kw[intervals_key] = data[intervals_key]
            else:
                save_kw[intervals_key] = np.empty((0, 2), dtype=np.float64)
        np.savez(out_path, **save_kw)
        written[split] = out_path
        print(f'{out_path}: wrote {n_rows} {split} rows.')
        row_start += n_rows

    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the trials-by-split splitter.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, falls back to `sys.argv`.

    Returns:
        Parsed arguments namespace.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--trials-npz', type=Path, required=True, help='combined frame_z_trials.npz to split',
    )
    ap.add_argument(
        '--out-dir', type=Path, required=True, help='directory to write per-split npz files into',
    )
    return ap.parse_args(argv)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    split_trials_by_split(args.trials_npz.resolve(), args.out_dir.resolve())


if __name__ == '__main__':
    main()
