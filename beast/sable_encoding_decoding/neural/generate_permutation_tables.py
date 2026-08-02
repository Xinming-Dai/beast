"""Generate frame-permutation index tables for the SABLE encoding null baseline.

For a chosen `--latent_kind`, reads each session's `<latent_kind>_trials.npz` (written by
`beast predict --extract-latents`, see `run_encoding_decoding.LATENT_KIND_LAYOUT`) only to
determine each split's `[N_trials, T_time_bins]` shape, then writes a small permutation
table under `--output_dir` with the same `<subdir>/<eid>/` layout as the real latents. The
tables are applied on the fly by `run_encoding_decoding.py`'s `--permutation_dir` (see
`_apply_frame_permutation`), so the (potentially large) latent tensors themselves are never
duplicated on disk.

Runnable as `python -m beast.sable_encoding_decoding.neural.generate_permutation_tables`.
"""

import argparse
from pathlib import Path

import numpy as np

from beast.sable_encoding_decoding.neural.utils import LATENT_KIND_LAYOUT, parse_latent_kind

_SPLITS = ('train', 'val', 'test')


def get_generate_permutation_tables_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the permutation-table generator.

    Args:
        argv: argument list to parse, e.g. `sys.argv[1:]`. If `None`, `argparse` falls back
            to reading `sys.argv` itself.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Generate frame-permutation index tables for the SABLE encoding null '
        'baseline (Frame permutation baseline).',
    )
    parser.add_argument(
        '--latent_root', type=str, required=True, help='Root latent directory (step1 OUTPUT_DIR)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Root output directory for permutation tables',
    )
    parser.add_argument(
        '--latent_kind',
        type=parse_latent_kind,
        required=True,
        help='Latent layout under --latent_root whose trial*time frame counts the '
        'permutation tables are built from (matches --latent_kind in run_encoding_decoding.py).',
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='Base random seed, reused per session',
    )
    parser.add_argument(
        '--eids',
        type=str,
        nargs='*',
        default=None,
        help='Session ids to process; default: every session found under '
        '<latent_root>/<subdir>/',
    )
    return parser.parse_args(argv)


def _discover_eids(latent_kind_dir: Path, fname: str) -> list[str]:
    """List session ids with a `<fname>` trials file directly under `latent_kind_dir`.

    `latent_kind_dir` (e.g. `<latent_root>/frame_z`) can also contain stray Ray Tune
    experiment directories (e.g. `train_cnn_encoder_2026-07-29_08-28-49`) left behind by
    `run_encoding_decoding.py` runs that omitted `--tune_storage_path`, which defaults to
    this same directory. Filtering on `fname` excludes those.

    Args:
        latent_kind_dir: `<latent_root>/<subdir>`.
        fname: trials npz filename expected inside each real session directory.

    Returns:
        Sorted session ids (subdirectory names containing `fname`).
    """
    return sorted(
        p.name for p in latent_kind_dir.iterdir() if p.is_dir() and (p / fname).exists()
    )


def _build_session_permutation_tables(
    trials_npz_path: Path, seed: int,
) -> dict[str, np.ndarray]:
    """Build one permutation per split for a session's trials npz.

    Args:
        trials_npz_path: path to `<latent_kind>_trials.npz`.
        seed: base seed; re-applied fresh for this session so every session gets the same
            seed, while `train`/`val`/`test` still get independent permutations because the
            RNG advances between the three `rng.permutation(...)` calls below.

    Returns:
        Dict with `perm_train`, `perm_val`, `perm_test` (`int64` arrays).

    Raises:
        KeyError: if a required `<split>_z_trials_time` key is missing from the npz.
    """
    data = np.load(trials_npz_path, allow_pickle=True)
    rng = np.random.default_rng(seed)
    tables = {}
    for split in _SPLITS:
        key = f'{split}_z_trials_time'
        if key not in data.files:
            raise KeyError(f'{trials_npz_path}: missing required key {key!r}')
        n, t = data[key].shape[:2]
        tables[f'perm_{split}'] = rng.permutation(n * t).astype(np.int64)
    return tables


def main() -> None:
    """Generate and write frame-permutation tables for every requested session."""
    args = get_generate_permutation_tables_args()
    subdir, fname = LATENT_KIND_LAYOUT[args.latent_kind]
    latent_kind_dir = Path(args.latent_root) / subdir

    eids = args.eids if args.eids else _discover_eids(latent_kind_dir, fname)
    print(
        f'Generating permutation tables for {len(eids)} session(s), '
        f'latent_kind={args.latent_kind}',
    )

    for eid in eids:
        trials_npz_path = latent_kind_dir / eid / fname
        print(f'[{eid}] reading shapes from {trials_npz_path}')
        tables = _build_session_permutation_tables(trials_npz_path, args.seed)

        output_path = Path(args.output_dir) / subdir / eid / 'permutation.npz'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            seed=args.seed,
            latent_kind=args.latent_kind,
            source_trials_npz=str(trials_npz_path),
            **tables,
        )
        print(f'[{eid}] wrote {output_path}')


if __name__ == '__main__':
    main()
