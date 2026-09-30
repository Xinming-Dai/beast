"""Retrieve neural data and video frames for chosen trials of one or more sessions.

For each EID, this reads `<neural_input_dir>/<eid>/<eid>_aligned.npz`, keeps only the requested
trial rows of one split, and copies the matching frame images into `frames/` using the same
directory structure as the eval frames root. The trimmed npz and `meta.pkl` go into
`neural_data/`. Each copied split folder holds a trimmed `frame_index_mapping.json` that maps
every frame name to its neural trial index and bin index.

Example:
    python -m beast.sable_encoding_decoding.neural.retrieve_neural_data \
        --eid 781b35fd-e1f0-4d14-b2bb-95b7263082bb --split test --trials 0:10,15 \
        --output_dir /path/to/out
"""

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from beast.sable_encoding_decoding.render.metrics import neural_aligned_npz_path

logger = logging.getLogger(__name__)

DEFAULT_FRAMES_ROOT = Path('/work/hdd/bfsr/xdai3/IBL_data/synchronized/extracted_frames')
DEFAULT_NEURAL_INPUT_DIR = DEFAULT_FRAMES_ROOT / 'neural_data'
DEFAULT_EVAL_FRAMES_DIR = DEFAULT_FRAMES_ROOT / 'eval'
CAMERAS = ('left', 'right')
MAPPING_NAME = 'frame_index_mapping.json'


def parse_trial_spec(spec: str) -> list[int]:
    """Parse a trial spec such as `0:5,9` into a sorted list of unique indices.

    `a:b` is an end-exclusive range; bare integers are single trials.

    Args:
        spec: comma-separated ranges and integers.

    Returns:
        Sorted unique trial indices.

    Raises:
        ValueError: if the spec is empty, malformed, or selects no trials.
    """
    idx_trials: set[int] = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            if ':' in part:
                start, stop = part.split(':')
                idx_trials.update(range(int(start), int(stop)))
            else:
                idx_trials.add(int(part))
        except ValueError as exc:
            raise ValueError(f'invalid trial spec element {part!r} in {spec!r}') from exc
    if not idx_trials:
        raise ValueError(f'trial spec {spec!r} selects no trials')
    return sorted(idx_trials)


def trim_aligned_npz(
    npz_path: Path,
    split: str,
    idx_trials: list[int],
    output_path: Path,
) -> dict[str, np.ndarray]:
    """Keep only the selected trial rows of one split and save them.

    Every `{split}_*` array whose leading dimension is the number of trials is sliced. The
    original indices are stored as `{split}_neural_trial_idx`.

    Args:
        npz_path: source `<eid>_aligned.npz`.
        split: split name (train, val, or test).
        idx_trials: trial indices (rows of the split) to keep.
        output_path: destination npz.

    Returns:
        The saved arrays.

    Raises:
        KeyError: if the split is not in the npz.
        IndexError: if any trial index is out of range for the split.
    """
    with np.load(npz_path, allow_pickle=False) as data:
        key_spikes = f'{split}_spikes'
        if key_spikes not in data.files:
            raise KeyError(f'{key_spikes} not found in {npz_path}; keys: {data.files}')
        n_trials = data[key_spikes].shape[0]
        if min(idx_trials) < 0 or max(idx_trials) >= n_trials:
            raise IndexError(
                f'trial indices {idx_trials[0]}..{idx_trials[-1]} out of range for split '
                f'{split!r} with {n_trials} trials in {npz_path}'
            )
        idx = np.asarray(idx_trials, dtype=np.int64)
        arrays = {
            key: data[key][idx]
            for key in data.files
            if key.startswith(f'{split}_') and data[key].shape[:1] == (n_trials,)
        }
    arrays[f'{split}_neural_trial_idx'] = idx
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return arrays


def _load_frame_mapping(
    eval_frames_dir: Path,
    eid: str,
    split: str,
    cam: str,
) -> tuple[Path, dict]:
    """Load one camera's `frame_index_mapping.json` for an EID and split.

    Args:
        eval_frames_dir: eval frames root.
        eid: session UUID.
        split: split name.
        cam: camera name, `left` or `right`.

    Returns:
        The split folder and the parsed mapping (empty dict if the folder is missing).
    """
    pattern = f'{cam}Camera.video/_iblrig_{cam}Camera.downsampled.{eid}/{split}'
    folders = sorted(Path(eval_frames_dir).glob(pattern))
    if not folders or not (folders[0] / MAPPING_NAME).is_file():
        logger.warning(f'no {cam} frame mapping for {eid} split {split} under {eval_frames_dir}')
        return Path(), {}
    with open(folders[0] / MAPPING_NAME) as f:
        return folders[0], json.load(f)


def _check_writable(path: Path, overwrite: bool) -> None:
    """Raise if `path` exists and overwriting was not requested.

    Args:
        path: output file to be written.
        overwrite: whether replacing an existing file is allowed.

    Raises:
        FileExistsError: if the file exists and `overwrite` is False.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path} already exists; pass --overwrite to replace it')


def copy_frames(
    eid: str,
    split: str,
    idx_trials: list[int],
    eval_frames_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, list[int]]:
    """Copy the frames of the selected trials, keeping the eval-frames directory structure.

    Each image is copied to `output_dir/frames/<path relative to eval_frames_dir>`. A trimmed
    `frame_index_mapping.json` holding only the copied entries is written next to the images.
    Missing images and trials without mapping entries are logged as warnings.

    Args:
        eid: session UUID.
        split: split name.
        idx_trials: selected trial indices.
        eval_frames_dir: eval frames root.
        output_dir: session output directory; frames go under `output_dir/frames`.
        overwrite: whether existing files may be replaced.

    Returns:
        Map from frame file name to `[neural trial index, bin index]`, shared by both cameras.

    Raises:
        FileExistsError: if a destination file exists and `overwrite` is False.
    """
    selected = set(idx_trials)
    frame_to_trial_bin: dict[str, list[int]] = {}
    plan: list[tuple[str, Path, dict]] = []
    for cam in CAMERAS:
        folder, mapping = _load_frame_mapping(eval_frames_dir, eid, split, cam)
        entries = {
            name: entry for name, entry in mapping.items() if entry['neural_trial_idx'] in selected
        }
        if entries:
            plan.append((cam, folder, entries))
    n_total = sum(len(entries) for _, _, entries in plan)
    logger.info(f'{eid}: copying {n_total} frame images for {len(idx_trials)} trials')
    log_every = max(1, n_total // 20)

    n_copied = 0
    n_done = 0
    for cam, folder, entries in plan:
        dest_dir = Path(output_dir) / 'frames' / folder.relative_to(eval_frames_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        _check_writable(dest_dir / MAPPING_NAME, overwrite)
        for name, entry in entries.items():
            n_done += 1
            frame_to_trial_bin[name] = [entry['neural_trial_idx'], entry['neural_bin_idx']]
            src = folder / name
            if not src.is_file():
                logger.warning(f'missing frame image {src}')
            else:
                _check_writable(dest_dir / name, overwrite)
                shutil.copy2(src, dest_dir / name)
                n_copied += 1
            if n_done % log_every == 0 or n_done == n_total:
                logger.info(
                    f'{eid}: frames {n_done}/{n_total} ({100 * n_done // n_total}%), '
                    f'current camera: {cam}'
                )
        with open(dest_dir / MAPPING_NAME, 'w') as f:
            json.dump(entries, f, indent=2)

    missing = selected - {idx_trial for idx_trial, _ in frame_to_trial_bin.values()}
    if missing:
        logger.warning(f'{eid}: no frame mapping entries for trials {sorted(missing)}')
    logger.info(f'{eid}: copied {n_copied} frame images')
    return frame_to_trial_bin


def retrieve_session(
    eid: str,
    split: str,
    idx_trials: list[int],
    neural_input_dir: Path,
    eval_frames_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    """Retrieve one session's trimmed neural data and frames.

    The trimmed npz, `meta.pkl`, and `params.json` go under `output_dir/eid/neural_data`, and
    the frames go under `output_dir/eid/frames`.

    Args:
        eid: session UUID.
        split: split name.
        idx_trials: selected trial indices.
        neural_input_dir: neural-data root.
        eval_frames_dir: eval frames root.
        output_dir: output root; files go under `output_dir/eid`.
        overwrite: whether existing outputs may be replaced.

    Raises:
        FileExistsError: if outputs already exist and `overwrite` is False.
    """
    npz_path = neural_aligned_npz_path(neural_input_dir, eid)
    out_dir = Path(output_dir) / eid
    out_neural_dir = out_dir / 'neural_data'
    out_npz = out_neural_dir / f'{eid}_aligned_trimmed.npz'
    _check_writable(out_npz, overwrite)

    logger.info(f'{eid}: trimming neural data from {npz_path}')
    arrays = trim_aligned_npz(npz_path, split, idx_trials, out_npz)
    logger.info(f'{eid}: saved {out_npz} (spikes {arrays[f"{split}_spikes"].shape})')

    # per-unit files are unaffected by trial trimming
    for name in (f'{eid}_meta.pkl', 'params.json'):
        if (npz_path.parent / name).is_file():
            shutil.copy2(npz_path.parent / name, out_neural_dir / name)

    copy_frames(
        eid,
        split,
        idx_trials,
        eval_frames_dir,
        out_dir,
        overwrite=overwrite,
    )
    logger.info(f'{eid}: done, kept {len(idx_trials)} {split} trials -> {out_dir}')


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and retrieve each requested session."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--eid', nargs='+', required=True, help='one or more session UUIDs')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument(
        '--trials',
        required=True,
        help='neural trial indices within the split, e.g. 0:10 (end-exclusive) or 3,5,9',
    )
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--neural_input_dir', type=Path, default=DEFAULT_NEURAL_INPUT_DIR)
    parser.add_argument('--eval_frames_dir', type=Path, default=DEFAULT_EVAL_FRAMES_DIR)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='replace existing outputs instead of raising an error',
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    idx_trials = parse_trial_spec(args.trials)

    time_start = time.perf_counter()
    logger.info(f'started at {datetime.now():%Y-%m-%d %H:%M:%S}')
    try:
        for idx_eid, eid in enumerate(args.eid):
            logger.info(f'[{idx_eid + 1}/{len(args.eid)}] retrieving {eid}')
            retrieve_session(
                eid,
                args.split,
                idx_trials,
                args.neural_input_dir,
                args.eval_frames_dir,
                args.output_dir,
                overwrite=args.overwrite,
            )
    finally:
        # also reported when a session fails, so the time to failure is visible
        elapsed = timedelta(seconds=round(time.perf_counter() - time_start))
        logger.info(f'finished at {datetime.now():%Y-%m-%d %H:%M:%S}, elapsed {elapsed}')


if __name__ == '__main__':
    main()
