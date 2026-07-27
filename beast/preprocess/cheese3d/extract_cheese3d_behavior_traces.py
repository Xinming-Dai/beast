"""Build a beast-compatible ``behavior_z_trials.npz`` from Cheese3D Lightning Pose keypoints.

Cheese3D analog of IBL's ``DYNAMIC_VARS`` continuous behavior traces (see
``/u/xdai3/project3d/erayzer_cls/E-RayZer-private/scripts/extract_neural_data.py``): a raw
behavior baseline to compare against SABLE-latent encoding/decoding.

Reads ``frame_manifest.json`` (written by ``extract_cheese3d_neural_data.py``) to find, per
trial in every split, the exact raw-video frame already used for that trial's SABLE latent
extraction, then pulls that frame's Lightning Pose keypoints from the left/right (``TL``/``TR``)
CSVs. Using the same representative frame keeps the behavior trace directly comparable to the
video-latent baselines and guarantees the trial rows line up 1:1 with ``<eid>_aligned.npz``
without re-deriving trial windows.

Writes ``<behavior-output-dir>/behavior_z/<eid>/behavior_z_trials.npz`` in the same
``train/val/test_z_trials_time`` contract ``beast.sable_encoding_decoding.neural.
run_encoding_decoding`` expects for any ``--latent_kind``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from beast.logging import log_step
from beast.preprocess.sable.sable_extract_litpose_correspondences import _load_dlc_frame_map

_VIEWS = ('TL', 'TR')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: argument list; ``None`` uses ``sys.argv``.

    Returns:
        parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Build behavior_z_trials.npz from Cheese3D TL/TR Lightning Pose keypoints.',
    )
    parser.add_argument('--frame-manifest', type=str, required=True)
    parser.add_argument('--lp-csv-tl', type=str, required=True)
    parser.add_argument('--lp-csv-tr', type=str, required=True)
    parser.add_argument('--eid', type=str, required=True)
    parser.add_argument('--behavior-output-dir', type=str, required=True)
    parser.add_argument(
        '--min-likelihood',
        type=float,
        default=0.0,
        help='drop bodyparts whose mean likelihood over the sampled trial frames, in either '
             'view, falls below this threshold',
    )
    return parser.parse_args(argv)


def _bodypart_names(bodyparts_flat: list[str]) -> list[str]:
    """Bodypart names in column order, deduplicated from the flat ``x,y,likelihood`` header.

    Args:
        bodyparts_flat: flat header list from ``_load_dlc_frame_map`` (3 columns per bodypart).

    Returns:
        list of bodypart names, one per keypoint, in header order.
    """
    return [bodyparts_flat[i] for i in range(0, len(bodyparts_flat), 3)]


def _row_xyl(row: list[str], start: int) -> tuple[float, float, float]:
    """Read ``(x, y, likelihood)`` for one bodypart from a parsed CSV row.

    Args:
        row: raw CSV row (as returned by ``_load_dlc_frame_map``).
        start: column offset of this bodypart's ``x`` value within ``row[1:]``.

    Returns:
        ``(x, y, likelihood)`` as floats.
    """
    vals = row[1:]
    return float(vals[start]), float(vals[start + 1]), float(vals[start + 2])


def _load_view(csv_path: str) -> tuple[list[str], dict[int, list[str]]]:
    """Load one view's LP CSV into a bodypart-name list and a frame-index row map.

    Args:
        csv_path: path to the DLC-style Lightning Pose CSV.

    Returns:
        Tuple of (bodypart names, ``{frame_idx: row}``).
    """
    bodyparts_flat, frame_map = _load_dlc_frame_map(Path(csv_path))
    return _bodypart_names(bodyparts_flat), frame_map


def _keep_mask(
    manifest_trials: list[dict],
    bodyparts: list[str],
    frame_maps: dict[str, dict[int, list[str]]],
    min_likelihood: float,
) -> np.ndarray:
    """Boolean keep-mask over bodyparts, by mean likelihood across sampled trials, both views.

    Args:
        manifest_trials: every trial entry across all splits (for computing session-wide stats).
        bodyparts: bodypart names, in column order (same order/count in both views).
        frame_maps: per-view ``{frame_idx: row}`` maps.
        min_likelihood: minimum mean likelihood, in either view, to keep a bodypart.

    Returns:
        boolean array ``[n_bodyparts]``, ``True`` for bodyparts to keep.
    """
    sums = {view: np.zeros(len(bodyparts)) for view in _VIEWS}
    for trial in manifest_trials:
        for view in _VIEWS:
            row = frame_maps[view][trial['frame_index'][view]]
            for i in range(len(bodyparts)):
                sums[view][i] += _row_xyl(row, i * 3)[2]
    n = len(manifest_trials)
    keep = np.ones(len(bodyparts), dtype=bool)
    for view in _VIEWS:
        keep &= (sums[view] / n) >= min_likelihood
    for name, kept in zip(bodyparts, keep):
        log_step(f'bodypart {name}: {"keep" if kept else "drop"}')
    return keep


def _trial_features(
    trial: dict,
    bodyparts: list[str],
    keep: np.ndarray,
    frame_maps: dict[str, dict[int, list[str]]],
) -> np.ndarray:
    """Build one trial's ``[V, D]`` feature array from kept bodyparts' ``(x, y)``.

    Args:
        trial: one ``frame_manifest.json`` trial entry.
        bodyparts: bodypart names, in column order.
        keep: boolean keep-mask over ``bodyparts``.
        frame_maps: per-view ``{frame_idx: row}`` maps.

    Returns:
        array of shape ``[len(_VIEWS), 2 * keep.sum()]``.
    """
    out = np.empty((len(_VIEWS), 2 * int(keep.sum())), dtype=np.float32)
    for v, view in enumerate(_VIEWS):
        row = frame_maps[view][trial['frame_index'][view]]
        xy = []
        for i, name in enumerate(bodyparts):
            if not keep[i]:
                continue
            x, y, _ = _row_xyl(row, i * 3)
            xy.extend((x, y))
        out[v] = np.asarray(xy, dtype=np.float32)
    return out


def main(argv: list[str] | None = None) -> None:
    """Build ``behavior_z_trials.npz`` for one Cheese3D session's TL/TR keypoint traces."""
    args = parse_args(argv)

    with open(args.frame_manifest) as f:
        manifest = json.load(f)
    if manifest['eid'] != args.eid:
        raise ValueError(f"manifest eid {manifest['eid']!r} != --eid {args.eid!r}")

    bodyparts_tl, frame_map_tl = _load_view(args.lp_csv_tl)
    bodyparts_tr, frame_map_tr = _load_view(args.lp_csv_tr)
    if bodyparts_tl != bodyparts_tr:
        raise ValueError('TL and TR Lightning Pose CSVs must share the same bodypart set/order')
    bodyparts = bodyparts_tl
    frame_maps = {'TL': frame_map_tl, 'TR': frame_map_tr}

    all_trials = [t for trials in manifest['splits'].values() for t in trials]
    keep = _keep_mask(all_trials, bodyparts, frame_maps, args.min_likelihood)
    n_kept = int(keep.sum())
    if n_kept == 0:
        raise ValueError(f'--min-likelihood={args.min_likelihood} drops all bodyparts.')
    log_step(f'keeping {n_kept}/{len(bodyparts)} bodyparts')

    behavior_output_dir = Path(args.behavior_output_dir) / 'behavior_z' / args.eid
    behavior_output_dir.mkdir(parents=True, exist_ok=True)

    npz_kwargs: dict[str, np.ndarray] = {}
    neural_trial_idx: list[int] = []
    for split_name in ('train', 'val', 'test'):
        trials = manifest['splits'][split_name]
        feats = np.stack(
            [_trial_features(t, bodyparts, keep, frame_maps) for t in trials], axis=0,
        )
        npz_kwargs[f'{split_name}_z_trials_time'] = feats[:, None, :, :]
        npz_kwargs[f'{split_name}_intervals'] = np.asarray(
            [t['neural_interval_sec'] for t in trials], dtype=np.float64,
        )
        neural_trial_idx.extend(t['neural_trial_idx'] for t in trials)

    npz_kwargs['neural_trial_idx'] = np.asarray(neural_trial_idx, dtype=np.int64)

    trials_npz_path = behavior_output_dir / 'behavior_z_trials.npz'
    np.savez(trials_npz_path, **npz_kwargs)
    log_step(f'wrote {trials_npz_path}')

    params = {
        'eid': args.eid,
        'frame_manifest': str(args.frame_manifest),
        'lp_csv_tl': str(args.lp_csv_tl),
        'lp_csv_tr': str(args.lp_csv_tr),
        'min_likelihood': args.min_likelihood,
        'bodyparts_kept': [name for name, k in zip(bodyparts, keep) if k],
        'view_names': list(_VIEWS),
        'split_sizes': {k: len(v) for k, v in manifest['splits'].items()},
    }
    params_path = behavior_output_dir / 'params.json'
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    log_step(f'wrote {params_path}')


if __name__ == '__main__':
    main()
