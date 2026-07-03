#!/usr/bin/env python3
"""Build LitPose keypoint correspondence bundles for Sable training.

Reads raw LitPose DLC-style CSVs (left camera 256×320, right camera 320×256) and writes
one ``.npz`` bundle per frame pair under
``{output_root}/litpose_correspondences/processed_correspondences/{session_id}/``
``correspondences{pair_idx:0{n_digits}d}.npz``.

Session discovery scans the extracted-frames ``input_dir`` from the extraction pipeline
config. Pass ``--config configs/multiview/extraction_pipeline_sable.yaml`` to supply
most parameters without repeating them on the command line.

With ``--no-left-frames-stretched`` (the default), left-camera coordinates are stored in raw
256×320 pixel space. ``SABLEDataset`` rescales them to the model's ``image_size`` at load time.
With ``--left-frames-stretched``, the in-memory stretch from 256×320 → 320×256 is applied
before saving (backward-compatible mode).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# IBL rig left-camera geometry: raw CSV is 256 wide × 320 tall; stretched to 320×256.
_LEFT_CSV_SRC_W, _LEFT_CSV_SRC_H = 256, 320
_LEFT_CSV_DST_W, _LEFT_CSV_DST_H = 320, 256

_UUID_RE = re.compile(
    r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_xy_pair(text: str) -> tuple[float, float]:
    """Argparse helper: ``'5,0'`` → two floats."""
    t = text.strip()
    if ',' not in t:
        raise argparse.ArgumentTypeError(f'expected X,Y, got {text!r}')
    a, b = t.split(',', 1)
    try:
        return float(a.strip()), float(b.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'invalid float pair: {text!r}') from exc


def _eid_from_csv_name(csv_path: Path) -> str | None:
    m = _UUID_RE.search(csv_path.name)
    return m.group(1) if m else None


def _litpose_csv_paths(csv_dir: Path, session_id: str) -> tuple[Path, Path]:
    left = csv_dir / f'_iblrig_leftCamera.downsampled.{session_id}.csv'
    right = csv_dir / f'_iblrig_rightCamera.downsampled.{session_id}.csv'
    return left, right


def _stretch_left_xy(x: float, y: float) -> tuple[float, float]:
    """Stretch raw 256×320 left coords to 320×256."""
    return (x * _LEFT_CSV_DST_W / _LEFT_CSV_SRC_W, y * _LEFT_CSV_DST_H / _LEFT_CSV_SRC_H)


def _report_progress(i: int, total: int, *, label: str, every: int) -> None:
    if every > 0 and total > 0 and (i % every == 0 or i == total):
        print(f'[progress] {label}: {i}/{total}', flush=True)


def _pair_progress(i: int, total: int) -> None:
    if total <= 0:
        return
    prev = 100 * (i - 1) // total
    cur = 100 * i // total
    for milestone in range(10, 101, 10):
        if prev < milestone <= cur:
            print(f'[progress] {milestone}% ({i}/{total} pairs)', flush=True)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _is_dlc_style(rows: list[list[str]], min_rows: int = 4) -> bool:
    if len(rows) < min_rows:
        return False
    if rows[1][0].lower() != 'bodyparts':
        return False
    if rows[2][0].lower() != 'coords':
        return False
    return True


def _transform_row_stretch(row_vals: list[str], bodyparts_flat: list[str]) -> list[str]:
    """Stretch all (x,y) pairs in a left-camera CSV row from 256×320 → 320×256."""
    out: list[str] = []
    for i in range(0, len(bodyparts_flat), 3):
        xs, ys, lh = row_vals[i], row_vals[i + 1], row_vals[i + 2]
        try:
            x, y = float(xs), float(ys)
        except ValueError:
            out.extend([xs, ys, lh])
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            out.extend([xs, ys, lh])
            continue
        nx, ny = _stretch_left_xy(x, y)
        out.extend([f'{nx:.12f}', f'{ny:.12f}', lh])
    return out


def _load_dlc_frame_map(
    csv_path: Path,
    *,
    stretch: bool = False,
    progress_label: str | None = None,
) -> tuple[list[str], dict[int, list[str]]]:
    """Parse a DLC-style LitPose CSV into a frame-index → row map.

    Args:
        csv_path: path to the CSV file.
        stretch: if True, apply left-camera stretch (256×320 → 320×256) in one pass.
        progress_label: optional label for progress reporting every 50 000 rows.

    Returns:
        tuple of (bodyparts_flat header list, {frame_idx: row_values}).
    """
    try:
        size_mb = csv_path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 0.0
    print(f'[info] CSV: reading {csv_path.name} (~{size_mb:.1f} MiB)', flush=True)
    t0 = time.perf_counter()
    raw = csv_path.read_text(encoding='utf-8')
    rows = list(csv.reader(raw.splitlines()))
    print(
        f'[info] CSV: parsed {len(rows)} rows in {time.perf_counter() - t0:.1f}s',
        flush=True,
    )
    if not _is_dlc_style(rows):
        raise ValueError(f'not a DLC-style CSV: {csv_path}')
    bodyparts_flat = rows[1][1:]
    if len(bodyparts_flat) % 3 != 0:
        raise ValueError(f'unexpected bodyparts header in {csv_path}')
    expected = len(bodyparts_flat)
    frame_map: dict[int, list[str]] = {}
    total = max(0, len(rows) - 3)
    for i, r in enumerate(rows[3:], start=1):
        if not r:
            continue
        try:
            frame_idx = int(float(str(r[0]).strip()))
        except ValueError:
            continue
        if stretch:
            vals = r[1:1 + expected]
            if len(vals) >= expected:
                r = [r[0], *_transform_row_stretch(vals, bodyparts_flat), *r[1 + expected:]]
        frame_map[frame_idx] = r
        if progress_label:
            _report_progress(i, total, label=progress_label, every=50000)
    return bodyparts_flat, frame_map


# ---------------------------------------------------------------------------
# Keypoint extraction
# ---------------------------------------------------------------------------

def _keypoint_starts(bodyparts_flat: list[str], names: list[str]) -> dict[str, int]:
    wanted = set(names)
    out: dict[str, int] = {}
    for i in range(0, len(bodyparts_flat), 3):
        name = bodyparts_flat[i]
        if name in wanted and name not in out:
            out[name] = i
    return out


def _read_xlh(vals: list[str], start: int) -> tuple[float, float, float] | None:
    if start + 2 >= len(vals):
        return None
    try:
        x, y, lh = float(vals[start]), float(vals[start + 1]), float(vals[start + 2])
    except ValueError:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(lh)):
        return None
    return x, y, lh


def _build_keypoint_arrays(
    *,
    bodyparts_flat: list[str],
    left_vals: list[str],
    right_vals: list[str],
    keypoint_names: list[str],
    keypoint_starts: dict[str, int],
    min_likelihood: float,
    shift_nose_left: tuple[float, float],
    shift_nose_right: tuple[float, float],
    left_already_stretched: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """Extract aligned (left_xy, right_xy, confidence, labels) from one pair of CSV rows.

    Args:
        bodyparts_flat: flat list of bodypart column headers from the CSV.
        left_vals: raw CSV row for the left frame.
        right_vals: raw CSV row for the right frame.
        keypoint_names: ordered list of keypoints to extract.
        keypoint_starts: mapping from keypoint name to column start index.
        min_likelihood: minimum likelihood threshold; rows below are skipped.
        shift_nose_left: (dx, dy) pixel shift baked into the nose keypoint for left camera.
        shift_nose_right: (dx, dy) pixel shift baked into the nose keypoint for right camera.
        left_already_stretched: if True, left (x,y) are already in stretched 320×256 space.
            If False, they are in raw 256×320 space and the stretch is NOT applied here
            (``--no-left-frames-stretched`` mode: coordinates stay raw for SABLEDataset to
            rescale at load time).

    Returns:
        tuple of (left_xy, right_xy, confidence, labels) or None when no valid keypoints.
    """
    exp = len(bodyparts_flat)
    lv = left_vals[1:1 + exp]
    rv = right_vals[1:1 + exp]
    if len(lv) < exp or len(rv) < exp:
        return None

    left_list: list[tuple[float, float]] = []
    right_list: list[tuple[float, float]] = []
    conf_list: list[float] = []
    labels: list[str] = []

    for name in keypoint_names:
        start = keypoint_starts.get(name)
        if start is None:
            continue
        lt = _read_xlh(lv, start)
        rt = _read_xlh(rv, start)
        if lt is None or rt is None:
            continue
        xl, yl, lhl = lt
        xr, yr, lhr = rt
        if lhl < min_likelihood or lhr < min_likelihood:
            continue
        if left_already_stretched:
            left_list.append((xl, yl))
        else:
            # --no-left-frames-stretched: keep in raw 256×320 space
            left_list.append((xl, yl))
        right_list.append((xr, yr))
        conf_list.append(min(lhl, lhr))
        labels.append(name)

    if not left_list:
        return None

    left_xy = np.asarray(left_list, dtype=np.float32)
    right_xy = np.asarray(right_list, dtype=np.float32)
    dx_l, dy_l = shift_nose_left
    dx_r, dy_r = shift_nose_right
    if (dx_l != 0.0 or dy_l != 0.0 or dx_r != 0.0 or dy_r != 0.0) and 'nose' in labels:
        for i, lab in enumerate(labels):
            if lab == 'nose':
                left_xy[i, 0] += dx_l
                left_xy[i, 1] += dy_l
                right_xy[i, 0] += dx_r
                right_xy[i, 1] += dy_r
                break

    return left_xy, right_xy, np.asarray(conf_list, dtype=np.float32), labels


# ---------------------------------------------------------------------------
# Bundle I/O
# ---------------------------------------------------------------------------

def _output_bundle_path(
    output_root: Path,
    *,
    session_id: str,
    pair_idx: int,
    n_digits: int,
) -> Path:
    fname = f'correspondences{pair_idx:0{n_digits}d}.npz'
    return (
        output_root / 'litpose_correspondences' / 'processed_correspondences' / session_id / fname
    )


def _save_correspondence_bundle(
    output_path: Path,
    *,
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    confidence: np.ndarray,
    labels: list[str],
    metadata: dict[str, Any],
) -> None:
    """Save a correspondence bundle as a .npz file.

    Args:
        output_path: path to write the .npz file.
        left_xy: [N, 2] float32 left-camera pixel coordinates.
        right_xy: [N, 2] float32 right-camera pixel coordinates.
        confidence: [N] float32 per-keypoint confidence.
        labels: list of N keypoint name strings.
        metadata: arbitrary metadata dict serialised as JSON into the bundle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(len(confidence))
    label_arr = np.asarray(labels if labels else ['dense'] * n)
    np.savez_compressed(
        output_path,
        left_xy=np.asarray(left_xy, dtype=np.float32),
        right_xy=np.asarray(right_xy, dtype=np.float32),
        confidence=np.asarray(confidence, dtype=np.float32),
        labels=label_arr,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


# ---------------------------------------------------------------------------
# Pair records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PairRecord:
    pair_json: Path | None
    session_id: str
    pair_idx: int
    left_source_frame_index: int
    right_source_frame_index: int


# ---------------------------------------------------------------------------
# Session discovery from input_dir
# ---------------------------------------------------------------------------

def _discover_sessions(
    input_dir: Path,
    anchor_cam: str,
    cam_subdir_tmpl: str,
    ext: str,
    eids: set[str] | None,
) -> dict[str, list[_PairRecord]]:
    """Discover sessions and frame pairs by scanning an extracted-frames directory.

    Expects ``{input_dir}/{cam_subdir}/`` to contain subdirectories whose names include
    a session UUID. Each subdirectory is scanned for ``img*.{ext}`` image files; the
    numeric index in the filename is used as both ``pair_idx`` and source frame index
    (cameras are synchronized).

    Args:
        input_dir: root of the extracted-frames directory (``input_dir`` from config).
        anchor_cam: camera name used to discover sessions (e.g. ``'left'``).
        cam_subdir_tmpl: template for the camera subdirectory (e.g. ``'{cam}Camera.video'``).
        ext: image file extension without leading dot (e.g. ``'png'``).
        eids: if not None, only return sessions whose UUID is in this set.

    Returns:
        mapping from session_id to list of _PairRecord, sorted by pair_idx.

    Raises:
        FileNotFoundError: if the anchor camera subdirectory does not exist.
        ValueError: if no sessions are found.
    """
    cam_subdir = cam_subdir_tmpl.format(cam=anchor_cam)
    cam_root = input_dir / cam_subdir
    if not cam_root.is_dir():
        raise FileNotFoundError(f'camera directory not found: {cam_root}')

    img_re = re.compile(rf'^img(\d+)\.{re.escape(ext)}$', re.IGNORECASE)
    grouped: dict[str, list[_PairRecord]] = {}

    for session_dir in sorted(cam_root.iterdir()):
        if not session_dir.is_dir():
            continue
        m = _UUID_RE.search(session_dir.name)
        if not m:
            continue
        session_id = m.group(1)
        if eids is not None and session_id not in eids:
            continue
        records: list[_PairRecord] = []
        for img_file in session_dir.iterdir():
            mm = img_re.match(img_file.name)
            if not mm:
                continue
            frame_idx = int(mm.group(1))
            records.append(_PairRecord(
                pair_json=None,
                session_id=session_id,
                pair_idx=frame_idx,
                left_source_frame_index=frame_idx,
                right_source_frame_index=frame_idx,
            ))
        records.sort(key=lambda r: r.pair_idx)
        if records:
            grouped[session_id] = records
            print(
                f'[info] discovered session={session_id} frames={len(records)}',
                flush=True,
            )

    return grouped


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def _build_session_rows(
    *,
    session_id: str,
    records: list[_PairRecord],
    csv_dir: Path,
    output_root: Path,
    keypoints: list[str],
    min_likelihood: float,
    shift_left: tuple[float, float],
    shift_right: tuple[float, float],
    left_frames_stretched: bool,
    overwrite: bool,
    n_digits: int,
) -> list[dict[str, Any]]:
    """Process one session: load CSVs, extract keypoints, write .npz bundles.

    Args:
        session_id: session UUID string.
        records: list of _PairRecord for this session.
        csv_dir: directory containing LitPose prediction CSV files.
        output_root: root directory for output bundle files.
        keypoints: list of keypoint names to extract.
        min_likelihood: minimum per-keypoint likelihood to include a match.
        shift_left: (dx, dy) pixel shift to bake into the left-camera nose keypoint.
        shift_right: (dx, dy) pixel shift to bake into the right-camera nose keypoint.
        left_frames_stretched: if True, left CSV rows are already stretched to 320×256.
            If False, keep raw 256×320 coordinates.
        overwrite: if False, skip pairs whose output bundle already exists.
        n_digits: zero-padding width for output filenames.

    Returns:
        list of status dicts, one per pair.
    """
    rows: list[dict[str, Any]] = []
    print(f'[info] start session={session_id} pairs={len(records)}', flush=True)

    if not records:
        print(f'[warn] skip session={session_id}: no usable pair entries', file=sys.stderr)
        return rows

    left_csv, right_csv = _litpose_csv_paths(csv_dir, session_id)
    if not left_csv.is_file() or not right_csv.is_file():
        msg = f'missing_csv left={left_csv.is_file()} right={right_csv.is_file()}'
        print(f'[warn] skip: {msg} :: session={session_id}', file=sys.stderr)
        for rec in records:
            rows.append({'session_id': session_id, 'pair_idx': rec.pair_idx, 'status': msg})
        return rows

    for csv_path, name in ((left_csv, 'left'), (right_csv, 'right')):
        eid = _eid_from_csv_name(csv_path)
        if eid != session_id:
            msg = f'eid_mismatch_{name}_file_eid={eid}'
            print(f'[warn] {msg} :: session={session_id}', file=sys.stderr)
            for rec in records:
                rows.append({'session_id': session_id, 'pair_idx': rec.pair_idx, 'status': msg})
            return rows

    try:
        t0 = time.perf_counter()
        bp_l, map_l = _load_dlc_frame_map(
            left_csv,
            stretch=left_frames_stretched,
            progress_label=f'left csv rows session={session_id}',
        )
        bp_r, map_r = _load_dlc_frame_map(
            right_csv,
            stretch=False,
            progress_label=f'right csv rows session={session_id}',
        )
        print(
            f'[info] loaded csvs session={session_id} '
            f'left_rows={len(map_l)} right_rows={len(map_r)} '
            f'elapsed={time.perf_counter() - t0:.1f}s',
            flush=True,
        )
    except (OSError, ValueError) as exc:
        print(f'[warn] skip csv session={session_id}: {exc}', file=sys.stderr)
        for rec in records:
            rows.append({
                'session_id': session_id,
                'pair_idx': rec.pair_idx,
                'status': f'csv_parse:{exc}',
            })
        return rows

    if bp_l != bp_r:
        print(f'[warn] bodyparts mismatch :: session={session_id}', file=sys.stderr)
        for rec in records:
            rows.append({
                'session_id': session_id,
                'pair_idx': rec.pair_idx,
                'status': 'bodyparts_mismatch',
            })
        return rows

    kp_starts = _keypoint_starts(bp_l, keypoints)

    for rec in records:
        out_path = _output_bundle_path(
            output_root,
            session_id=rec.session_id,
            pair_idx=rec.pair_idx,
            n_digits=n_digits,
        )
        if out_path.exists() and not overwrite:
            rows.append({
                'session_id': session_id,
                'pair_idx': rec.pair_idx,
                'bundle_path': str(out_path),
                'status': 'skipped_existing',
            })
            continue

        row_l = map_l.get(rec.left_source_frame_index)
        row_r = map_r.get(rec.right_source_frame_index)
        if row_l is None or row_r is None:
            print(
                f'[warn] skip: missing CSV row '
                f'left_idx={rec.left_source_frame_index} '
                f'right_idx={rec.right_source_frame_index} '
                f':: session={session_id} pair={rec.pair_idx}',
                file=sys.stderr,
            )
            rows.append({
                'session_id': session_id,
                'pair_idx': rec.pair_idx,
                'status': 'missing_frame_row',
                'left_index': rec.left_source_frame_index,
                'right_index': rec.right_source_frame_index,
            })
            continue

        built = _build_keypoint_arrays(
            bodyparts_flat=bp_l,
            left_vals=row_l,
            right_vals=row_r,
            keypoint_names=keypoints,
            keypoint_starts=kp_starts,
            min_likelihood=min_likelihood,
            shift_nose_left=shift_left,
            shift_nose_right=shift_right,
            left_already_stretched=left_frames_stretched,
        )
        if built is None:
            print(
                f'[warn] skip: no valid keypoints :: session={session_id} pair={rec.pair_idx}',
                file=sys.stderr,
            )
            rows.append({
                'session_id': session_id,
                'pair_idx': rec.pair_idx,
                'status': 'no_keypoints',
            })
            continue

        left_xy, right_xy, confidence, labels = built
        metadata: dict[str, Any] = {
            'backend': 'litpose_keypoints',
            'session_id': rec.session_id,
            'pair_idx': rec.pair_idx,
            'left_csv': str(left_csv),
            'right_csv': str(right_csv),
            'left_source_frame_index': rec.left_source_frame_index,
            'right_source_frame_index': rec.right_source_frame_index,
            'keypoints': list(keypoints),
            'left_csv_space': (
                'stretched_in_memory_from_256x320_to_320x256'
                if left_frames_stretched
                else 'raw_256x320'
            ),
            'confidence_rule': 'min(left_likelihood, right_likelihood) per keypoint',
        }
        if shift_left != (0.0, 0.0):
            metadata['shift_nose_leftCamera_applied'] = list(shift_left)
        if shift_right != (0.0, 0.0):
            metadata['shift_nose_rightCamera_applied'] = list(shift_right)

        _save_correspondence_bundle(
            out_path,
            left_xy=left_xy,
            right_xy=right_xy,
            confidence=confidence,
            labels=labels,
            metadata=metadata,
        )
        rows.append({
            'session_id': session_id,
            'pair_idx': rec.pair_idx,
            'bundle_path': str(out_path),
            'status': 'written',
            'n': int(len(confidence)),
        })

    print(f'[info] done session={session_id}', flush=True)
    return rows


# ---------------------------------------------------------------------------
# Parallel worker (must be picklable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SessionJob:
    session_id: str
    # (pair_idx, left_source_frame_index, right_source_frame_index)
    pair_entries: tuple[tuple[int, int, int], ...]
    csv_dir: str
    output_root: str
    keypoints: tuple[str, ...]
    min_likelihood: float
    shift_left: tuple[float, float]
    shift_right: tuple[float, float]
    left_frames_stretched: bool
    overwrite: bool
    n_digits: int


def _run_session_job(job: _SessionJob) -> list[dict[str, Any]]:
    records = [
        _PairRecord(
            pair_json=None,
            session_id=job.session_id,
            pair_idx=idx,
            left_source_frame_index=li,
            right_source_frame_index=ri,
        )
        for idx, li, ri in job.pair_entries
    ]
    return _build_session_rows(
        session_id=job.session_id,
        records=records,
        csv_dir=Path(job.csv_dir),
        output_root=Path(job.output_root),
        keypoints=list(job.keypoints),
        min_likelihood=job.min_likelihood,
        shift_left=job.shift_left,
        shift_right=job.shift_right,
        left_frames_stretched=job.left_frames_stretched,
        overwrite=job.overwrite,
        n_digits=job.n_digits,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--config',
        type=Path,
        default=None,
        metavar='YAML',
        help='path to extraction_pipeline_sable.yaml; supplies defaults for most flags',
    )
    p.add_argument(
        '--litpose-root',
        type=Path,
        default=None,
        help=(
            'root containing video_preds/ subdirectory with LitPose CSV files. '
            'Overrides litpose.video_preds_dir from --config (which is used directly '
            'as the CSV dir without appending video_preds/).'
        ),
    )
    p.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='where to write bundles. Overrides output_dir/dataset from --config.',
    )
    p.add_argument(
        '--keypoints',
        type=str,
        default=None,
        help='comma-separated bodypart names (default from config or: pawL,pawR,nose)',
    )
    p.add_argument('--min-likelihood', type=float, default=None)
    p.add_argument(
        '--overwrite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='overwrite existing bundles (default: --no-overwrite)',
    )
    p.add_argument(
        '--left-frames-stretched',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'when set, left-camera frames are already stored stretched to 320×256 and '
            'the in-memory stretch is applied before saving. '
            'Default: --no-left-frames-stretched — coordinates saved in raw 256×320 space '
            'for SABLEDataset to rescale at load time.'
        ),
    )
    p.add_argument(
        '--shift-nose-leftCamera',
        type=_parse_xy_pair,
        default=None,
        metavar='X,Y',
        help='pixel shift (dx,dy) baked into the left-camera nose row',
    )
    p.add_argument(
        '--shift-nose-rightCamera',
        type=_parse_xy_pair,
        default=None,
        metavar='X,Y',
        help='pixel shift (dx,dy) baked into the right-camera nose row',
    )
    p.add_argument(
        '--n-digits',
        type=int,
        default=None,
        metavar='N',
        help='zero-padding width for output filenames (default from config or: 8)',
    )
    p.add_argument(
        '--eids',
        nargs='+',
        metavar='EID',
        default=None,
        help='only process these session UUIDs (overrides sessionids from --config)',
    )
    p.add_argument(
        '--max-workers',
        type=int,
        default=None,
        metavar='N',
        help='parallel sessions (0 = min(cpu_count, n_sessions); default from config or: 1)',
    )
    return p


def _load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: path to the YAML file.

    Returns:
        parsed config as a dict.

    Raises:
        ValueError: if the file does not contain a YAML mapping.
    """
    with path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'expected a YAML mapping in {path}')
    return cfg


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()

    cfg: dict[str, Any] = {}
    if args.config is not None:
        cfg = _load_yaml_config(args.config.expanduser().resolve())

    litpose_cfg: dict[str, Any] = cfg.get('litpose') or {}
    frame_cfg: dict[str, Any] = cfg.get('frame') or {}
    video_naming_cfg: dict[str, Any] = cfg.get('video_naming') or {}

    # ---------------------------------------------------------------------------
    # Resolve csv_dir
    # ---------------------------------------------------------------------------
    if args.litpose_root is not None:
        csv_dir = args.litpose_root.expanduser().resolve() / 'video_preds'
    else:
        vp = litpose_cfg.get('video_preds_dir')
        if not vp:
            raise ValueError(
                'LitPose CSV directory unknown: pass --litpose-root or set '
                'litpose.video_preds_dir in --config'
            )
        csv_dir = Path(str(vp)).expanduser().resolve()

    # ---------------------------------------------------------------------------
    # Resolve output_root
    # ---------------------------------------------------------------------------
    if args.output_root is not None:
        output_root = args.output_root.expanduser().resolve()
    elif cfg.get('output_dir'):
        output_root = Path(str(cfg['output_dir'])).expanduser().resolve()
    else:
        raise ValueError(
            'output directory unknown: pass --output-root or set output_dir in --config'
        )

    # ---------------------------------------------------------------------------
    # Resolve input_dir for session discovery
    # ---------------------------------------------------------------------------
    if not cfg.get('input_dir'):
        raise ValueError('input_dir must be set in --config for session discovery')
    input_dir = Path(str(cfg['input_dir'])).expanduser().resolve()

    # ---------------------------------------------------------------------------
    # Scalar parameters: CLI overrides config overrides default
    # ---------------------------------------------------------------------------
    keypoints = (
        [s.strip() for s in str(args.keypoints).split(',') if s.strip()]
        if args.keypoints is not None
        else [str(k) for k in litpose_cfg.get('keypoints', ['pawL', 'pawR', 'nose'])]
    )
    min_likelihood = (
        args.min_likelihood if args.min_likelihood is not None
        else float(litpose_cfg.get('min_likelihood', 0.0))
    )
    n_digits = (
        args.n_digits if args.n_digits is not None
        else int(frame_cfg.get('n_digits', 8))
    )
    max_workers = (
        args.max_workers if args.max_workers is not None
        else int(cfg.get('max_workers', 1))
    )
    anchor_cam = str(cfg.get('anchor_view', 'left'))
    cam_subdir_tmpl = str(video_naming_cfg.get('camera_video_subdir', '{cam}Camera.video'))
    ext = str(frame_cfg.get('extension', 'png'))

    nose_shifts = (litpose_cfg.get('keypoint_shifts') or {}).get('nose') or {}
    _raw_left = nose_shifts.get('left') or [0.0, 0.0]
    default_shift_left: tuple[float, float] = (float(_raw_left[0]), float(_raw_left[1]))
    _raw_right = nose_shifts.get('right') or [0.0, 0.0]
    default_shift_right: tuple[float, float] = (float(_raw_right[0]), float(_raw_right[1]))
    shift_left: tuple[float, float] = (
        args.shift_nose_leftCamera if args.shift_nose_leftCamera is not None
        else default_shift_left
    )
    shift_right: tuple[float, float] = (
        args.shift_nose_rightCamera if args.shift_nose_rightCamera is not None
        else default_shift_right
    )

    # session filter: CLI --eids overrides config sessionids
    if args.eids is not None:
        eids: set[str] | None = set(args.eids)
    else:
        cfg_eids = cfg.get('sessionids')
        eids = {str(e) for e in cfg_eids} if cfg_eids else None

    print(f'[info] input_dir={input_dir}', flush=True)
    print(f'[info] csv_dir={csv_dir}', flush=True)
    print(f'[info] output_root={output_root}', flush=True)
    print(
        f'[info] left_frames_stretched={args.left_frames_stretched} '
        f'keypoints={keypoints} min_likelihood={min_likelihood} n_digits={n_digits}',
        flush=True,
    )

    # ---------------------------------------------------------------------------
    # Discover sessions
    # ---------------------------------------------------------------------------
    grouped = _discover_sessions(
        input_dir=input_dir,
        anchor_cam=anchor_cam,
        cam_subdir_tmpl=cam_subdir_tmpl,
        ext=ext,
        eids=eids,
    )
    if not grouped:
        raise ValueError(f'no sessions discovered under {input_dir}')

    n_sessions = len(grouped)
    total_pairs = sum(len(v) for v in grouped.values())
    print(f'[info] sessions={n_sessions} total_pairs={total_pairs}', flush=True)

    session_order = list(grouped.keys())
    cpus = os.cpu_count() or 1
    job_workers = (
        max(1, min(cpus, n_sessions)) if max_workers <= 0
        else max(1, min(max_workers, n_sessions))
    )

    rows_out: list[dict[str, Any]] = []
    processed = 0

    if job_workers <= 1 or n_sessions <= 1:
        for session_id in session_order:
            for row in _build_session_rows(
                session_id=session_id,
                records=grouped[session_id],
                csv_dir=csv_dir,
                output_root=output_root,
                keypoints=keypoints,
                min_likelihood=min_likelihood,
                shift_left=shift_left,
                shift_right=shift_right,
                left_frames_stretched=args.left_frames_stretched,
                overwrite=args.overwrite,
                n_digits=n_digits,
            ):
                rows_out.append(row)
                processed += 1
                _pair_progress(processed, total_pairs)
    else:
        print(f'[info] parallel: workers={job_workers} sessions={n_sessions}', flush=True)
        jobs = [
            _SessionJob(
                session_id=sid,
                pair_entries=tuple(
                    (rec.pair_idx, rec.left_source_frame_index, rec.right_source_frame_index)
                    for rec in grouped[sid]
                ),
                csv_dir=str(csv_dir),
                output_root=str(output_root),
                keypoints=tuple(keypoints),
                min_likelihood=float(min_likelihood),
                shift_left=shift_left,
                shift_right=shift_right,
                left_frames_stretched=args.left_frames_stretched,
                overwrite=args.overwrite,
                n_digits=n_digits,
            )
            for sid in session_order
        ]
        per_session: dict[str, list[dict[str, Any]]] = {}
        with ProcessPoolExecutor(max_workers=job_workers) as ex:
            futs = {ex.submit(_run_session_job, job): job.session_id for job in jobs}
            for fut in as_completed(futs):
                per_session[futs[fut]] = fut.result()
        for sid in session_order:
            for row in per_session[sid]:
                rows_out.append(row)
                processed += 1
                _pair_progress(processed, total_pairs)

    summary_dir = output_root / 'litpose_correspondences'
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / 'litpose_correspondence_precompute_summary.json'
    summary_path.write_text(json.dumps(rows_out, indent=2), encoding='utf-8')
    print(
        json.dumps({
            'rows': len(rows_out),
            'summary': str(summary_path),
            'output_root': str(output_root),
        }),
    )


if __name__ == '__main__':
    main()
