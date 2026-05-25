#!/usr/bin/env python3
"""Build LitPose keypoint correspondence bundles for Sable training.

Reads raw LitPose DLC-style CSVs (left camera 256×320, right camera 320×256) and writes
one ``.npz`` bundle per stereo pair under
``{output_root}/{session_id}/pair_{pair_idx:06d}/litpose_matches.npz``.

With ``--no-left-frames-stretched`` (the default), left-camera coordinates are stored in raw
256×320 pixel space. ``IBLDataset`` rescales them to the model's ``image_size`` at load time.
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

# IBL rig left-camera geometry: raw CSV is 256 wide × 320 tall; stretched to 320×256.
_LEFT_CSV_SRC_W, _LEFT_CSV_SRC_H = 256, 320
_LEFT_CSV_DST_W, _LEFT_CSV_DST_H = 320, 256

_BUNDLE_FILENAME = 'litpose_matches.npz'
_UUID_RE = re.compile(
    r'([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})'
)
_PAIR_IDX_RE = re.compile(r'(\d+)$')


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


def _load_pair_list(dataset_list: Path) -> list[Path]:
    """Read a text file of pair-JSON paths, one per line."""
    lines = dataset_list.read_text(encoding='utf-8').splitlines()
    return [Path(l.strip()) for l in lines if l.strip()]


def _session_id_from_path(pair_json: Path) -> str:
    return pair_json.resolve().parent.parent.name


def _pair_idx_from_path(pair_json: Path) -> int:
    m = _PAIR_IDX_RE.search(pair_json.stem)
    if m is None:
        raise RuntimeError(f'cannot parse pair_idx from filename: {pair_json}')
    return int(m.group(1))


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
            (``--no-left-frames-stretched`` mode: coordinates stay raw for IBLDataset to rescale).

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

def _output_bundle_path(output_root: Path, *, session_id: str, pair_idx: int) -> Path:
    return output_root / session_id / f'pair_{pair_idx:06d}' / _BUNDLE_FILENAME


def _save_correspondence_bundle(
    output_path: Path,
    *,
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    confidence: np.ndarray,
    labels: list[str],
    metadata: dict[str, Any],
) -> None:
    """Save a litpose_matches.npz correspondence bundle.

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
# Pair metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PairRecord:
    pair_json: Path
    session_id: str
    pair_idx: int
    left_source_frame_index: int
    right_source_frame_index: int


def _load_pair_frame_index_map(pair_metadata_json: Path) -> dict[int, tuple[int, int]]:
    payload = json.loads(pair_metadata_json.read_text(encoding='utf-8'))
    pairs = payload.get('pairs')
    if not isinstance(pairs, list):
        raise RuntimeError(f'missing pairs list: {pair_metadata_json}')
    out: dict[int, tuple[int, int]] = {}
    for item in pairs:
        if not isinstance(item, dict):
            continue
        try:
            out[int(item['pair_idx'])] = (
                int(item['left_source_frame_index']),
                int(item['right_source_frame_index']),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _load_pair_record(pair_json: Path) -> _PairRecord:
    payload = json.loads(pair_json.read_text(encoding='utf-8'))
    frames = payload.get('frames')
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f'invalid frames payload: {pair_json}')
    session_id = str(payload.get('session_id') or pair_json.resolve().parent.parent.name)
    pair_idx = int(payload.get('pair_id', payload.get('pair_idx', 0)))
    left_idx: int | None = None
    right_idx: int | None = None
    for frame in frames:
        if not isinstance(frame, dict):
            raise RuntimeError(f'invalid frame in {pair_json}')
        cam = str(frame.get('camera_name', '')).lower()
        if cam == 'left' and left_idx is None:
            left_idx = int(frame['source_frame_index'])
        elif cam == 'right' and right_idx is None:
            right_idx = int(frame['source_frame_index'])
    if left_idx is None or right_idx is None:
        raise RuntimeError(f'missing left/right camera_name in {pair_json}')
    return _PairRecord(
        pair_json=pair_json,
        session_id=session_id,
        pair_idx=pair_idx,
        left_source_frame_index=left_idx,
        right_source_frame_index=right_idx,
    )


def _records_for_session(
    *,
    session_id: str,
    pair_entries: list[tuple[int, Path]],
    dataset_root: Path,
) -> tuple[list[_PairRecord], list[dict[str, Any]]]:
    early: list[dict[str, Any]] = []
    metadata_path = (
        dataset_root / 'processed' / 'precached_video' / session_id / 'pair_metadata.json'
    )
    pair_index_map: dict[int, tuple[int, int]] | None = None
    if metadata_path.is_file():
        try:
            print(f'[info] loading pair metadata: {metadata_path}', flush=True)
            pair_index_map = _load_pair_frame_index_map(metadata_path)
            print(
                f'[info] loaded pair metadata session={session_id} '
                f'indexed_pairs={len(pair_index_map)}',
                flush=True,
            )
        except Exception as exc:
            print(
                f'[warn] pair metadata unreadable session={session_id} ({exc}); '
                f'falling back to pair JSON',
                file=sys.stderr,
            )
            pair_index_map = None
    else:
        print(
            f'[info] no pair_metadata.json for session={session_id}; using pair JSON',
            flush=True,
        )

    records: list[_PairRecord] = []
    if pair_index_map is not None:
        for pair_idx, pair_json in pair_entries:
            frames = pair_index_map.get(pair_idx)
            if frames is None:
                early.append({
                    'pair_json': str(pair_json),
                    'session_id': session_id,
                    'status': 'pair_idx_missing_in_pair_metadata',
                    'pair_idx': pair_idx,
                })
                continue
            li, ri = frames
            records.append(_PairRecord(
                pair_json=pair_json,
                session_id=session_id,
                pair_idx=pair_idx,
                left_source_frame_index=li,
                right_source_frame_index=ri,
            ))
        return records, early

    for pair_idx, pair_json in pair_entries:
        try:
            records.append(_load_pair_record(pair_json))
        except Exception as exc:
            early.append({'pair_json': str(pair_json), 'status': f'error_load_payload:{exc}'})
            print(f'[warn] skip (payload): {pair_json}: {exc}', file=sys.stderr)
    return records, early


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def _build_session_rows(
    *,
    session_id: str,
    pair_entries: list[tuple[int, Path]],
    dataset_root: Path,
    csv_dir: Path,
    output_root: Path,
    keypoints: list[str],
    min_likelihood: float,
    shift_left: tuple[float, float],
    shift_right: tuple[float, float],
    left_frames_stretched: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Process one session: load CSVs, extract keypoints, write .npz bundles.

    Args:
        session_id: session UUID string.
        pair_entries: list of (pair_idx, pair_json_path) tuples.
        dataset_root: root directory of the dataset (parent of ``processed/``).
        csv_dir: directory containing LitPose prediction CSV files.
        output_root: root directory for output bundle files.
        keypoints: list of keypoint names to extract.
        min_likelihood: minimum per-keypoint likelihood to include a match.
        shift_left: (dx, dy) pixel shift to bake into the left-camera nose keypoint.
        shift_right: (dx, dy) pixel shift to bake into the right-camera nose keypoint.
        left_frames_stretched: if True, left CSV rows are already stretched to 320×256
            (apply stretch in-memory). If False, keep raw 256×320 coordinates.
        overwrite: if False, skip pairs whose output bundle already exists.

    Returns:
        list of status dicts, one per pair.
    """
    rows: list[dict[str, Any]] = []
    print(f'[info] start session={session_id} pairs={len(pair_entries)}', flush=True)
    records, early = _records_for_session(
        session_id=session_id,
        pair_entries=pair_entries,
        dataset_root=dataset_root,
    )
    rows.extend(early)

    if not records:
        print(
            f'[warn] skip session={session_id}: no usable pair entries',
            file=sys.stderr,
        )
        return rows

    left_csv, right_csv = _litpose_csv_paths(csv_dir, session_id)
    if not left_csv.is_file() or not right_csv.is_file():
        msg = f'missing_csv left={left_csv.is_file()} right={right_csv.is_file()}'
        print(f'[warn] skip: {msg} :: session={session_id}', file=sys.stderr)
        for rec in records:
            rows.append({'pair_json': str(rec.pair_json), 'session_id': session_id, 'status': msg})
        return rows

    for csv_path, name in ((left_csv, 'left'), (right_csv, 'right')):
        eid = _eid_from_csv_name(csv_path)
        if eid != session_id:
            msg = f'eid_mismatch_{name}_file_eid={eid}'
            print(f'[warn] {msg} :: session={session_id}', file=sys.stderr)
            for rec in records:
                rows.append({'pair_json': str(rec.pair_json), 'status': msg})
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
            rows.append({'pair_json': str(rec.pair_json), 'status': f'csv_parse:{exc}'})
        return rows

    if bp_l != bp_r:
        print(f'[warn] bodyparts mismatch :: session={session_id}', file=sys.stderr)
        for rec in records:
            rows.append({'pair_json': str(rec.pair_json), 'status': 'bodyparts_mismatch'})
        return rows

    kp_starts = _keypoint_starts(bp_l, keypoints)

    for rec in records:
        out_path = _output_bundle_path(output_root, session_id=rec.session_id, pair_idx=rec.pair_idx)
        if out_path.exists() and not overwrite:
            rows.append({
                'pair_json': str(rec.pair_json),
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
                f'right_idx={rec.right_source_frame_index} :: {rec.pair_json}',
                file=sys.stderr,
            )
            rows.append({
                'pair_json': str(rec.pair_json),
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
            print(f'[warn] skip: no valid keypoints :: {rec.pair_json}', file=sys.stderr)
            rows.append({'pair_json': str(rec.pair_json), 'status': 'no_keypoints'})
            continue

        left_xy, right_xy, confidence, labels = built
        metadata: dict[str, Any] = {
            'backend': 'litpose_keypoints',
            'pair_json': str(rec.pair_json),
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
            'pair_json': str(rec.pair_json),
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
    pair_entries: tuple[tuple[int, str], ...]
    dataset_root: str
    csv_dir: str
    output_root: str
    keypoints: tuple[str, ...]
    min_likelihood: float
    shift_left: tuple[float, float]
    shift_right: tuple[float, float]
    left_frames_stretched: bool
    overwrite: bool


def _run_session_job(job: _SessionJob) -> list[dict[str, Any]]:
    return _build_session_rows(
        session_id=job.session_id,
        pair_entries=[(idx, Path(p)) for idx, p in job.pair_entries],
        dataset_root=Path(job.dataset_root),
        csv_dir=Path(job.csv_dir),
        output_root=Path(job.output_root),
        keypoints=list(job.keypoints),
        min_likelihood=job.min_likelihood,
        shift_left=job.shift_left,
        shift_right=job.shift_right,
        left_frames_stretched=job.left_frames_stretched,
        overwrite=job.overwrite,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--dataset-list',
        type=Path,
        required=True,
        help='path to opencv_cameras_pairs.txt (one pair-JSON path per line)',
    )
    p.add_argument(
        '--litpose-root',
        type=Path,
        required=True,
        help='root containing video_preds/ subdirectory with LitPose CSV files',
    )
    p.add_argument(
        '--output-root',
        type=Path,
        default=None,
        help='where to write bundles (default: {litpose-root}/processed_correspondences)',
    )
    p.add_argument(
        '--keypoints',
        type=str,
        default='pawL,pawR,nose',
        help='comma-separated bodypart names (default: pawL,pawR,nose)',
    )
    p.add_argument('--min-likelihood', type=float, default=0.0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument(
        '--left-frames-stretched',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'when set, left-camera frames are already stored stretched to 320×256 and '
            'the in-memory stretch is applied before saving. '
            'Default: --no-left-frames-stretched — coordinates saved in raw 256×320 space '
            'for IBLDataset to rescale at load time.'
        ),
    )
    p.add_argument(
        '--shift-nose-leftCamera',
        type=_parse_xy_pair,
        default=(0.0, 0.0),
        metavar='X,Y',
        help='pixel shift (dx,dy) baked into the left-camera nose row',
    )
    p.add_argument(
        '--shift-nose-rightCamera',
        type=_parse_xy_pair,
        default=(0.0, 0.0),
        metavar='X,Y',
        help='pixel shift (dx,dy) baked into the right-camera nose row',
    )
    p.add_argument(
        '--eids',
        nargs='+',
        metavar='EID',
        default=None,
        help='only process these session UUIDs',
    )
    p.add_argument(
        '--jobs', '-j',
        type=int,
        default=1,
        metavar='N',
        help='parallel sessions (0 = min(cpu_count, n_sessions); default: 1)',
    )
    return p


def main() -> None:
    """Entry point."""
    args = _build_parser().parse_args()
    dataset_list = args.dataset_list.expanduser().resolve()
    dataset_root = dataset_list.parent
    litpose_root = args.litpose_root.expanduser().resolve()
    csv_dir = litpose_root / 'video_preds'
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else litpose_root / 'processed_correspondences'
    )
    keypoints = [s.strip() for s in str(args.keypoints).split(',') if s.strip()]
    shift_left: tuple[float, float] = tuple(float(x) for x in args.shift_nose_leftCamera)
    shift_right: tuple[float, float] = tuple(float(x) for x in args.shift_nose_rightCamera)

    print(f'[info] dataset_list={dataset_list}', flush=True)
    print(f'[info] csv_dir={csv_dir}', flush=True)
    print(f'[info] output_root={output_root}', flush=True)
    print(
        f'[info] left_frames_stretched={args.left_frames_stretched} '
        f'keypoints={keypoints} min_likelihood={args.min_likelihood}',
        flush=True,
    )

    pairs = _load_pair_list(dataset_list)
    total_pairs = len(pairs)
    print(f'[info] total pairs: {total_pairs}', flush=True)

    grouped: dict[str, list[tuple[int, Path]]] = {}
    rows_out: list[dict[str, Any]] = []
    processed = 0

    for i, pair_json in enumerate(pairs, start=1):
        try:
            session_id = _session_id_from_path(pair_json)
            pair_idx = _pair_idx_from_path(pair_json)
        except (RuntimeError, OSError, ValueError) as exc:
            rows_out.append({'pair_json': str(pair_json), 'status': f'error_parse_path:{exc}'})
            processed += 1
            _pair_progress(processed, total_pairs)
            print(f'[warn] skip (path): {pair_json}: {exc}', file=sys.stderr)
            continue
        grouped.setdefault(session_id, []).append((pair_idx, pair_json))
        _report_progress(i, total_pairs, label='loaded pair paths', every=500)

    if args.eids is not None:
        want = set(args.eids)
        unknown = sorted(want - set(grouped))
        if unknown:
            print(
                f'[warn] --eids not found in pair paths (ignored): {", ".join(unknown)}',
                file=sys.stderr,
            )
        grouped = {sid: pe for sid, pe in grouped.items() if sid in want}
        print(f'[info] after --eids filter: sessions={len(grouped)}', flush=True)
        if not grouped:
            raise ValueError('no sessions left after --eids filter')

    session_order = list(grouped.keys())
    n_sessions = len(session_order)
    cpus = os.cpu_count() or 1
    job_workers = max(1, min(cpus, n_sessions)) if args.jobs <= 0 else max(1, min(args.jobs, n_sessions))

    if job_workers <= 1 or n_sessions <= 1:
        for session_id in session_order:
            for row in _build_session_rows(
                session_id=session_id,
                pair_entries=grouped[session_id],
                dataset_root=dataset_root,
                csv_dir=csv_dir,
                output_root=output_root,
                keypoints=keypoints,
                min_likelihood=float(args.min_likelihood),
                shift_left=shift_left,
                shift_right=shift_right,
                left_frames_stretched=args.left_frames_stretched,
                overwrite=args.overwrite,
            ):
                rows_out.append(row)
                processed += 1
                _pair_progress(processed, total_pairs)
    else:
        print(f'[info] parallel: workers={job_workers} sessions={n_sessions}', flush=True)
        jobs = [
            _SessionJob(
                session_id=sid,
                pair_entries=tuple((idx, str(p)) for idx, p in grouped[sid]),
                dataset_root=str(dataset_root),
                csv_dir=str(csv_dir),
                output_root=str(output_root),
                keypoints=tuple(keypoints),
                min_likelihood=float(args.min_likelihood),
                shift_left=shift_left,
                shift_right=shift_right,
                left_frames_stretched=args.left_frames_stretched,
                overwrite=args.overwrite,
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

    summary_path = output_root / 'litpose_correspondence_precompute_summary.json'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
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
