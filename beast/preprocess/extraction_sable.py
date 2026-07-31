"""SABLE IBL stereo extraction pipeline: stats, trim, downsample, extract, VDA, LitPose CSV.

Processes synchronized left/right stereo video pairs from an IBL-2view directory layout.
Output is a self-contained dataset directory compatible with IBLTwoViewDataset and SABLEDataset.

File naming is controlled by module-level template strings at the top of this file.
Modify those templates (and only those) to adapt the pipeline to different naming conventions.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from beast.preprocess.config_sable import SABLEConfig, VideoNamingConfig
from beast.preprocess.extraction import select_frame_idxs_kmeans
from beast.video import downsample_video, get_video_stats, trim_video

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# per-video worker functions (mirrors extraction_3d.py's _trim_one_video /
# _downsample_one; inlined here to avoid pulling in extraction_3d's optional
# aniposelib dependency via beast.io)
# ---------------------------------------------------------------------------

def _trim_one_video(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    threads: int | None,
) -> tuple[Path, int, int, bool]:
    """Worker: trim one video. Returns (input_path, start_frame, end_frame, was_skipped)."""
    if output_path.exists():
        return input_path, start_frame, end_frame, True
    trim_video(input_path, output_path, start_frame, end_frame, threads=threads)
    return input_path, start_frame, end_frame, False


def _downsample_one(
    input_path: Path,
    output_path: Path,
    target_fps: float | None,
    threads: int | None,
    phase_offset_frames: int,
) -> tuple[Path, bool]:
    """Worker: downsample one video. Returns (input_path, was_skipped)."""
    if output_path.exists():
        return input_path, True
    downsample_video(
        input_path,
        output_path,
        target_fps,
        threads=threads,
        phase_offset_frames=phase_offset_frames,
    )
    return input_path, False

# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def _video_path(input_dir: Path, cam: str, session_id: str, *, video_naming) -> Path:
    """Return the video path for one camera and session.

    Args:
        input_dir: IBL-2view root directory.
        cam: camera name (e.g. 'left').
        session_id: session identifier.
        video_naming: VideoNamingConfig object.

    Returns:
        path to the mp4 file.
    """
    subdir = video_naming.camera_video_subdir.format(cam=cam)
    filename = video_naming.video_filename.format(cam=cam, session_id=session_id)
    return input_dir / subdir / filename


def _timestamps_path(input_dir: Path, cam: str, session_id: str, *, video_naming) -> Path:
    """Return the timestamps .npy path for one camera and session.

    Args:
        input_dir: IBL-2view root directory.
        cam: camera name.
        session_id: session identifier.
        video_naming: VideoNamingConfig object.

    Returns:
        path to the .npy timestamps file.
    """
    return input_dir / video_naming.timestamp_subdir / video_naming.timestamp_filename.format(
        cam=cam, session_id=session_id
    )


def _video_session_re(cam: str, *, video_naming) -> re.Pattern:
    """Build a regex that extracts session_id from a video filename for one camera.

    The regex is derived from the video_naming.video_filename template so it stays
    in sync when the template is changed.

    Args:
        cam: camera name.
        video_naming: VideoNamingConfig object.

    Returns:
        compiled regex with one capture group for the session_id.
    """
    sentinel = '__SESSION_ID__'
    template_filled = video_naming.video_filename.format(cam=cam, session_id=sentinel)
    escaped = re.escape(template_filled)
    pattern = '^' + escaped.replace(re.escape(sentinel), r'(.+)') + '$'
    return re.compile(pattern)


# ---------------------------------------------------------------------------
# session discovery
# ---------------------------------------------------------------------------

def discover_sessions(
    input_dir: Path,
    cameras: list[str],
    video_naming,
    sessionids: list[str] | None = None,
) -> list[str]:
    """Find session IDs present in all camera video directories.

    Args:
        input_dir: IBL-2view root directory.
        cameras: list of camera names to require (e.g. ['left', 'right']).
        video_naming: VideoNamingConfig object.
        sessionids: optional explicit list of session IDs to filter to.

    Returns:
        sorted list of session ID strings.

    Raises:
        FileNotFoundError: if any camera video directory is missing.
        RuntimeError: if no matching sessions are found.
    """
    per_cam_ids: list[set[str]] = []
    for cam in cameras:
        cam_dir = input_dir / video_naming.camera_video_subdir.format(cam=cam)
        if not cam_dir.is_dir():
            raise FileNotFoundError(f'camera video directory not found: {cam_dir}')
        session_re = _video_session_re(cam, video_naming=video_naming)
        per_cam_ids.append({
            m.group(1)
            for f in cam_dir.iterdir()
            if (m := session_re.match(f.name))
        })

    ids = sorted(set.intersection(*per_cam_ids)) if per_cam_ids else []

    if sessionids:
        requested = set(sessionids)
        missing = requested - set(ids)
        if missing:
            _logger.warning(f'session IDs not found in all camera dirs: {sorted(missing)}')
        ids = [sid for sid in ids if sid in requested]

    if not ids:
        raise RuntimeError(
            f'no matching session IDs found under {input_dir} '
            f'(cameras={cameras})'
        )
    return ids


# ---------------------------------------------------------------------------
# stats step
# ---------------------------------------------------------------------------

def run_stats(
    cfg: SABLEConfig,
    session_ids: list[str],
) -> dict:
    """Scan all session videos and write video_stats.csv and video_stats.json.

    Args:
        cfg: SABLE pipeline config.
        session_ids: list of session IDs to scan.

    Returns:
        summary dict with aggregate statistics.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_fps: list[float] = []
    total_frames = 0
    total_duration = 0.0

    for session_id in session_ids:
        for cam in cfg.cameras:
            vpath = _video_path(input_dir, cam, session_id, video_naming=cfg.video_naming)
            if not vpath.is_file():
                _logger.warning(f'video not found, skipping stats: {vpath}')
                continue
            try:
                stats = get_video_stats(vpath)
            except OSError as exc:
                _logger.warning(f'could not read stats for {vpath}: {exc}')
                continue
            rows.append({
                'session_id': session_id,
                'camera': cam,
                'filename': vpath.name,
                **stats,
            })
            all_fps.append(stats['fps'])
            total_frames += stats['total_frames']
            total_duration += stats['duration_sec']

    if not rows:
        _logger.warning('no videos scanned — video_stats.csv will be empty')

    csv_path = output_dir / 'video_stats.csv'
    fieldnames = ['session_id', 'camera', 'filename', 'fps', 'width', 'height',
                  'total_frames', 'duration_sec', 'codec']
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    _logger.info(f'per-video stats → {csv_path}')

    n = len(rows)
    avg_fps = sum(all_fps) / n if n else 0.0
    summary = {
        'num_sessions': len(session_ids),
        'num_videos': n,
        'fps_min': round(min(all_fps), 2) if all_fps else 0.0,
        'fps_max': round(max(all_fps), 2) if all_fps else 0.0,
        'fps_avg': round(avg_fps, 2),
        'total_frames': total_frames,
        'total_duration_sec': round(total_duration, 2),
    }
    json_path = output_dir / 'video_stats.json'
    with open(json_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    _logger.info(f'summary stats → {json_path}')
    return summary


# ---------------------------------------------------------------------------
# trim step
# ---------------------------------------------------------------------------

def run_trim(cfg: SABLEConfig, session_ids: list[str]) -> Path:
    """Trim all session videos in parallel.

    Writes trimmed videos to ``output_dir/videos_trim/{cam}Camera/``.

    Args:
        cfg: SABLE pipeline config with trim.enabled=True.
        session_ids: list of session IDs.

    Returns:
        path to the trim output directory.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    trim_dir = output_dir / 'videos_trim'

    ffmpeg_threads: int | None = cfg.trim.ffmpeg_threads
    if ffmpeg_threads is None:
        cpu_count = os.cpu_count() or cfg.max_workers
        ffmpeg_threads = max(1, math.ceil(cpu_count / cfg.max_workers))

    jobs: list[tuple] = []
    for session_id in session_ids:
        for cam in cfg.cameras:
            src = _video_path(input_dir, cam, session_id, video_naming=cfg.video_naming)
            if not src.is_file():
                _logger.warning(f'trim: video not found, skipping: {src}')
                continue
            dst_dir = trim_dir / cfg.video_naming.camera_video_subdir.format(cam=cam)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / src.name

            stats = get_video_stats(src)
            total = stats['total_frames']
            fps = stats['fps']

            sf = cfg.trim.start_frame
            ef = cfg.trim.end_frame
            if sf is None:
                sf = int(round(cfg.trim.start_sec * fps)) if cfg.trim.start_sec is not None else 0
            if ef is None:
                ef = int(round(cfg.trim.end_sec * fps)) - 1 if cfg.trim.end_sec is not None else total - 1
            sf = max(0, int(sf))
            ef = min(total - 1, int(ef))
            jobs.append((src, dst, sf, ef))

    _logger.info(f'trimming {len(jobs)} videos (workers={cfg.max_workers})')
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(_trim_one_video, src, dst, sf, ef, ffmpeg_threads): src
            for src, dst, sf, ef in jobs
        }
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                _, _, _, skipped = fut.result()
            except Exception as exc:
                raise RuntimeError(f'trim failed for {src.name}: {exc}') from exc
            tag = 'skip (exists)' if skipped else 'done'
            _logger.info(f'  {src.name} {tag}')
    return trim_dir


# ---------------------------------------------------------------------------
# downsample step
# ---------------------------------------------------------------------------

def run_downsample(cfg: SABLEConfig, session_ids: list[str]) -> Path:
    """Downsample all session videos in parallel.

    Reads from trim output when trim is enabled, otherwise from input_dir.
    Writes downsampled videos to ``output_dir/videos/{cam}Camera/``.

    Args:
        cfg: SABLE pipeline config with downsample.enabled=True.
        session_ids: list of session IDs.

    Returns:
        path to the downsample output directory.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    ds_out = output_dir / 'videos'

    ffmpeg_threads: int | None = cfg.downsample.ffmpeg_threads
    if ffmpeg_threads is None:
        cpu_count = os.cpu_count() or cfg.max_workers
        ffmpeg_threads = max(1, math.ceil(cpu_count / cfg.max_workers))

    phase_offset = int(cfg.downsample.phase_offset_frames)

    jobs: list[tuple] = []
    for session_id in session_ids:
        for cam in cfg.cameras:
            cam_subdir = cfg.video_naming.camera_video_subdir.format(cam=cam)
            if cfg.trim.enabled:
                src = output_dir / 'videos_trim' / cam_subdir / cfg.video_naming.video_filename.format(
                    cam=cam, session_id=session_id
                )
            else:
                src = _video_path(input_dir, cam, session_id, video_naming=cfg.video_naming)
            if not src.is_file():
                _logger.warning(f'downsample: video not found, skipping: {src}')
                continue
            out_dir = ds_out / cam_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / src.name
            jobs.append((src, dst))

    _logger.info(f'downsampling {len(jobs)} videos (workers={cfg.max_workers})')
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(_downsample_one, src, dst, cfg.downsample.target_fps,
                        ffmpeg_threads, phase_offset): src
            for src, dst in jobs
        }
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                _, skipped = fut.result()
            except Exception as exc:
                raise RuntimeError(f'downsample failed for {src.name}: {exc}') from exc
            tag = 'skip (exists)' if skipped else 'done'
            _logger.info(f'  {src.name} {tag}')
    return ds_out


# ---------------------------------------------------------------------------
# extract step helpers
# ---------------------------------------------------------------------------

def _resolve_video_path(
    cfg: SABLEConfig,
    cam: str,
    session_id: str,
) -> Path:
    """Resolve the final video path for one camera/session after trim/downsample.

    Priority: downsample output > trim output > raw input.

    Args:
        cfg: SABLE pipeline config.
        cam: camera name.
        session_id: session identifier.

    Returns:
        resolved path to the mp4 file.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)
    cam_subdir = cfg.video_naming.camera_video_subdir.format(cam=cam)
    filename = cfg.video_naming.video_filename.format(cam=cam, session_id=session_id)

    if cfg.downsample.enabled:
        p = output_dir / 'videos' / cam_subdir / filename
        if p.is_file():
            return p

    if cfg.trim.enabled:
        p = output_dir / 'videos_trim' / cam_subdir / filename
        if p.is_file():
            return p

    return input_dir / cam_subdir / filename


def _assign_splits(
    n_pairs: int,
    split_names: list[str],
    split_ratios: list[float],
) -> list[str]:
    """Assign a split label to each pair index deterministically.

    Args:
        n_pairs: total number of pairs.
        split_names: list of split names.
        split_ratios: corresponding fractions; need not sum to 1.

    Returns:
        list of length ``n_pairs`` with split name per pair.
    """
    if not split_names or not split_ratios:
        return ['train'] * n_pairs
    total = sum(split_ratios)
    cumulative = 0.0
    boundaries: list[int] = []
    for ratio in split_ratios[:-1]:
        cumulative += ratio / total
        boundaries.append(int(round(cumulative * n_pairs)))

    splits: list[str] = []
    for i in range(n_pairs):
        bucket = 0
        for b in boundaries:
            if i >= b:
                bucket += 1
            else:
                break
        splits.append(split_names[min(bucket, len(split_names) - 1)])
    return splits


def _load_optional_timestamps(
    input_dir: Path,
    cam: str,
    session_id: str,
    n_frames: int,
    video_naming,
) -> np.ndarray | None:
    """Load per-frame timestamps from .npy file if it exists.

    Args:
        input_dir: IBL-2view root directory.
        cam: camera name.
        session_id: session identifier.
        n_frames: expected number of frames (for length validation).
        video_naming: VideoNamingConfig object.

    Returns:
        1-D float64 array of timestamps, or None if the file is absent.
    """
    ts_path = _timestamps_path(input_dir, cam, session_id, video_naming=video_naming)
    if not ts_path.is_file():
        return None
    times = np.load(str(ts_path))
    if len(times) < n_frames:
        _logger.debug(
            f'timestamps for {cam}/{session_id} shorter than video '
            f'({len(times)} vs {n_frames}); will use None for out-of-range frames'
        )
    return times


def _extract_one_session(
    session_id: str,
    cfg: SABLEConfig,
    dataset_dir: Path,
    overwrite: bool,
) -> dict:
    """Extract frames for one session mirroring extraction_3d.py's approach.

    1. Runs ``select_frame_idxs_kmeans`` on the anchor view video to select frame indices.
    2. Applies the same indices to every other camera (same as extraction_3d.py).
    3. Writes images and ``pair_metadata.json``.

    Args:
        session_id: session identifier.
        cfg: SABLE pipeline config.
        dataset_dir: output_dir/dataset/ directory.
        overwrite: re-extract even if pair_metadata.json already exists.

    Returns:
        summary dict with 'session_id' and 'n_selected'.
    """
    input_dir = Path(cfg.input_dir)
    session_dir = dataset_dir / session_id
    metadata_path = session_dir / 'pair_metadata.json'

    if metadata_path.exists() and not overwrite:
        _logger.info(f'extract: skipping {session_id} (pair_metadata.json exists)')
        existing = json.loads(metadata_path.read_text(encoding='utf-8'))
        return {'session_id': session_id, 'n_selected': len(existing.get('pairs', []))}

    # resolve video paths
    cam_videos: dict[str, Path] = {}
    for cam in cfg.cameras:
        vpath = _resolve_video_path(cfg, cam, session_id)
        if not vpath.is_file():
            _logger.warning(f'extract: video not found for {cam}/{session_id}: {vpath}')
            return {'session_id': session_id, 'n_selected': 0}
        cam_videos[cam] = vpath

    # run k-means on anchor view (mirrors extraction_3d.py assemble step)
    anchor_video = cam_videos[cfg.anchor_view]
    try:
        frame_idxs = select_frame_idxs_kmeans(
            video_file=anchor_video,
            resize_dims=cfg.frame.kmeans_resize,
            n_frames_to_select=cfg.frame.frames_per_video,
        )
    except ValueError as exc:
        if 'valid video segment too short' in str(exc):
            cap = cv2.VideoCapture(str(anchor_video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            sample_size = min(cfg.frame.frames_per_video, total)
            frame_idxs = np.sort(np.random.choice(total, size=sample_size, replace=False))
            _logger.warning(
                f'  {session_id}: video too short for k-means, randomly sampled '
                f'{sample_size} frames'
            )
        else:
            raise
    frame_idxs = np.sort(frame_idxs)

    # load optional timestamps per camera for metadata storage
    anchor_cap = cv2.VideoCapture(str(anchor_video))
    n_anchor_frames = int(anchor_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchor_cap.release()
    cam_timestamps: dict[str, np.ndarray | None] = {
        cam: _load_optional_timestamps(input_dir, cam, session_id, n_anchor_frames, cfg.video_naming)
        for cam in cfg.cameras
    }

    # assign splits
    split_labels = _assign_splits(
        len(frame_idxs),
        cfg.frame.split_names,
        cfg.frame.split_ratios,
    )

    n_digits = cfg.frame.n_digits
    extension = cfg.frame.extension

    # create camera output directories
    for cam in cfg.cameras:
        (session_dir / cam).mkdir(parents=True, exist_ok=True)

    # extract frames: same index applied to every camera (mirrors extraction_3d.py)
    cam_caps: dict[str, cv2.VideoCapture] = {
        cam: cv2.VideoCapture(str(cam_videos[cam])) for cam in cfg.cameras
    }
    try:
        pairs_meta = []
        for pair_idx, (frame_idx, split) in enumerate(zip(frame_idxs, split_labels)):
            pair_entry: dict = {
                'pair_idx': pair_idx,
                'split': split,
            }
            skip_pair = False
            for cam in cfg.cameras:
                out_path = session_dir / cam / f'img{frame_idx:0{n_digits}d}.{extension}'
                if not out_path.exists() or overwrite:
                    cam_caps[cam].set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ret, frame = cam_caps[cam].read()
                    if ret:
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(out_path)
                    else:
                        _logger.warning(
                            f'could not read frame {frame_idx} from {cam}/{session_id}'
                        )
                        skip_pair = True
                        break
                pair_entry[f'{cam}_path'] = f'{cam}/img{frame_idx:0{n_digits}d}.{extension}'
                pair_entry[f'{cam}_source_frame_index'] = int(frame_idx)
                times = cam_timestamps.get(cam)
                if times is not None and int(frame_idx) < len(times):
                    pair_entry[f'{cam}_timestamp_sec'] = round(float(times[int(frame_idx)]), 6)

            if not skip_pair:
                pairs_meta.append(pair_entry)
    finally:
        for cap in cam_caps.values():
            cap.release()

    metadata: dict = {'session_id': session_id, 'pairs': pairs_meta}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    _logger.info(f'extract: {session_id} → {len(pairs_meta)} pairs')
    return {'session_id': session_id, 'n_selected': len(pairs_meta)}


def run_extract(
    cfg: SABLEConfig,
    session_ids: list[str],
    overwrite: bool = False,
) -> Path:
    """Extract synchronized stereo frame pairs for all sessions.

    Args:
        cfg: SABLE pipeline config.
        session_ids: list of session IDs.
        overwrite: re-extract even if pair_metadata.json exists.

    Returns:
        path to the dataset directory.
    """
    dataset_dir = Path(cfg.output_dir) / 'dataset'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for session_id in tqdm(session_ids, desc='extract sessions'):
        summary = _extract_one_session(session_id, cfg, dataset_dir, overwrite)
        summaries.append(summary)

    info = {
        'name': cfg.name,
        'author': cfg.author,
        'created_at': datetime.now().isoformat(),
        'num_sessions': len(summaries),
        'total_pairs': sum(s.get('n_selected', 0) for s in summaries),
        'sessions': [s['session_id'] for s in summaries],
    }
    info_path = dataset_dir / 'info.json'
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')
    _logger.info(
        f'extract complete: {len(summaries)} sessions, '
        f'{info["total_pairs"]} total pairs → {dataset_dir}'
    )
    return dataset_dir


# ---------------------------------------------------------------------------
# litpose CSV → per-frame npy step
# ---------------------------------------------------------------------------

def run_litpose_csv_to_npy(
    cfg: SABLEConfig,
    dataset_dir: Path,
    session_ids: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Convert LitPose prediction CSVs to per-frame correspondence .npy files.

    For each pair in pair_metadata.json, reads the row for each camera's frame
    index from the LitPose CSV and saves a ``[K, 3]`` float32 array (x, y,
    likelihood) alongside the extracted images.

    Output path: ``{session_dir}/{cam}/correspondence{frame_idx:08d}.npy``

    Args:
        cfg: SABLE pipeline config (reads litpose sub-config).
        dataset_dir: output_dir/dataset/ directory.
        session_ids: optional session filter.
        overwrite: re-process even if .npy files already exist.

    Raises:
        ValueError: if video_preds_dir is not set.
    """
    lp = cfg.litpose
    if not lp.video_preds_dir:
        raise ValueError('litpose.video_preds_dir must be set when litpose.enabled is true')
    video_preds_dir = Path(lp.video_preds_dir).expanduser().resolve()

    keypoints: list[str] = list(lp.keypoints)
    min_likelihood: float = float(lp.min_likelihood)

    session_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if session_ids:
        sid_set = set(session_ids)
        session_dirs = [p for p in session_dirs if p.name in sid_set]

    for session_dir in session_dirs:
        metadata_path = session_dir / 'pair_metadata.json'
        if not metadata_path.is_file():
            continue
        payload = json.loads(metadata_path.read_text(encoding='utf-8'))
        session_id = payload.get('session_id', session_dir.name)
        pairs = payload.get('pairs', [])
        if not pairs:
            continue

        # load per-camera CSVs
        cam_dfs: dict = {}
        missing_cam = False
        for cam in cfg.cameras:
            csv_name = cfg.video_naming.video_filename.format(cam=cam, session_id=session_id).replace(
                '.mp4', '.csv'
            )
            csv_path = video_preds_dir / csv_name
            if not csv_path.is_file():
                _logger.warning(
                    f'litpose csv→npy: CSV not found for {cam}/{session_id}: {csv_path}'
                )
                missing_cam = True
                break
            try:
                import pandas as pd
                df = pd.read_csv(str(csv_path), header=[1, 2], index_col=0)
                cam_dfs[cam] = df
            except Exception as exc:
                _logger.warning(f'litpose csv→npy: failed to read {csv_path}: {exc}')
                missing_cam = True
                break
        if missing_cam:
            continue

        for pair in pairs:
            for cam in cfg.cameras:
                frame_idx = int(pair.get(f'{cam}_source_frame_index', pair.get('pair_idx', 0)))
                out_path = session_dir / cam / f'correspondence{frame_idx:08d}.npy'
                if out_path.exists() and not overwrite:
                    continue

                df = cam_dfs[cam]
                kp_data: list[list[float]] = []
                for kp in keypoints:
                    try:
                        row = df.loc[frame_idx]
                        x = float(row[(kp, 'x')])
                        y = float(row[(kp, 'y')])
                        lh = float(row[(kp, 'likelihood')])
                    except (KeyError, TypeError):
                        kp_data.append([0.0, 0.0, 0.0])
                        continue

                    # apply per-keypoint per-camera shift
                    shift_entry = lp.keypoint_shifts.get(kp, {})
                    cam_shifts = shift_entry.get(cam, [0.0, 0.0])
                    x += float(cam_shifts[0]) if len(cam_shifts) > 0 else 0.0
                    y += float(cam_shifts[1]) if len(cam_shifts) > 1 else 0.0

                    conf = lh if lh >= min_likelihood else 0.0
                    kp_data.append([x, y, conf])

                arr = np.array(kp_data, dtype=np.float32)
                np.save(str(out_path), arr)

        _logger.info(f'litpose csv→npy: {session_id} → {len(pairs)} pairs processed')


# ---------------------------------------------------------------------------
# main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: SABLEConfig,
    *,
    skip_stats: bool = False,
    overwrite: bool = False,
) -> None:
    """Run the full SABLE IBL stereo extraction pipeline.

    Steps (each gated by config flags):
    1. Stats (always runs unless skip_stats)
    2. Trim (cfg.trim.enabled)
    3. Downsample (cfg.downsample.enabled)
    4. Extract (always runs)
    5. VDA depth precompute (cfg.vda.enabled)
    6. LitPose CSV→npy (cfg.litpose.enabled)

    Session filtering is read from ``cfg.sessionids``; ``None`` processes all sessions.

    Args:
        cfg: SABLE pipeline config.
        skip_stats: if True, skip the stats step.
        overwrite: if True, re-run steps even if outputs exist.
    """
    input_dir = Path(cfg.input_dir).expanduser().resolve()
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(f'SABLE extraction pipeline: input={input_dir} output={output_dir}')

    session_ids = discover_sessions(input_dir, cfg.cameras, cfg.video_naming, cfg.sessionids or None)
    _logger.info(f'sessions to process: {len(session_ids)}')

    if not skip_stats:
        _logger.info('--- step 1: stats ---')
        run_stats(cfg, session_ids)
    else:
        _logger.info('--- step 1: stats (skipped) ---')

    if cfg.trim.enabled:
        _logger.info('--- step 2: trim ---')
        run_trim(cfg, session_ids)
    else:
        _logger.info('--- step 2: trim (disabled) ---')

    if cfg.downsample.enabled:
        _logger.info('--- step 3: downsample ---')
        run_downsample(cfg, session_ids)
    else:
        _logger.info('--- step 3: downsample (disabled) ---')

    _logger.info('--- step 4: extract ---')
    dataset_dir = run_extract(cfg, session_ids, overwrite=overwrite)

    if cfg.vda.enabled:
        _logger.info('--- step 5: VDA depth precompute ---')
        from beast.preprocess.precompute_vda_sable import run_vda_precompute
        run_vda_precompute(
            dataset_root=dataset_dir,
            vda_cfg=cfg.vda,
            cameras=cfg.cameras,
            sessionids=session_ids,
            overwrite=overwrite,
        )
    else:
        _logger.info('--- step 5: VDA depth (disabled) ---')

    if cfg.segmentation.enabled:
        _logger.info('--- step 6: SAM3 segmentation ---')
        from beast.preprocess.precompute_sam3_sable import run_sam3_precompute
        run_sam3_precompute(
            dataset_root=dataset_dir,
            seg_cfg=cfg.segmentation,
            cameras=cfg.cameras,
            sessionids=session_ids,
            overwrite=overwrite,
        )
    else:
        _logger.info('--- step 6: SAM3 segmentation (disabled) ---')

    if cfg.litpose.enabled:
        _logger.info('--- step 7: litpose CSV → npy ---')
        run_litpose_csv_to_npy(cfg, dataset_dir, session_ids=session_ids, overwrite=overwrite)
    else:
        _logger.info('--- step 7: litpose CSV → npy (disabled) ---')

    _logger.info(f'pipeline complete → {output_dir}')
