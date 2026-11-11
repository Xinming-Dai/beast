"""Ground-truth camera calibration helpers for the cheese3d rig.

Loads anipose-style ``.toml`` calibrations, expresses the cameras relative to the view
the model uses as its canonical frame, and normalises the rig scale by its baseline. The
same output feeds both ``training.camera_prior: gt`` (dataset provides the cameras) and
``training.camera_prior: init`` (pose head is initialised at the GT rig).
"""

import logging
from pathlib import Path

import numpy as np
import toml

_logger = logging.getLogger(__name__)

CAMERA_PRIOR_MODES = ('none', 'init', 'gt')

# the model pads the views to this many before picking its canonical (middle) view
NUM_VIEWS_PADDED = 10


def resolve_camera_prior(training: dict) -> str:
    """Resolve which camera prior mode a ``training`` config section asks for.

    ``training.camera_prior`` is authoritative when set. Otherwise the legacy flag
    ``training.use_camera_params`` selects ``gt`` (read from per-frame ``.npy`` files).

    Args:
        training: the ``training`` section of the config.

    Returns:
        one of ``'none'``, ``'init'`` or ``'gt'``.

    Raises:
        ValueError: if ``camera_prior`` is unknown, or ``init``/``gt`` is requested
            without ``training.calibration_dir``.
    """
    mode = training.get('camera_prior')
    if mode is None:
        return 'gt' if training.get('use_camera_params', False) else 'none'
    mode = str(mode).lower()
    if mode not in CAMERA_PRIOR_MODES:
        raise ValueError(
            f'training.camera_prior must be one of {CAMERA_PRIOR_MODES}; got {mode!r}'
        )
    if mode != 'none' and not training.get('calibration_dir'):
        raise ValueError(
            f'training.camera_prior={mode!r} requires training.calibration_dir to be set.'
        )
    return mode


def resolve_cheese3d_cameras(
    training: dict,
) -> tuple[str, str, str | None, list[str]]:
    """Resolve the cheese3d camera names from a ``training`` config section.

    View order is ``[left, right, (center), *extras]``.

    Args:
        training: the ``training`` section of the config.

    Returns:
        tuple of (left camera, right camera, center camera or ``None``, extra cameras).

    Raises:
        ValueError: if ``cheese3d_cameras`` is malformed or combined with
            ``cheese3d_center_camera``.
    """
    cameras_raw = training.get('cheese3d_cameras')
    if cameras_raw:
        if training.get('cheese3d_center_camera'):
            raise ValueError(
                'training.cheese3d_cameras cannot be combined with '
                'training.cheese3d_center_camera.'
            )
        cameras = [str(camera) for camera in cameras_raw]
        if len(cameras) < 2 or len(set(cameras)) != len(cameras):
            raise ValueError(
                f'training.cheese3d_cameras must list two or more distinct cameras; '
                f'got {cameras}'
            )
        left_camera, right_camera, *extra_cameras = cameras
        return left_camera, right_camera, None, extra_cameras
    center_camera_raw = training.get('cheese3d_center_camera')
    return (
        str(training.get('cheese3d_left_camera', 'TL')),
        str(training.get('cheese3d_right_camera', 'TR')),
        str(center_camera_raw) if center_camera_raw else None,
        [],
    )


def resolve_cheese3d_camera_names(training: dict) -> list[str]:
    """Resolve the ordered cheese3d camera names (view order) from a ``training`` section.

    Args:
        training: the ``training`` section of the config.

    Returns:
        camera names in view order, ``[left, right, (center), *extras]``.
    """
    left, right, center, extras = resolve_cheese3d_cameras(training)
    return [left, right, *([center] if center else []), *extras]


def resolve_reference_index(
    num_views: int,
    num_views_padded: int = NUM_VIEWS_PADDED,
) -> int:
    """Index of the real view that the model's ``canonical: middle`` frame maps to.

    The model pads to ``num_views_padded`` views by repeating the last real view and
    uses view ``num_views_padded // 2`` as canonical, which is a copy of the last real
    view whenever there are fewer than ``num_views_padded // 2 + 1`` real views.

    Args:
        num_views: number of real views.
        num_views_padded: number of views after padding.

    Returns:
        index into the real views of the canonical view.
    """
    return min(num_views_padded // 2, num_views - 1)


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a rotation matrix.

    Args:
        rvec: rotation vector ``[3]`` (axis * angle, radians).

    Returns:
        float64 rotation matrix ``[3, 3]``.
    """
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    skew = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * skew @ skew


def load_toml_calibration(
    path: Path | str,
    camera_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load world-to-camera extrinsics and normalised intrinsics from an anipose toml.

    Distortion is ignored, matching the per-frame ``.npy`` calibrations.

    Args:
        path: calibration ``.toml`` file with ``[cam_i]`` tables.
        camera_names: cameras to load, in the order they should be returned.

    Returns:
        w2c: float32 ``[V, 4, 4]`` world-to-camera matrices.
        fxfycxcy: float32 ``[V, 4]`` intrinsics ``[fx/W, fy/H, cx/W, cy/H]``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        KeyError: if a requested camera is not in the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'calibration file not found: {path}')
    tables = {
        table['name']: table
        for key, table in toml.load(path).items()
        if key.startswith('cam_')
    }
    w2c_list, fxfycxcy_list = [], []
    for name in camera_names:
        if name not in tables:
            raise KeyError(
                f'camera {name!r} not in {path}; available cameras: {sorted(tables)}'
            )
        table = tables[name]
        w2c = np.eye(4)
        w2c[:3, :3] = rodrigues_to_matrix(table['rotation'])
        w2c[:3, 3] = table['translation']
        matrix = np.asarray(table['matrix'], dtype=np.float64)
        width, height = table['size']
        w2c_list.append(w2c)
        fxfycxcy_list.append([
            matrix[0, 0] / width,
            matrix[1, 1] / height,
            matrix[0, 2] / width,
            matrix[1, 2] / height,
        ])
    return (
        np.stack(w2c_list).astype(np.float32),
        np.asarray(fxfycxcy_list, dtype=np.float32),
    )


def to_reference_frame(
    c2w: np.ndarray,
    idx_reference: int,
    baseline: float,
) -> tuple[np.ndarray, float]:
    """Express camera-to-world poses relative to a reference view, with scale normalised.

    The reference view becomes the identity and translations are divided by
    ``scale = mean_i ||C_i - C_ref|| / baseline`` so that the mean camera-to-reference
    distance equals ``baseline``.

    Args:
        c2w: camera-to-world matrices ``[V, 4, 4]``.
        idx_reference: index of the reference view.
        baseline: target mean camera-to-reference distance.

    Returns:
        c2w_rel: float32 ``[V, 4, 4]`` poses in the reference frame, rescaled.
        scale: the divisor applied to translations.

    Raises:
        ValueError: if ``baseline`` is not positive.
    """
    if baseline <= 0:
        raise ValueError(f'baseline must be positive; got {baseline}')
    c2w = c2w.astype(np.float64)
    c2w_rel = np.linalg.inv(c2w[idx_reference])[None] @ c2w
    dists = np.linalg.norm(c2w_rel[:, :3, 3], axis=-1)
    num_others = len(dists) - 1
    mean_dist = dists.sum() / num_others if num_others > 0 else 0.0
    scale = mean_dist / baseline if mean_dist > 0 else 1.0
    c2w_rel[:, :3, 3] /= scale
    return c2w_rel.astype(np.float32), float(scale)


def build_camera_prior(
    path: Path | str,
    camera_names: list[str],
    baseline: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the GT camera prior used by both ``init`` and ``gt`` modes.

    Args:
        path: calibration ``.toml`` file.
        camera_names: cameras in view order.
        baseline: target mean camera-to-reference distance.

    Returns:
        c2w: float32 ``[V, 4, 4]`` poses relative to the reference view, rescaled.
        fxfycxcy: float32 ``[V, 4]`` normalised intrinsics.
        idx_reference: index of the reference (canonical) view.
    """
    w2c, fxfycxcy = load_toml_calibration(path, camera_names)
    idx_reference = resolve_reference_index(len(camera_names))
    c2w, scale = to_reference_frame(np.linalg.inv(w2c), idx_reference, baseline)
    _logger.info(
        f'loaded camera prior from {path}: cameras={camera_names}, '
        f'reference={camera_names[idx_reference]}, translation scale={scale:.3f}'
    )
    return c2w, fxfycxcy, idx_reference
