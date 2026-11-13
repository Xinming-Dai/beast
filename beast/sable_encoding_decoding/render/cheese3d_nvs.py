"""Novel-view synthesis of a Cheese3D scene onto ground-truth cameras, including unseen ones.

A Sable model trained on a few Cheese3D cameras (e.g. TL/TR with or without TC) predicts a
Gaussian point cloud plus its own camera poses in an arbitrary canonical frame. This module
aligns the predicted poses to the ground-truth (GT) calibration with a similarity transform
(the same alignment ``save_camera_pointcloud_scene`` uses for its .glb overlay), then renders
the point cloud from the GT cameras, including cameras the model never saw (e.g. BC).

For one session/frame it saves a 3-row figure (predicted-camera renders, renders from the
aligned GT cameras, GT images) plus whole-image PSNR/SSIM (no segmentation mask) as JSON.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from beast.logging import log_step

_DEFAULT_CAMERAS = ('TL', 'TR', 'TC', 'BC')
_RENDER_CHECK_ATOL = 1e-2


def resolve_model_cameras(training: dict) -> list[str]:
    """Return the ordered camera names a Cheese3D model was trained on.

    Mirrors the camera resolution in ``Cheese3DDataset.__init__``: ``cheese3d_cameras`` wins,
    otherwise left, right, then the optional center camera.

    Args:
        training: the ``training`` section of the model config.

    Returns:
        camera names in view-index order.
    """
    cameras_raw = training.get('cheese3d_cameras')
    if cameras_raw:
        return [str(camera) for camera in cameras_raw]
    cameras = [
        str(training.get('cheese3d_left_camera', 'TL')),
        str(training.get('cheese3d_right_camera', 'TR')),
    ]
    if training.get('cheese3d_center_camera'):
        cameras.append(str(training['cheese3d_center_camera']))
    return cameras


def invert_similarity_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 similarity transform ``[sR | t]``.

    Args:
        transform: 4x4 similarity transform with an isotropic scale in its rotation block.

    Returns:
        4x4 similarity transform ``[R^T / s | -R^T t / s]``.
    """
    sr = np.asarray(transform, dtype=float)[:3, :3]
    scale = float(np.cbrt(np.linalg.det(sr)))
    rot = sr / scale
    inverse = np.eye(4)
    inverse[:3, :3] = rot.T / scale
    inverse[:3, 3] = -(rot.T @ np.asarray(transform, dtype=float)[:3, 3]) / scale
    return inverse


def map_gt_poses_to_model_frame(transform: np.ndarray, gt_c2w: np.ndarray) -> np.ndarray:
    """Express GT camera poses in the model's canonical frame.

    ``transform`` maps predicted poses onto GT poses, so applying its inverse to the GT poses
    is equivalent to moving the point cloud into the GT frame, without touching the Gaussians.

    Args:
        transform: 4x4 pred-to-GT similarity transform.
        gt_c2w: GT camera-to-world matrices, shape [V, 4, 4].

    Returns:
        GT camera-to-world matrices in the model frame, shape [V, 4, 4].
    """
    from beast.models.model_utils.utils_icp import apply_similarity_transform_to_poses

    return apply_similarity_transform_to_poses(invert_similarity_transform(transform), gt_c2w)


def pose_residuals(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> dict[str, np.ndarray]:
    """Per-camera pose error between aligned predicted and GT camera-to-world matrices.

    Args:
        pred_c2w: predicted poses already moved into the GT frame, shape [V, 4, 4].
        gt_c2w: GT poses, shape [V, 4, 4].

    Returns:
        dict with ``center_error`` ([V], camera-center distance in GT units) and
        ``rotation_error_deg`` ([V], geodesic angle between the two orientations).
    """
    pred_c2w = np.asarray(pred_c2w, dtype=float)
    gt_c2w = np.asarray(gt_c2w, dtype=float)
    center_error = np.linalg.norm(pred_c2w[:, :3, 3] - gt_c2w[:, :3, 3], axis=-1)
    relative = np.einsum('vji,vjk->vik', pred_c2w[:, :3, :3], gt_c2w[:, :3, :3])
    cos_angle = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    rotation_error_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return {'center_error': center_error, 'rotation_error_deg': rotation_error_deg}


def build_render_intrinsics(
    mode: str,
    cameras: list[str],
    gt_px: np.ndarray,
    pred_px: dict[str, np.ndarray],
    fit_cameras: list[str],
) -> np.ndarray:
    """Pick the pixel-unit intrinsics used to render each GT camera.

    Args:
        mode: ``'gt'`` uses the GT calibration for every camera. ``'pred_scaled'`` uses the
            model's predicted intrinsics for cameras it has, and for the others scales the GT
            focal lengths by the mean predicted/GT focal ratio over ``fit_cameras`` (the model's
            focal scale need not match the calibration's), keeping the GT principal point.
        cameras: camera names, matching the rows of ``gt_px``.
        gt_px: GT ``[fx, fy, cx, cy]`` in pixels, shape [V, 4].
        pred_px: predicted ``[fx, fy, cx, cy]`` in pixels by camera name.
        fit_cameras: cameras used for the pose alignment (must be in ``pred_px`` for
            ``'pred_scaled'``).

    Returns:
        pixel-unit intrinsics, shape [V, 4].

    Raises:
        ValueError: if ``mode`` is unknown.
    """
    gt_px = np.asarray(gt_px, dtype=np.float64)
    if mode == 'gt':
        return gt_px.copy()
    if mode != 'pred_scaled':
        raise ValueError(f"intrinsics mode must be 'gt' or 'pred_scaled', got {mode!r}")
    ratios = np.mean(
        [pred_px[camera][:2] / gt_px[cameras.index(camera), :2] for camera in fit_cameras],
        axis=0,
    )
    out = gt_px.copy()
    for idx, camera in enumerate(cameras):
        if camera in pred_px:
            out[idx] = pred_px[camera]
        else:
            out[idx, :2] = gt_px[idx, :2] * ratios
    return out


def estimate_scale_from_pairs(pred_c2w: np.ndarray, gt_c2w: np.ndarray) -> float:
    """GT-units-per-model-unit from pairwise camera-center distances.

    Unlike the similarity fit, this makes no assumption that the predicted rig is a rotated
    copy of the GT rig; it only compares how far apart the cameras are.

    Args:
        pred_c2w: predicted camera-to-world matrices, shape [V, 4, 4], V >= 2.
        gt_c2w: matching GT camera-to-world matrices, shape [V, 4, 4].

    Returns:
        mean over camera pairs of ``|gt_i - gt_j| / |pred_i - pred_j|``.

    Raises:
        ValueError: if no pair of predicted cameras is separated.
    """
    pred_centers = np.asarray(pred_c2w, dtype=float)[:, :3, 3]
    gt_centers = np.asarray(gt_c2w, dtype=float)[:, :3, 3]
    ratios = []
    for i, j in itertools.combinations(range(len(pred_centers)), 2):
        pred_dist = float(np.linalg.norm(pred_centers[i] - pred_centers[j]))
        if pred_dist > 1e-12:
            ratios.append(float(np.linalg.norm(gt_centers[i] - gt_centers[j])) / pred_dist)
    if not ratios:
        raise ValueError('need at least two separated cameras to estimate scale')
    return float(np.mean(ratios))


def place_camera_relative(
    anchor_pred_c2w: np.ndarray,
    anchor_gt_c2w: np.ndarray,
    target_gt_c2w: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Place a camera in the model frame from its GT pose relative to an anchor camera.

    The GT rig fixes where ``target`` sits in the anchor's own frame; the same relative
    rotation is applied to the anchor's predicted pose, and the relative translation is
    converted from GT units to model units with ``scale``.

    Args:
        anchor_pred_c2w: anchor camera-to-world in the model frame, shape [4, 4].
        anchor_gt_c2w: the anchor's GT camera-to-world, shape [4, 4].
        target_gt_c2w: the target camera's GT camera-to-world, shape [4, 4].
        scale: GT units per model unit.

    Returns:
        the target's camera-to-world in the model frame, shape [4, 4].
    """
    relative = np.linalg.inv(np.asarray(anchor_gt_c2w, dtype=float)) @ np.asarray(
        target_gt_c2w, dtype=float,
    )
    relative[:3, 3] /= scale
    return np.asarray(anchor_pred_c2w, dtype=float) @ relative


def place_cameras_anchored(
    cameras: list[str],
    pred_c2w_by_camera: dict[str, np.ndarray],
    pred_px_by_camera: dict[str, np.ndarray],
    gt_c2w: np.ndarray,
    gt_px: np.ndarray,
    anchor_camera: str,
    anchor_pred_c2w: np.ndarray,
    scale: float,
    focal_ratio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Poses and pixel intrinsics to render every camera, anchored on one camera.

    Cameras the model has keep their predicted pose and intrinsics. The anchor (if the model
    has no pose for it) takes ``anchor_pred_c2w``; every other camera is placed relative to
    the anchor with ``place_camera_relative``. Cameras without a prediction get the GT focal
    lengths multiplied by ``focal_ratio`` so they stay consistent with the model's own
    (shared, pseudo-) focal length.

    Args:
        cameras: camera names, matching the rows of ``gt_c2w``/``gt_px``.
        pred_c2w_by_camera: predicted camera-to-world by camera name.
        pred_px_by_camera: predicted ``[fx, fy, cx, cy]`` in pixels by camera name.
        gt_c2w: GT camera-to-world, shape [V, 4, 4].
        gt_px: GT ``[fx, fy, cx, cy]`` in pixels, shape [V, 4].
        anchor_camera: the camera others are placed relative to.
        anchor_pred_c2w: the anchor's pose in the model frame.
        scale: GT units per model unit for relative translations.
        focal_ratio: ``[fx, fy]`` factor mapping GT focal lengths to the model's.

    Returns:
        tuple of (poses [V, 4, 4], intrinsics [V, 4], ``{camera: source label}``) where the
        label is ``'predicted'``, ``'anchor'`` or ``'placed_from_<anchor>'``.
    """
    anchor_idx = cameras.index(anchor_camera)
    poses, intrinsics, sources = [], [], {}
    for idx, camera in enumerate(cameras):
        if camera in pred_c2w_by_camera:
            poses.append(np.asarray(pred_c2w_by_camera[camera], dtype=float))
            intrinsics.append(np.asarray(pred_px_by_camera[camera], dtype=float))
            sources[camera] = 'predicted'
            continue
        if camera == anchor_camera:
            poses.append(np.asarray(anchor_pred_c2w, dtype=float))
            sources[camera] = 'anchor'
        else:
            poses.append(
                place_camera_relative(anchor_pred_c2w, gt_c2w[anchor_idx], gt_c2w[idx], scale),
            )
            sources[camera] = f'placed_from_{anchor_camera}'
        px = np.asarray(gt_px[idx], dtype=float).copy()
        px[:2] *= focal_ratio
        intrinsics.append(px)
    return np.stack(poses), np.stack(intrinsics), sources


def held_out_variants(training_regime: str, choice: str) -> list[tuple[str | None, bool]]:
    """Which held-out-view variants to run for a model.

    Args:
        training_regime: ``training.training_regime`` of the model config.
        choice: ``'both'``, ``'visible'`` (held-out view left visible at inference, which is
            what validation did in training) or ``'masked'`` (strict holdout: its image tokens
            zeroed and its Gaussians dropped, so it is rendered from the other views only).

    Returns:
        list of ``(variant name, mask_held_out)``. Models without a held-out view run once with
        variant name ``None``, because the variants would be identical.

    Raises:
        ValueError: if ``choice`` is unknown.
    """
    if choice not in ('both', 'visible', 'masked'):
        raise ValueError(f'held-out TC choice must be both/visible/masked, got {choice!r}')
    if str(training_regime).strip().lower() != 'center_camera_holdout':
        return [(None, False)]
    variants = {'visible': ('tc_visible', False), 'masked': ('tc_masked', True)}
    names = ('visible', 'masked') if choice == 'both' else (choice,)
    return [variants[name] for name in names]


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a ``[3, H, W]`` float image in [0, 1] to an ``[H, W, 3]`` uint8 array."""
    array = image.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (array * 255.0).round().astype(np.uint8)


def format_metric_table(
    cameras: list[str],
    rows: dict[str, dict[str, dict[str, float]]],
) -> str:
    """Format per-camera PSNR/SSIM as a fixed-width text table.

    Args:
        cameras: column order.
        rows: ``{row_name: {camera: {'psnr': float, 'ssim': float}}}``; cameras missing from a
            row are printed as ``-``.

    Returns:
        the table as a multi-line string.
    """
    lines = []
    for row_name, by_camera in rows.items():
        cells = [
            f'{camera}: {by_camera[camera]["psnr"]:6.2f} dB / {by_camera[camera]["ssim"]:.3f}'
            if camera in by_camera else f'{camera}: -'
            for camera in cameras
        ]
        lines.append(f'{row_name:<22}' + '   '.join(cells))
    return '\n'.join(lines)


def build_grid_figure(
    rows: list[list[np.ndarray | None]],
    column_labels: list[str],
    row_labels: list[str],
    out_path: Path,
) -> Path:
    """Save a grid of images, drawing an ``N/A`` tile wherever an entry is ``None``.

    Args:
        rows: one list per row, one uint8 ``[H, W, 3]`` image (or ``None``) per column.
        column_labels: one title per column.
        row_labels: one label per row.
        out_path: output PNG path.

    Returns:
        ``out_path``.
    """
    num_rows, num_cols = len(rows), len(column_labels)
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3.2 * num_cols, 3.2 * num_rows), squeeze=False,
    )
    for row_idx, row in enumerate(rows):
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            image = row[col_idx]
            if image is None:
                ax.set_facecolor('0.9')
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='0.4')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(column_labels[col_idx], fontsize=14)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _load_gt_cameras(
    dataset,
    rec,
    cameras: list[str],
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Load GT images and calibration for ``cameras`` from sibling directories of ``rec``.

    Args:
        dataset: ``Cheese3DDataset`` (used for its image loader/resizing).
        rec: record whose left-camera path locates the session directory and frame filename.
        cameras: camera directory names, in output order.

    Returns:
        tuple of (images ``[V, 3, H, W]`` in [0, 1], c2w ``[V, 4, 4]``, normalised fxfycxcy
        ``[V, 4]``).
    """
    from beast.data.sable_dataset import _npy_to_c2w_fxfycxcy

    session_dir = rec.left_path.parent.parent
    images, c2ws, intrinsics = [], [], []
    for camera in cameras:
        image_path = session_dir / camera / rec.left_path.name
        image, _, _ = dataset._load_image(image_path)
        c2w, fxfycxcy = _npy_to_c2w_fxfycxcy(image_path.with_suffix('.npy'))
        images.append(image)
        c2ws.append(c2w)
        intrinsics.append(fxfycxcy)
    return torch.stack(images), np.stack(c2ws), np.stack(intrinsics)


def _select_record(dataset, session: str, frame_idx: int | None):
    """Return ``(dataset_idx, record)`` for the requested session and frame.

    Args:
        dataset: ``Cheese3DDataset``.
        session: session directory name.
        frame_idx: source frame index, or ``None`` for the session's first record.

    Returns:
        tuple of (index into the dataset, its record).

    Raises:
        ValueError: if no record matches.
    """
    for idx, rec in enumerate(dataset._records):
        if rec.session_id != session:
            continue
        if frame_idx is None or rec.left_source_frame_index == frame_idx:
            return idx, rec
    raise ValueError(f'no record for session {session!r} with frame_idx={frame_idx}')


def _render_at_cameras(
    gaussians,
    c2w: torch.Tensor,
    fxfycxcy_px: torch.Tensor,
    height: int,
    width: int,
    near_plane: float,
) -> torch.Tensor:
    """Render one Gaussian model from several cameras.

    Args:
        gaussians: ``GaussianModel`` from the model outputs.
        c2w: camera-to-world matrices ``[V, 4, 4]`` in the Gaussians' frame.
        fxfycxcy_px: pixel-unit intrinsics ``[V, 4]``.
        height: render height.
        width: render width.
        near_plane: near clipping plane, as used by the model's renderer.

    Returns:
        renders ``[V, 3, H, W]`` in [0, 1].
    """
    from beast.models.model_utils.gaussians_renderer import render_opencv_cam_gsplat

    with torch.no_grad():
        buffers = render_opencv_cam_gsplat(
            gaussians, height, width, c2w, fxfycxcy_px,
            sh_degree=gaussians.sh_degree, near_plane=near_plane,
        )
    return buffers['render'].clamp(0.0, 1.0)


def _metrics_by_camera(
    render: torch.Tensor,
    target: torch.Tensor,
    cameras: list[str],
) -> dict[str, dict[str, float]]:
    """Whole-image PSNR/SSIM per camera.

    Args:
        render: renders ``[V, 3, H, W]`` in [0, 1].
        target: GT images ``[V, 3, H, W]`` in [0, 1].
        cameras: one name per view.

    Returns:
        ``{camera: {'psnr': float, 'ssim': float}}``.
    """
    from beast.sable_encoding_decoding.render.metrics import _image_metrics_by_view

    psnr, ssim = _image_metrics_by_view(render[None], target[None])
    return {
        camera: {'psnr': float(psnr[0, idx]), 'ssim': float(ssim[0, idx])}
        for idx, camera in enumerate(cameras)
    }


def _mean_metrics(by_camera: dict, names: list[str]) -> dict[str, float]:
    """Mean PSNR/SSIM over ``names`` (those present in ``by_camera``); NaN when none are."""
    picked = [by_camera[name] for name in names if name in by_camera]
    return {
        metric: float(np.mean([m[metric] for m in picked])) if picked else float('nan')
        for metric in ('psnr', 'ssim')
    }


def _run_nvs_session(
    model,
    config: dict,
    dataset,
    model_dir: Path,
    output_dir: Path,
    session: str,
    frame_idx: int | None,
    cameras: list[str],
    align_cameras: list[str] | None,
    device: str,
    amp: str = 'none',
    context_batch: int = 1,
    intrinsics: str = 'gt',
    save_glb: bool = True,
    pose_mode: str = 'anchor',
    anchor_camera: str = 'TC',
    anchor_source: str = 'auto',
    mask_held_out: bool = False,
    variant: str | None = None,
    hide_held_out_from_encoders: bool = False,
) -> dict:
    """Run NVS for one frame of one session and save the figure, metrics and scenes.

    Two ways of choosing the cameras to render from are always evaluated; ``pose_mode`` picks
    which one fills row 2 of the figure:

    - ``global_fit``: every GT camera is mapped into the model frame with one similarity
      transform fitted on ``align_cameras``.
    - ``anchor``: cameras the model has keep their predicted pose; the rest are placed relative
      to ``anchor_camera`` using the GT rig geometry (``place_cameras_anchored``).

    Args:
        model: trained Sable model, already on ``device`` in eval mode.
        config: inference config (``training.session_names`` covers ``session``).
        dataset: ``Cheese3DDataset`` built from ``config``.
        model_dir: model directory, recorded in the metrics.
        output_dir: session directory; results go to ``output_dir / frame_<idx:08d>`` with
            ``nvs_grid.png``, per-tile PNGs, ``nvs_metrics.json`` and ``glb/``.
        session: session directory name.
        frame_idx: source frame index, or ``None`` for the session's first frame in the split.
        cameras: GT cameras to render/score, in column order.
        align_cameras: cameras used to fit the pred-to-GT transform. Defaults to every camera
            the model took as input, excluding views masked from the encoders (center holdout).
        device: torch device string.
        amp: ``'bf16'`` runs the forward under bf16 autocast, as training does; ``'none'``
            runs plain fp32.
        context_batch: number of consecutive dataset records collated into the forward batch
            (the requested frame is sample 0). Training validates with batches of 16.
        intrinsics: intrinsics for the ``global_fit`` renders, see ``build_render_intrinsics``.
            Metrics for both modes of it are written.
        save_glb: whether to save the point cloud + predicted/placed camera frustum scenes.
        pose_mode: ``'anchor'`` or ``'global_fit'``; which renders fill row 2.
        anchor_camera: camera the ``anchor`` mode places the others relative to.
        anchor_source: ``'predicted'`` uses the model's own pose for the anchor, ``'derived'``
            maps its GT pose into the model frame with the similarity fit, ``'auto'`` picks
            ``'predicted'`` when the model has a pose for it.
        mask_held_out: strict holdout at inference (the ``strict_holdout`` batch key read by
            ``Sable.forward``): the held-out view's image tokens are zeroed and its Gaussians are
            dropped. Only meaningful for ``center_camera_holdout`` models.
        variant: subfolder name for this run's outputs under the frame folder, or ``None``.
        hide_held_out_from_encoders: with ``mask_held_out``, also hide the held-out image from
            the pose encoder and DINO (``hide_held_out_from_encoders`` batch key). Only sensible
            for models trained with ``training.strict_holdout_hide_encoders``.

    Returns:
        the metrics dict that is also written to ``nvs_metrics.json``.

    Raises:
        ValueError: if no record matches, too few cameras remain to align, or the anchor
            settings cannot be satisfied.
    """
    from beast.data.sable_dataset import collate_with_correspondence_padding
    from beast.inference import _dedupe_cameras_by_view_index, save_camera_pointcloud_scene
    from beast.models.model_utils.utils_icp import (
        apply_similarity_transform_to_poses,
        estimate_camera_similarity_transform,
    )

    model_cameras = resolve_model_cameras(config['training'])
    dataset_idx, rec = _select_record(dataset, session, frame_idx)
    batch_indices = range(dataset_idx, min(dataset_idx + context_batch, len(dataset)))
    batch = collate_with_correspondence_padding([dataset[idx] for idx in batch_indices])
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    if mask_held_out:
        batch['strict_holdout'] = True
        if hide_held_out_from_encoders:
            batch['hide_held_out_from_encoders'] = True
    source_frame_idx = int(rec.left_source_frame_index)
    output_dir = Path(output_dir) / f'frame_{source_frame_idx:08d}'
    if variant is not None:
        output_dir = output_dir / variant
    log_step(
        f'cheese3d_nvs: {rec.scene_name}, model cameras {model_cameras}, amp={amp}, '
        f'batch={len(batch_indices)}, strict holdout={mask_held_out}',
        level='info',
    )

    # GT images/poses come from the camera directories directly so BC never enters the model
    gt_images, gt_c2w, gt_fxfycxcy = _load_gt_cameras(dataset, rec, cameras)

    with torch.no_grad(), torch.autocast(
        device_type='cuda', dtype=torch.bfloat16, enabled=amp == 'bf16' and device == 'cuda',
    ):
        result = model.get_model_outputs(batch)

    input_idx = result['input_indices'][0].cpu()
    target_idx = result['target_indices'][0].cpu()
    view_indices, pred_c2w = _dedupe_cameras_by_view_index(
        input_idx,
        result['c2w_input'][0].detach().float().cpu().numpy(),
        target_idx,
        result['c2w_target'][0].detach().float().cpu().numpy(),
    )
    pred_c2w_by_camera = {
        model_cameras[view_idx]: pose for view_idx, pose in zip(view_indices, pred_c2w)
    }

    # views masked from every encoder (center holdout) carry no real pose information
    held_out = set()
    full_mask = batch.get('context_full_mask')
    if full_mask is not None:
        held_out = {
            model_cameras[view_idx] for view_idx in input_idx[full_mask[0].cpu()].tolist()
        }
    fit_cameras = align_cameras or [
        model_cameras[view_idx] for view_idx in input_idx.unique().tolist()
        if model_cameras[view_idx] not in held_out
    ]
    if len(fit_cameras) < 2:
        raise ValueError(f'need at least 2 cameras to align, got {fit_cameras}')
    novel_cameras = [camera for camera in cameras if camera not in fit_cameras]

    fit_pred_c2w = np.stack([pred_c2w_by_camera[camera] for camera in fit_cameras])
    fit_gt_c2w = np.stack([gt_c2w[cameras.index(camera)] for camera in fit_cameras])
    transform = estimate_camera_similarity_transform(fit_pred_c2w, fit_gt_c2w)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    log_step(
        f'cheese3d_nvs: aligned on {fit_cameras} (scale={scale:.4f}); novel: {novel_cameras}',
        level='info',
    )

    # pose error of every predicted camera after alignment, fit cameras and others alike
    pose_cameras = list(pred_c2w_by_camera)
    aligned_pred = apply_similarity_transform_to_poses(
        transform, np.stack([pred_c2w_by_camera[camera] for camera in pose_cameras]),
    )
    residuals = pose_residuals(
        aligned_pred, np.stack([gt_c2w[cameras.index(camera)] for camera in pose_cameras]),
    )
    pose_error = {
        camera: {
            'in_fit': camera in fit_cameras,
            'center_error': float(residuals['center_error'][idx]),
            'rotation_error_deg': float(residuals['rotation_error_deg'][idx]),
        }
        for idx, camera in enumerate(pose_cameras)
    }

    gaussians = result['gaussians'][0]
    height, width = result['render'].shape[-2:]
    near_plane = config['model'].get('near_plane', 0.2)
    pixel_scale = np.array([width, height, width, height], dtype=np.float64)
    gt_px = gt_fxfycxcy * pixel_scale
    # fxfycxcy_target is already in pixels (SABLE.forward rescales it before storing)
    pred_px = {
        model_cameras[view_idx]: result['fxfycxcy_target'][0, pos].detach().float().cpu().numpy()
        for pos, view_idx in enumerate(target_idx.tolist())
    }

    # global_fit: every GT camera mapped into the model frame by the one similarity fit
    global_c2w = map_gt_poses_to_model_frame(transform, gt_c2w)
    global_px = build_render_intrinsics(intrinsics, cameras, gt_px, pred_px, fit_cameras)

    # anchor: predicted poses where the model has them, the rest placed relative to one camera
    if anchor_camera not in cameras:
        raise ValueError(f'anchor camera {anchor_camera!r} must be one of {cameras}')
    if anchor_source == 'predicted' and anchor_camera not in pred_c2w_by_camera:
        raise ValueError(f'the model has no predicted pose for anchor camera {anchor_camera!r}')
    anchor_idx = cameras.index(anchor_camera)
    if anchor_camera in pred_c2w_by_camera and anchor_source in ('auto', 'predicted'):
        anchor_used = 'predicted'
        anchor_pred = pred_c2w_by_camera[anchor_camera]
        anchor_scale = estimate_scale_from_pairs(fit_pred_c2w, fit_gt_c2w)
        focal_ratio = pred_px[anchor_camera][:2] / gt_px[anchor_idx, :2]
        if anchor_camera in held_out:
            log_step(
                f'cheese3d_nvs: anchor {anchor_camera} is this model\'s held-out view; compare '
                'anchor_source=derived if its predicted pose looks off',
                level='warning',
            )
    else:
        anchor_used = 'derived'
        anchor_pred = global_c2w[anchor_idx]
        anchor_scale = scale
        focal_ratio = np.mean(
            [pred_px[c][:2] / gt_px[cameras.index(c), :2] for c in fit_cameras], axis=0,
        )
    anchor_c2w, anchor_px, pose_source = place_cameras_anchored(
        cameras, pred_c2w_by_camera, pred_px, gt_c2w, gt_px,
        anchor_camera, anchor_pred, anchor_scale, focal_ratio,
    )
    log_step(
        f'cheese3d_nvs: anchor {anchor_camera} ({anchor_used}, scale={anchor_scale:.4f}); '
        f'pose sources {pose_source}',
        level='info',
    )

    gt_images = gt_images.to(device)
    render_by_mode, projected_by_mode = {}, {}
    for mode, (mode_c2w, mode_px) in {
        'global_fit': (global_c2w, global_px),
        'anchor': (anchor_c2w, anchor_px),
    }.items():
        render_by_mode[mode] = _render_at_cameras(
            gaussians,
            torch.from_numpy(mode_c2w).float().to(device),
            torch.from_numpy(mode_px).float().to(device),
            height, width, near_plane,
        )
        projected_by_mode[mode] = _metrics_by_camera(render_by_mode[mode], gt_images, cameras)
    # global_fit under the other intrinsics setting, for comparison
    other_intrinsics = 'pred_scaled' if intrinsics == 'gt' else 'gt'
    other_render = _render_at_cameras(
        gaussians,
        torch.from_numpy(global_c2w).float().to(device),
        torch.from_numpy(
            build_render_intrinsics(other_intrinsics, cameras, gt_px, pred_px, fit_cameras),
        ).float().to(device),
        height, width, near_plane,
    )
    global_fit_by_intrinsics = {
        intrinsics: projected_by_mode['global_fit'],
        other_intrinsics: _metrics_by_camera(other_render, gt_images, cameras),
    }
    nvs_render = render_by_mode[pose_mode]
    projected = projected_by_mode[pose_mode]

    # row 1: the model's own renders at its predicted cameras
    pred_render = {
        model_cameras[view_idx]: result['render'][0, pos].clamp(0.0, 1.0)
        for pos, view_idx in enumerate(target_idx.tolist())
    }

    # the manual render path must reproduce the model's render at its own predicted cameras
    check_cameras = list(pred_render)
    check_c2w = torch.from_numpy(
        np.stack([pred_c2w_by_camera[camera] for camera in check_cameras]),
    ).float().to(device)
    check_render = _render_at_cameras(
        gaussians, check_c2w, result['fxfycxcy_target'][0].float(), height, width, near_plane,
    )
    max_diff = float((check_render - torch.stack(list(pred_render.values()))).abs().max())
    log_step(
        f'cheese3d_nvs: render-path check max |diff| = {max_diff:.5f}',
        level='warning' if max_diff > _RENDER_CHECK_ATOL else 'info',
    )

    predicted = _metrics_by_camera(
        torch.stack(list(pred_render.values())),
        torch.stack([gt_images[cameras.index(camera)] for camera in pred_render]),
        list(pred_render),
    )

    metrics = {
        'model_dir': str(model_dir),
        'session': session,
        'frame_idx': source_frame_idx,
        'scene_name': rec.scene_name,
        'amp': amp,
        'held_out_tc': (
            'none' if variant is None else ('masked' if mask_held_out else 'visible')
        ),
        'context_batch': len(batch_indices),
        'pose_mode': pose_mode,
        'global_fit_intrinsics': intrinsics,
        'anchor_camera': anchor_camera,
        'anchor_source_requested': anchor_source,
        'anchor_source_used': anchor_used,
        'anchor_scale': anchor_scale,
        'pose_source': pose_source,
        'cameras': cameras,
        'model_cameras': model_cameras,
        'fit_cameras': fit_cameras,
        'novel_cameras': novel_cameras,
        'similarity_scale': scale,
        'render_path_max_abs_diff': max_diff,
        'predicted_intrinsics_px': {c: v.tolist() for c, v in pred_px.items()},
        'gt_intrinsics_px': {c: gt_px[idx].tolist() for idx, c in enumerate(cameras)},
        'pose_error_after_alignment': pose_error,
        'predicted_views': predicted,
        'projected_to_gt': projected,
        'projected_by_pose_mode': projected_by_mode,
        'global_fit_by_intrinsics': global_fit_by_intrinsics,
        'summary': {
            'predicted_mean': _mean_metrics(predicted, cameras),
            'projected_mean_all': _mean_metrics(projected, cameras),
            'projected_mean_novel': _mean_metrics(projected, novel_cameras),
            'projected_mean_novel_by_pose_mode': {
                mode: _mean_metrics(by_camera, novel_cameras)
                for mode, by_camera in projected_by_mode.items()
            },
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        [tensor_to_uint8(pred_render[c]) if c in pred_render else None for c in cameras],
        [tensor_to_uint8(nvs_render[idx]) for idx in range(len(cameras))],
        [tensor_to_uint8(gt_images[idx]) for idx in range(len(cameras))],
    ]
    row_labels = ['predicted views', f'projected ({pose_mode})', 'ground truth']
    build_grid_figure(rows, cameras, row_labels, output_dir / 'nvs_grid.png')
    for row_label, row in zip(('predicted', 'projected', 'gt'), rows):
        for camera, image in zip(cameras, row):
            if image is not None:
                Image.fromarray(image).save(output_dir / f'{row_label}_{camera}.png')
    (output_dir / 'nvs_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    if save_glb:
        # global_fit scene: predicted cameras and every GT camera in the GT frame. View index
        # == position in model_cameras, so order the GT rows the same way and append the
        # cameras the model does not have (e.g. BC) so their frusta are drawn too.
        glb_cameras = model_cameras + [c for c in cameras if c not in model_cameras]
        glb_gt_c2w = np.stack([gt_c2w[cameras.index(camera)] for camera in glb_cameras])
        glb_paths = save_camera_pointcloud_scene(
            {**result, 'gt_c2w': torch.from_numpy(glb_gt_c2w).float()[None]},
            output_dir,
            batch_idx=0,
            sample_indices=[0],
            scene_names=[rec.scene_name],
            align_view_indices=[model_cameras.index(camera) for camera in fit_cameras],
            file_suffix='_global_fit',
        )
        log_step(f'cheese3d_nvs: GT camera order in scene: {glb_cameras}', level='info')
        # anchor scene: model frame, predicted cameras (colored) and the cameras placed
        # relative to the anchor (black)
        placed = [idx for idx, c in enumerate(cameras) if c not in pred_c2w_by_camera]
        if placed:
            glb_paths += save_camera_pointcloud_scene(
                result,
                output_dir,
                batch_idx=0,
                sample_indices=[0],
                scene_names=[rec.scene_name],
                extra_c2w=anchor_c2w[placed],
                file_suffix='_anchor',
            )
            log_step(
                f'cheese3d_nvs: anchor scene draws placed cameras {[cameras[i] for i in placed]}',
                level='info',
            )
        metrics['glb_files'] = [str(path) for path in glb_paths]
        (output_dir / 'nvs_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    log_step(
        'cheese3d_nvs: whole-image PSNR / SSIM\n'
        + format_metric_table(
            cameras,
            {
                'predicted views': predicted,
                'global_fit': projected_by_mode['global_fit'],
                'anchor': projected_by_mode['anchor'],
            },
        )
        + '\ncamera pose error after alignment (GT units / deg): '
        + ', '.join(
            f'{c}{"*" if e["in_fit"] else ""}: {e["center_error"]:.1f} / '
            f'{e["rotation_error_deg"]:.1f}'
            for c, e in pose_error.items()
        )
        + ' (* = in fit)'
        + f'\nmean over novel cams {novel_cameras}: '
        + str(metrics['summary']['projected_mean_novel_by_pose_mode']),
        level='info',
    )
    log_step(f'cheese3d_nvs: saved outputs to {output_dir}', level='info')
    return metrics


def run_nvs(
    model_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    sessions: list[str] | None = None,
    frame_idx: int | None = None,
    split: str = 'val',
    cameras: tuple[str, ...] = _DEFAULT_CAMERAS,
    align_cameras: list[str] | None = None,
    amp: str = 'none',
    context_batch: int = 1,
    intrinsics: str = 'gt',
    save_glb: bool = True,
    pose_mode: str = 'anchor',
    anchor_camera: str = 'TC',
    anchor_source: str = 'auto',
    held_out_tc: str = 'both',
    hide_held_out_from_encoders: bool = False,
) -> dict[str, dict[str, dict]]:
    """Run NVS for one frame of each session, loading the model once.

    The dataset always covers every session in the saved training config (in config order) so
    ``context_batch`` collates the same neighbours a training validation batch would; ``sessions``
    only selects which sessions are evaluated. Each session's outputs go to
    ``output_dir / session / frame_<idx:08d>`` (``nvs_grid.png``, per-tile PNGs,
    ``nvs_metrics.json``, ``glb/``);
    ``output_dir / 'nvs_metrics_all_sessions.json'`` collects every session's metrics plus
    their mean.

    Args:
        model_dir: directory with ``config.yaml`` and the best checkpoint.
        dataset_path: root Cheese3D directory.
        output_dir: root output directory.
        sessions: session directory names to evaluate. ``None`` uses every session in the
            saved training config (duplicates removed).
        frame_idx: source frame index used for every session, or ``None`` for each session's
            first frame in ``split``.
        split: dataset split to draw frames from.
        cameras: GT cameras to render/score, in column order.
        align_cameras: cameras used to fit the pred-to-GT transform; see
            ``_run_nvs_session``.
        amp: ``'none'`` (fp32) or ``'bf16'`` (autocast), see ``_run_nvs_session``.
        context_batch: records collated per forward pass, see ``_run_nvs_session``.
        intrinsics: ``global_fit`` intrinsics mode, see ``build_render_intrinsics``.
        save_glb: whether to save the camera/point-cloud scenes per session.
        pose_mode: ``'anchor'`` or ``'global_fit'``; which renders fill row 2.
        anchor_camera: camera the ``anchor`` mode places the others relative to.
        anchor_source: ``'auto'``, ``'predicted'`` or ``'derived'``; see ``_run_nvs_session``.
        held_out_tc: ``'both'``, ``'visible'`` or ``'masked'`` for ``center_camera_holdout``
            models, see ``held_out_variants``. Each variant is saved under
            ``frame_<idx:08d>/<variant>/`` with its own
            ``nvs_metrics_all_sessions_<variant>.json``; other models run once.
        hide_held_out_from_encoders: for the masked variant, also hide the held-out image from
            the pose encoder and DINO; see ``_run_nvs_session``.

    Returns:
        ``{variant name or 'default': {session: metrics dict}}`` for the sessions that had a
        frame in ``split``.

    Raises:
        ValueError: if a model camera is missing from ``cameras``.
        RuntimeError: if no session produced a result.
    """
    from beast.api.model import Model
    from beast.inference import _build_sable_dataset

    cameras = list(cameras)
    wrapper = Model.from_dir(model_dir)
    config = {**wrapper.config}
    configured = config['training'].get('session_names') or []
    configured = [configured] if isinstance(configured, str) else list(configured)
    dataset_sessions = list(dict.fromkeys(configured + list(sessions or [])))
    if sessions is None:
        sessions = list(dict.fromkeys(configured))
    config['inference'] = True
    config['training'] = {
        **config['training'],
        'dataset_path': str(dataset_path),
        'session_names': dataset_sessions,
    }
    model_cameras = resolve_model_cameras(config['training'])
    missing = [camera for camera in model_cameras if camera not in cameras]
    if missing:
        raise ValueError(f'model cameras {missing} must be included in cameras {cameras}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = wrapper.model.to(device).eval()
    dataset = _build_sable_dataset(config, include_splits=[split])

    output_dir = Path(output_dir)
    variants = held_out_variants(config['training'].get('training_regime', ''), held_out_tc)
    results: dict[str, dict[str, dict]] = {}
    for variant, mask_held_out in variants:
        all_metrics: dict[str, dict] = {}
        for session in sessions:
            try:
                all_metrics[session] = _run_nvs_session(
                    model, config, dataset, model_dir, output_dir / session, session,
                    frame_idx, cameras, align_cameras, device,
                    amp=amp, context_batch=context_batch, intrinsics=intrinsics,
                    save_glb=save_glb, pose_mode=pose_mode, anchor_camera=anchor_camera,
                    anchor_source=anchor_source, mask_held_out=mask_held_out, variant=variant,
                    hide_held_out_from_encoders=hide_held_out_from_encoders,
                )
            except ValueError as err:
                log_step(f'cheese3d_nvs: skipping session {session}: {err}', level='warning')

        if not all_metrics:
            raise RuntimeError(f'no session produced an NVS result; tried {sessions}')

        summary = {
            key: {
                metric: float(np.nanmean([
                    m['summary'][key][metric] for m in all_metrics.values()
                ]))
                for metric in ('psnr', 'ssim')
            }
            for key in ('predicted_mean', 'projected_mean_all', 'projected_mean_novel')
        }
        summary['projected_mean_novel_by_pose_mode'] = {
            mode: {
                metric: float(np.nanmean([
                    m['summary']['projected_mean_novel_by_pose_mode'][mode][metric]
                    for m in all_metrics.values()
                ]))
                for metric in ('psnr', 'ssim')
            }
            for mode in ('global_fit', 'anchor')
        }
        suffix = '' if variant is None else f'_{variant}'
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f'nvs_metrics_all_sessions{suffix}.json').write_text(
            json.dumps({'mean_over_sessions': summary, 'sessions': all_metrics}, indent=2) + '\n',
        )
        log_step(
            f'cheese3d_nvs [{variant or "default"}]: {len(all_metrics)}/{len(sessions)} '
            f'sessions done; mean over sessions: {summary}',
            level='info',
        )
        results[variant or 'default'] = all_metrics
    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='model dir with config.yaml')
    parser.add_argument('--input', type=Path, required=True, help='root Cheese3D directory')
    parser.add_argument('--output', type=Path, required=True, help='output directory')
    parser.add_argument(
        '--sessions',
        nargs='+',
        default=None,
        help='session directory names (default: all sessions in the saved training config)',
    )
    parser.add_argument(
        '--frame-idx',
        type=int,
        default=None,
        help='source frame index for every session (default: first frame of each)',
    )
    parser.add_argument('--split', default='val')
    parser.add_argument('--cameras', nargs='+', default=list(_DEFAULT_CAMERAS))
    parser.add_argument(
        '--align-cameras',
        nargs='+',
        default=None,
        help='cameras used to fit the pred-to-GT transform (default: model input cameras)',
    )
    parser.add_argument(
        '--amp',
        choices=('none', 'bf16'),
        default='none',
        help='bf16 runs the forward under autocast like training (default: fp32)',
    )
    parser.add_argument(
        '--context-batch',
        type=int,
        default=1,
        help='records collated per forward pass; the requested frame is sample 0',
    )
    parser.add_argument(
        '--intrinsics',
        choices=('gt', 'pred_scaled'),
        default='gt',
        help='intrinsics for the global_fit renders; metrics for both are always saved',
    )
    parser.add_argument(
        '--pose-mode',
        choices=('anchor', 'global_fit'),
        default='anchor',
        help='how the projected row places cameras; metrics for both are always saved',
    )
    parser.add_argument(
        '--anchor-camera',
        default='TC',
        help='anchor mode places cameras the model lacks relative to this camera',
    )
    parser.add_argument(
        '--anchor-source',
        choices=('auto', 'predicted', 'derived'),
        default='auto',
        help='anchor pose: the predicted one (auto when the model has it) or derived from the '
        'GT rig via the similarity fit',
    )
    parser.add_argument(
        '--held-out-tc',
        choices=('both', 'visible', 'masked'),
        default='both',
        help='for center_camera_holdout models: run with the held-out view visible, with its '
        'strict holdout (tokens zeroed, its Gaussians dropped), or both',
    )
    parser.add_argument(
        '--hide-held-out-from-encoders',
        action='store_true',
        help='masked variant only: also hide the held-out image from the pose encoder and DINO '
        '(for models trained with training.strict_holdout_hide_encoders)',
    )
    parser.add_argument('--no-save-glb', action='store_true', help='skip the .glb scene')
    args = parser.parse_args()
    run_nvs(
        model_dir=args.model,
        dataset_path=args.input,
        output_dir=args.output,
        sessions=args.sessions,
        frame_idx=args.frame_idx,
        split=args.split,
        cameras=tuple(args.cameras),
        align_cameras=args.align_cameras,
        amp=args.amp,
        context_batch=args.context_batch,
        intrinsics=args.intrinsics,
        save_glb=not args.no_save_glb,
        pose_mode=args.pose_mode,
        anchor_camera=args.anchor_camera,
        anchor_source=args.anchor_source,
        held_out_tc=args.held_out_tc,
        hide_held_out_from_encoders=args.hide_held_out_from_encoders,
    )


if __name__ == '__main__':
    main()
