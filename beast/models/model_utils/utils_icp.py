"""ICP and Kabsch alignment utilities for point cloud registration."""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch

try:
    import open3d as o3d
    from open3d.pipelines.registration import RegistrationResult
except ImportError:
    o3d = None
    RegistrationResult = Any


def pixel_xy_to_pointcloud_flat_indices(
    xy: np.ndarray,
    height: int,
    width: int,
    ph: int,
    pw: int,
) -> list[int]:
    """Map OpenCV-style pixel coordinates (x=column, y=row) to flat point cloud indices.

    Uses the same ordering as ``pseudo_pointcloud_normalized`` / einops
    ``(hh ph) (ww pw) -> (hh ww ph pw)``.

    Args:
        xy: array of shape (N, 2) with (x, y) per row.
        height: depth/RGB grid height.
        width: depth/RGB grid width.
        ph: patch height (must divide height).
        pw: patch width (must divide width).

    Returns:
        list of flat indices of length N.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f'Expected xy shape (N, 2), got {xy.shape}')

    hh = height // ph
    ww = width // pw
    if hh * ph != height or ww * pw != width:
        raise ValueError(
            f'height={height}, width={width} must be divisible by ph={ph}, pw={pw}'
        )

    cols = np.rint(xy[:, 0]).astype(np.int64)
    rows = np.rint(xy[:, 1]).astype(np.int64)
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    pr = rows // ph
    pc = cols // pw
    ir = rows % ph
    ic = cols % pw
    flat = pr * (ww * ph * pw) + pc * (ph * pw) + ir * pw + ic
    return [int(i) for i in flat.tolist()]


def pixel_xy_to_pointcloud_flat_indices_torch(
    xy: torch.Tensor,
    height: int,
    width: int,
    ph: int,
    pw: int,
) -> list[int]:
    """Same mapping as pixel_xy_to_pointcloud_flat_indices for torch tensors.

    Args:
        xy: tensor of shape (N, 2) on any device/dtype.
        height: depth/RGB grid height.
        width: depth/RGB grid width.
        ph: patch height (must divide height).
        pw: patch width (must divide width).

    Returns:
        list of flat indices of length N.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f'Expected xy shape (N, 2), got {tuple(xy.shape)}')

    hh = height // ph
    ww = width // pw
    if hh * ph != height or ww * pw != width:
        raise ValueError(
            f'height={height}, width={width} must be divisible by ph={ph}, pw={pw}'
        )

    cols = torch.round(xy[:, 0]).long().clamp(0, width - 1)
    rows = torch.round(xy[:, 1]).long().clamp(0, height - 1)
    pr = rows // ph
    pc = cols // pw
    ir = rows % ph
    ic = cols % pw
    flat = pr * (ww * ph * pw) + pc * (ph * pw) + ir * pw + ic
    return [int(i) for i in flat.detach().cpu().tolist()]


def filter_icp_pairs_by_depth_masks(
    src_idx: list[int],
    tgt_idx: list[int],
    src_foreground_mask_flat: np.ndarray,
    tgt_foreground_mask_flat: np.ndarray,
    max_pairs: int,
) -> tuple[list[int], list[int]]:
    """Keep ICP pairs where both endpoints pass per-view foreground masks.

    Args:
        src_idx: list of source point indices.
        tgt_idx: list of target point indices.
        src_foreground_mask_flat: flat foreground mask for the source view.
        tgt_foreground_mask_flat: flat foreground mask for the target view.
        max_pairs: maximum number of pairs to keep (0 = unlimited).

    Returns:
        tuple of (filtered_src_idx, filtered_tgt_idx).
    """
    s_out: list[int] = []
    t_out: list[int] = []
    for s, t in zip(src_idx, tgt_idx):
        if bool(src_foreground_mask_flat[s]) and bool(tgt_foreground_mask_flat[t]):
            s_out.append(s)
            t_out.append(t)
            if max_pairs > 0 and len(s_out) >= max_pairs:
                break
    return s_out, t_out


def run_icp(
    source_pcd,
    target_pcd,
    src_idx: list[int],
    tgt_idx: list[int],
    max_correspondence_distance: float = 0.05,
) -> tuple[np.ndarray, Any]:
    """Run the full ICP pipeline: Kabsch init → ICP refinement.

    Args:
        source_pcd: Open3D PointCloud to be aligned.
        target_pcd: Open3D PointCloud used as reference.
        src_idx: source landmark indices for Kabsch initialisation.
        tgt_idx: target landmark indices for Kabsch initialisation.
        max_correspondence_distance: ICP correspondence threshold.

    Returns:
        tuple of (T_icp [4, 4] ndarray, RegistrationResult).
    """
    T_init = estimate_initial_transform(source_pcd, target_pcd, src_idx, tgt_idx)

    result, T_icp = refine_with_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)

    return T_icp, result


def kabsch_transform(src_pts: np.ndarray, tgt_pts: np.ndarray) -> np.ndarray:
    """Estimate rigid-body transform from N≥3 point correspondences via Kabsch algorithm.

    Args:
        src_pts: source points of shape (N, 3).
        tgt_pts: corresponding target points of shape (N, 3).

    Returns:
        SE(3) transformation matrix of shape (4, 4) such that tgt ≈ T @ [src | 1]ᵀ.
    """
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)

    src_c = src.mean(axis=0)
    tgt_c = tgt.mean(axis=0)
    src_demean = src - src_c
    tgt_demean = tgt - tgt_c

    H = src_demean.T @ tgt_demean

    U, _, Vt = np.linalg.svd(H)

    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)

    R = Vt.T @ D @ U.T
    t = tgt_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_similarity_transform_to_poses(transform: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Apply a similarity transform to camera-to-world poses, keeping rotation orthonormal.

    The transform's rotation block ``s*R`` is decomposed back into scale and a proper
    rotation so that only camera *centers* are scaled; camera *orientations* remain
    orthonormal (a raw ``transform @ c2w`` would instead scale the orientation basis
    vectors too, which would shear the frustum mesh drawn by ``add_scene_cam``).

    Args:
        transform: 4x4 similarity transform, e.g. from
            ``estimate_camera_similarity_transform``, with rotation block ``s*R``.
        c2w: camera-to-world matrices, shape (..., 4, 4).

    Returns:
        transformed camera-to-world matrices, same shape as ``c2w``.
    """
    sr = transform[:3, :3]
    scale = float(np.linalg.det(sr)) ** (1.0 / 3.0)
    R = sr / scale
    t = transform[:3, 3]

    c2w = np.asarray(c2w, dtype=float)
    out = c2w.copy()
    out[..., :3, :3] = R @ c2w[..., :3, :3]
    out[..., :3, 3] = scale * np.einsum('ij,...j->...i', R, c2w[..., :3, 3]) + t
    return out


def estimate_camera_similarity_transform(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
) -> np.ndarray:
    """Estimate a similarity transform aligning predicted camera poses to GT camera poses.

    Solves rotation and scale/translation in two decoupled steps, rather than fitting
    a single Umeyama-style similarity transform to synthetic orientation-derived
    points: rotation columns are unit vectors, so mixing them with camera centers
    (which carry the real, unknown inter-frame scale) into one fit would require
    offsetting each camera center by a length that is *already* scaled consistently
    between the two frames — which is circular, since that scale is exactly what's
    being solved for. Using a fixed absolute offset length instead
    injects scale-inconsistent correspondences that bias both the recovered
    rotation and scale. This two-step approach is standard for camera-trajectory
    Sim(3) alignment (e.g. TUM/KITTI-style evaluation):

    1. Rotation: orthogonal Procrustes over all stacked rotation-matrix columns
       (unit vectors, so this step carries no scale information and works even
       from a single camera pose).
    2. Scale + translation: 1-D least squares fit of GT camera centers to
       rotated predicted camera centers, with the rotation from step 1 held
       fixed. Requires >= 2 camera poses for scale to be observable (with only 1,
       scale defaults to 1.0, since a lone camera center carries no baseline to
       measure scale against).

    Args:
        pred_c2w: predicted camera-to-world matrices, shape (V, 4, 4), V >= 1.
        gt_c2w: corresponding ground-truth camera-to-world matrices, shape (V, 4, 4).

    Returns:
        4x4 similarity transform ``T`` such that
        ``gt_c2w ≈ apply_similarity_transform_to_poses(T, pred_c2w)``.
    """
    pred_c2w = np.asarray(pred_c2w, dtype=float)
    gt_c2w = np.asarray(gt_c2w, dtype=float)
    num_views = pred_c2w.shape[0]

    # rotation: orthogonal Procrustes over stacked rotation-matrix columns
    pred_axes = pred_c2w[:, :3, :3].transpose(0, 2, 1).reshape(-1, 3)
    gt_axes = gt_c2w[:, :3, :3].transpose(0, 2, 1).reshape(-1, 3)
    H = pred_axes.T @ gt_axes
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T

    # scale + translation: 1-D least squares on camera centers, rotation held fixed
    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    rotated_pred_centers = pred_centers @ R.T
    pred_c = rotated_pred_centers.mean(axis=0)
    gt_c = gt_centers.mean(axis=0)
    pred_demean = rotated_pred_centers - pred_c
    gt_demean = gt_centers - gt_c
    denom = float((pred_demean ** 2).sum())
    scale = float((pred_demean * gt_demean).sum() / denom) if num_views >= 2 and denom > 1e-12 else 1.0
    t = gt_c - scale * pred_c

    transform = np.eye(4)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t
    return transform


def kabsch_rotation_batched(
    cross_cov: torch.Tensor,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Batched Wahba/Kabsch rotation from cross-covariance H = Xc^T Yc of shape [B, 3, 3].

    Runs SVD/det in float32. Returns proper SO(3) R of shape [B, 3, 3] cast to out_dtype.

    Args:
        cross_cov: cross-covariance matrices of shape [B, 3, 3].
        eps: regularisation term added to the diagonal before SVD.
        out_dtype: output dtype.

    Returns:
        rotation matrices of shape [B, 3, 3].
    """
    dev = cross_cov.device
    h32 = cross_cov.float()
    eye = torch.eye(3, device=dev, dtype=torch.float32).unsqueeze(0).expand_as(h32)
    h32 = h32 + eye * float(eps)

    u, _, vh = torch.linalg.svd(h32)
    v = vh.transpose(-2, -1)
    r = torch.bmm(v, u.transpose(-2, -1))

    neg = torch.linalg.det(r) < 0
    if neg.any():
        vf = v.clone()
        vf[neg, :, 2] *= -1.0
        r = torch.bmm(vf, u.transpose(-2, -1))

    return r.to(out_dtype)


def estimate_merge_kabsch_rt_torch(
    src_cloud: torch.Tensor,
    tgt_cloud: torch.Tensor,
    source_indices: list[int],
    target_indices: list[int],
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kabsch SE(3) from indexed correspondences on full (N, 3) point clouds.

    Row-vector convention: x' = x @ R.T + t maps source toward target.
    Differentiable w.r.t. src_cloud and tgt_cloud; indices are discrete.

    Args:
        src_cloud: source point cloud of shape (N, 3).
        tgt_cloud: target point cloud of shape (N, 3).
        source_indices: correspondence indices into src_cloud.
        target_indices: correspondence indices into tgt_cloud.
        eps: regularisation for SVD.

    Returns:
        tuple of (R [3, 3], t [3]) tensors.
    """
    if src_cloud.dim() != 2 or src_cloud.shape[-1] != 3:
        raise ValueError(f'Expected src_cloud (N, 3), got {src_cloud.shape}')
    if tgt_cloud.shape != src_cloud.shape:
        raise ValueError(f'tgt_cloud shape {tgt_cloud.shape} != src {src_cloud.shape}')

    ii = torch.tensor(source_indices, device=src_cloud.device, dtype=torch.long)
    jj = torch.tensor(target_indices, device=src_cloud.device, dtype=torch.long)
    src_corr = src_cloud.index_select(0, ii)
    tgt_corr = tgt_cloud.index_select(0, jj)
    out_dtype = src_cloud.dtype

    mu_s = src_corr.mean(dim=0)
    mu_t = tgt_corr.mean(dim=0)
    xc = src_corr - mu_s
    yc = tgt_corr - mu_t
    h = (xc.T @ yc).unsqueeze(0)

    amp_off = (
        torch.amp.autocast(device_type='cuda', enabled=False)
        if src_cloud.is_cuda
        else contextlib.nullcontext()
    )
    with amp_off:
        r = kabsch_rotation_batched(h, eps=eps, out_dtype=out_dtype).squeeze(0)

    t = mu_t - mu_s @ r.T
    return r, t


def run_point_to_point_icp(
    source,
    target,
    trans_init: np.ndarray,
    threshold: float = 0.02,
):
    """Run point-to-point ICP registration.

    Args:
        source: Open3D source PointCloud.
        target: Open3D target PointCloud.
        trans_init: initial transformation of shape (4, 4).
        threshold: maximum correspondence distance.

    Returns:
        Open3D RegistrationResult.
    """
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )


def estimate_initial_transform(
    source_pcd,
    target_pcd,
    source_indices: list[int],
    target_indices: list[int],
) -> np.ndarray:
    """Compute initial SE(3) transform from manual correspondences via Kabsch algorithm.

    Args:
        source_pcd: Open3D source PointCloud.
        target_pcd: Open3D target PointCloud.
        source_indices: indices of landmark points in source_pcd.
        target_indices: indices of corresponding landmark points in target_pcd.

    Returns:
        T_init: (4, 4) ndarray initial transformation.
    """
    src_pts = np.asarray(source_pcd.points)[source_indices]
    tgt_pts = np.asarray(target_pcd.points)[target_indices]

    T_init = kabsch_transform(src_pts, tgt_pts)

    src_transformed = (T_init[:3, :3] @ src_pts.T).T + T_init[:3, 3]
    residuals = np.linalg.norm(src_transformed - tgt_pts, axis=1)  # noqa: F841
    return T_init


def refine_with_icp(
    source_pcd,
    target_pcd,
    T_init: np.ndarray,
    max_correspondence_distance: float = 0.05,
) -> tuple[Any, np.ndarray]:
    """Refine the Kabsch initial transform with point-to-point ICP.

    Args:
        source_pcd: Open3D source PointCloud.
        target_pcd: Open3D target PointCloud.
        T_init: initial transformation of shape (4, 4).
        max_correspondence_distance: ICP correspondence threshold (same units as cloud).

    Returns:
        tuple of (RegistrationResult, T_icp [4, 4] ndarray).
    """
    result = run_point_to_point_icp(source_pcd, target_pcd, T_init, max_correspondence_distance)

    T_icp = np.asarray(result.transformation)
    return result, T_icp
