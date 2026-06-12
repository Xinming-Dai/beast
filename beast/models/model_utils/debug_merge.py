"""Debug utilities for visualizing merged point clouds during Sable training."""

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import torch
from einops import rearrange
from PIL import Image


def _save_icp_twoview3d_bundle(
    save_dir: str,
    *,
    src_xyz: np.ndarray,
    tgt_xyz: np.ndarray,
    src_colors: np.ndarray,
    tgt_colors: np.ndarray,
    source_indices: list[int],
    target_indices: list[int],
    src_ply_name: str = 'icp_source_view0.ply',
    tgt_ply_name: str = 'icp_target_view1.ply',
    corr_npz_name: str = 'icp_correspondences.npz',
) -> None:
    """Export pre-alignment PLYs + NPZ for twoview3d icp_with_keypoints.py.

    Expects ``source_indices`` / ``target_indices`` as flat point indices into
    ``src_xyz`` / ``tgt_xyz`` (same convention as Sable merge_pcd Kabsch).

    twoview3d loads: np.load(npz)['source_indices'], ['target_indices'].
    """
    if len(source_indices) < 3 or len(target_indices) < 3:
        return
    if len(source_indices) != len(target_indices):
        return

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(np.asarray(src_xyz, dtype=np.float64))
    src_pcd.colors = o3d.utility.Vector3dVector(np.clip(src_colors.astype(np.float64), 0.0, 1.0))

    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(np.asarray(tgt_xyz, dtype=np.float64))
    tgt_pcd.colors = o3d.utility.Vector3dVector(np.clip(tgt_colors.astype(np.float64), 0.0, 1.0))

    o3d.io.write_point_cloud(str(save_path / src_ply_name), src_pcd)
    o3d.io.write_point_cloud(str(save_path / tgt_ply_name), tgt_pcd)

    np.savez(
        str(save_path / corr_npz_name),
        source_indices=np.asarray(source_indices, dtype=np.int64),
        target_indices=np.asarray(target_indices, dtype=np.int64),
    )


def _chw01_to_bgr_u8(chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] float [0,1] -> uint8 HxWx3 BGR for OpenCV."""
    x = chw.detach().float().cpu().clamp(0, 1)
    hwc = (x * 255.0).permute(1, 2, 0).numpy().round().clip(0, 255).astype(np.uint8)
    return hwc[..., ::-1].copy()


def _draw_xy_points_bgr(
    bgr: np.ndarray,
    xy: np.ndarray,
    *,
    color_bgr: tuple,
    radius: int = 4,
    with_index: bool = False,
) -> np.ndarray:
    """Draw points onto a BGR image, optionally labelled with their index."""
    import cv2

    out = bgr.copy()
    xy = np.asarray(xy, dtype=np.float64)
    for i in range(xy.shape[0]):
        x = int(np.clip(round(float(xy[i, 0])), 0, out.shape[1] - 1))
        y = int(np.clip(round(float(xy[i, 1])), 0, out.shape[0] - 1))
        cv2.circle(out, (x, y), radius, color_bgr, -1, lineType=cv2.LINE_AA)
        if with_index:
            cv2.putText(
                out,
                str(i),
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color_bgr,
                1,
                cv2.LINE_AA,
            )
    return out


def _save_pair_images_and_corr_overlays(
    *,
    save_dir: str,
    rgb_bgr_views: list[np.ndarray],
    depth_gray_u8_views: list[np.ndarray] | None,
    depth_bgr_views: list[np.ndarray | None] | None,
    corr_left_xy: np.ndarray | None,
    corr_right_xy: np.ndarray | None,
) -> None:
    """Save per-view RGB (and optional depth) plus correspondence overlays."""
    import cv2

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    n = len(rgb_bgr_views)
    has_depth = (
        depth_gray_u8_views is not None
        and len(depth_gray_u8_views) == n
        and depth_bgr_views is not None
        and len(depth_bgr_views) == n
    )
    for vi in range(n):
        cv2.imwrite(str(save_path / f'pair_view{vi}_rgb.png'), rgb_bgr_views[vi])

    if (
        corr_left_xy is not None
        and corr_right_xy is not None
        and len(corr_left_xy) > 0
        and len(corr_left_xy) == len(corr_right_xy)
        and n >= 2
    ):
        green = (0, 255, 0)
        magenta = (255, 0, 255)
        o0 = _draw_xy_points_bgr(rgb_bgr_views[0], corr_left_xy, color_bgr=green, with_index=True)
        o1 = _draw_xy_points_bgr(rgb_bgr_views[1], corr_right_xy, color_bgr=green, with_index=True)
        cv2.imwrite(str(save_path / 'pair_view0_rgb_corr_overlay.png'), o0)
        cv2.imwrite(str(save_path / 'pair_view1_rgb_corr_overlay.png'), o1)

        if has_depth:
            dg0 = cv2.cvtColor(depth_gray_u8_views[0], cv2.COLOR_GRAY2BGR)
            dg1 = cv2.cvtColor(depth_gray_u8_views[1], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(
                str(save_path / 'pair_view0_depth_gray_corr_overlay.png'),
                _draw_xy_points_bgr(dg0, corr_left_xy, color_bgr=magenta),
            )
            cv2.imwrite(
                str(save_path / 'pair_view1_depth_gray_corr_overlay.png'),
                _draw_xy_points_bgr(dg1, corr_right_xy, color_bgr=magenta),
            )

            if depth_bgr_views[0] is not None and depth_bgr_views[1] is not None:
                cv2.imwrite(
                    str(save_path / 'pair_view0_depth_turbo_corr_overlay.png'),
                    _draw_xy_points_bgr(depth_bgr_views[0], corr_left_xy, color_bgr=green),
                )
                cv2.imwrite(
                    str(save_path / 'pair_view1_depth_turbo_corr_overlay.png'),
                    _draw_xy_points_bgr(depth_bgr_views[1], corr_right_xy, color_bgr=green),
                )

        sep = np.full((o0.shape[0], 8, 3), 255, dtype=np.uint8)
        grid = np.hstack([o0, sep, o1])
        cv2.imwrite(str(save_path / 'pair_rgb_sidebyside_corr_overlay.png'), grid)


def _per_batch_corr_indices(
    corr: Any | None, batch_idx: int
) -> list[int] | None:
    """Legacy ``List[int]`` applies to batch 0 only; else index a per-batch list."""
    if corr is None:
        return None
    if not isinstance(corr, list) or len(corr) == 0:
        return None
    if isinstance(corr[0], int):
        return list(corr) if batch_idx == 0 else None
    if batch_idx < len(corr):
        item = corr[batch_idx]
        return list(item) if item is not None else None
    return None


def _per_batch_ndarray_or_list(
    arr: Any | None, batch_idx: int
) -> np.ndarray | None:
    """Legacy ``(N,3)`` ndarray applies to batch 0 only; else index a per-batch list."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr if batch_idx == 0 else None
    if isinstance(arr, list) and batch_idx < len(arr):
        a = arr[batch_idx]
        return np.asarray(a) if a is not None else None
    return None


def _per_batch_corr_xy(
    xy: Any | None, batch_idx: int
) -> np.ndarray | None:
    """Legacy ``(N,2)`` ndarray for batch 0; else per-batch list of arrays."""
    if xy is None:
        return None
    if isinstance(xy, np.ndarray):
        return xy if batch_idx == 0 else None
    if isinstance(xy, list) and batch_idx < len(xy):
        a = xy[batch_idx]
        return np.asarray(a) if a is not None else None
    return None


def _save_merged_pcd_single(
    *,
    batch_idx: int,
    batch_save_dir: str,
    xyz_init: torch.Tensor,
    data: dict,
    target_idx: torch.Tensor,
    v_target: int,
    pcd_h: int,
    pcd_w: int,
    ph: int,
    pw: int,
    filename: str,
    src_corr_idx: list[int] | None,
    tgt_corr_idx: list[int] | None,
    depth_maps: torch.Tensor | None,
    icp_pre_kabsch_src_xyz: np.ndarray | None,
    icp_pre_kabsch_tgt_xyz: np.ndarray | None,
    corr_left_xy: np.ndarray | None,
    corr_right_xy: np.ndarray | None,
) -> None:
    """Save debug outputs for a single batch item."""
    color_list = []
    rgb_bgr_views: list[np.ndarray] = []
    depth_gray_list: list[np.ndarray] | None = [] if depth_maps is not None else None
    depth_bgr_list: list[np.ndarray | None] | None = [] if depth_maps is not None else None
    for vi in range(v_target):
        view_global = target_idx[batch_idx, vi].item()
        img = data['image'][batch_idx, view_global].float()
        img_resized = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(pcd_h, pcd_w),
            mode='bilinear',
            align_corners=True,
        )[0]
        rgb_bgr_views.append(_chw01_to_bgr_u8(img_resized))
        img_flat = rearrange(
            img_resized,
            'c (hh ph) (ww pw) -> (hh ww ph pw) c',
            hh=pcd_h // ph, ww=pcd_w // pw,
            ph=ph, pw=pw,
        )
        color_list.append(img_flat)

        if depth_gray_list is not None and depth_bgr_list is not None:
            if vi < depth_maps.shape[1]:
                d = depth_maps[batch_idx, vi].detach().float().cpu().numpy()
                d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
                d_u8 = (d_norm * 255.0).round().clip(0, 255).astype(np.uint8)
                depth_gray_list.append(d_u8)
                d_bgr = None
                try:
                    import cv2
                    d_bgr = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
                except Exception:
                    pass
                depth_bgr_list.append(d_bgr)
            else:
                depth_gray_list.append(
                    np.zeros((pcd_h, pcd_w), dtype=np.uint8)
                )
                depth_bgr_list.append(None)

    colors_np = torch.cat(color_list, dim=0).detach().cpu().clamp(0, 1).numpy()
    pts_np = xyz_init[batch_idx].detach().cpu().float().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    save_path = Path(batch_save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(save_path / filename), pcd)

    if depth_maps is not None:
        for vi in range(min(v_target, depth_maps.shape[1])):
            d_u8 = depth_gray_list[vi]
            Image.fromarray(d_u8).save(save_path / f'depth_view{vi}_gray.png')
            if depth_bgr_list[vi] is not None:
                Image.fromarray(depth_bgr_list[vi][..., ::-1]).save(
                    save_path / f'depth_view{vi}_turbo.png'
                )

    _save_pair_images_and_corr_overlays(
        save_dir=batch_save_dir,
        rgb_bgr_views=rgb_bgr_views,
        depth_gray_u8_views=depth_gray_list,
        depth_bgr_views=depth_bgr_list,
        corr_left_xy=corr_left_xy,
        corr_right_xy=corr_right_xy,
    )

    if (
        icp_pre_kabsch_src_xyz is not None
        and icp_pre_kabsch_tgt_xyz is not None
        and src_corr_idx is not None
        and tgt_corr_idx is not None
        and len(src_corr_idx) >= 3
        and len(src_corr_idx) == len(tgt_corr_idx)
    ):
        n_per_view = pts_np.shape[0] // v_target
        src_c = colors_np[:n_per_view]
        tgt_c = colors_np[n_per_view : 2 * n_per_view]
        _save_icp_twoview3d_bundle(
            batch_save_dir,
            src_xyz=icp_pre_kabsch_src_xyz,
            tgt_xyz=icp_pre_kabsch_tgt_xyz,
            src_colors=src_c,
            tgt_colors=tgt_c,
            source_indices=src_corr_idx,
            target_indices=tgt_corr_idx,
        )


def save_merged_pcd(
    xyz_init: torch.Tensor,
    data: dict,
    target_idx: torch.Tensor,
    v_target: int,
    pcd_h: int,
    pcd_w: int,
    ph: int,
    pw: int,
    save_dir: str = 'debug_ckpt/debug_pcd',
    filename: str = 'xyz_init.ply',
    num_batches: int = 1,
    src_corr_idx: list[int] | list[list[int] | None] | None = None,
    tgt_corr_idx: list[int] | list[list[int] | None] | None = None,
    depth_maps: torch.Tensor | None = None,
    pre_kabsch_src_xyz: np.ndarray | list[np.ndarray] | None = None,
    pre_kabsch_tgt_xyz: np.ndarray | list[np.ndarray] | None = None,
    corr_left_xy: np.ndarray | list[np.ndarray | None] | None = None,
    corr_right_xy: np.ndarray | list[np.ndarray | None] | None = None,
) -> None:
    """Save the merged point cloud colored by its corresponding target-view images.

    When correspondence indices are provided, an additional PLY is saved with
    source correspondence points colored red and target correspondence points
    colored blue, connected by green lines.

    Args:
        xyz_init: [b, v_target*N, 3] world-space points.
        data: forward() data dict containing "image" [b, v_all, 3, H, W] in [0, 1].
        target_idx: [b, v_target] global view indices into data["image"].
        v_target: number of target views.
        pcd_h: depth-map height used when building xyz_init.
        pcd_w: depth-map width used when building xyz_init.
        ph: patch height (self.ph from the model).
        pw: patch width (self.pw from the model).
        save_dir: root directory to write outputs. Defaults to 'debug_ckpt/debug_pcd'.
        filename: output PLY filename (same name in each batch subfolder when applicable).
        num_batches: save batch indices ``0 .. min(num_batches, B) - 1``. If
            ``num_batches > 1``, writes under ``save_dir/batch_000``, etc.;
            if ``1``, writes directly in ``save_dir``.
        src_corr_idx: flat indices into view-0 points, or per-batch list of same.
        tgt_corr_idx: flat indices into view-1 points.
        depth_maps: [b, v_target, H_vda, W_vda] VDA depth maps.
        pre_kabsch_src_xyz: pre-Kabsch (N,3) for one batch, or list per batch index.
            With valid correspondence indices, writes ``icp_source_view0.ply``,
            ``icp_target_view1.ply``, and ``icp_correspondences.npz``.
        pre_kabsch_tgt_xyz: pre-Kabsch (N,3) for one batch, or list per batch index.
        corr_left_xy: (N,2) pixel coords on pcd_h×pcd_w for one batch, or per-batch list.
        corr_right_xy: (N,2) pixel coords on pcd_h×pcd_w for one batch, or per-batch list.
    """
    b = xyz_init.shape[0]
    n_req = max(1, int(num_batches))
    use_subdirs = n_req > 1
    n_save = min(n_req, b)

    for bi in range(n_save):
        batch_dir = (
            str(Path(save_dir) / f'batch_{bi:03d}') if use_subdirs else save_dir
        )
        _save_merged_pcd_single(
            batch_idx=bi,
            batch_save_dir=batch_dir,
            xyz_init=xyz_init,
            data=data,
            target_idx=target_idx,
            v_target=v_target,
            pcd_h=pcd_h,
            pcd_w=pcd_w,
            ph=ph,
            pw=pw,
            filename=filename,
            src_corr_idx=_per_batch_corr_indices(src_corr_idx, bi),
            tgt_corr_idx=_per_batch_corr_indices(tgt_corr_idx, bi),
            depth_maps=depth_maps,
            icp_pre_kabsch_src_xyz=_per_batch_ndarray_or_list(pre_kabsch_src_xyz, bi),
            icp_pre_kabsch_tgt_xyz=_per_batch_ndarray_or_list(pre_kabsch_tgt_xyz, bi),
            corr_left_xy=_per_batch_corr_xy(corr_left_xy, bi),
            corr_right_xy=_per_batch_corr_xy(corr_right_xy, bi),
        )
