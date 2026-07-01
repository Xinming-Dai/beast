"""Debug utilities for visualizing merged point clouds during Sable training."""

from pathlib import Path

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
    """Export pre-alignment PLYs (always) + NPZ (when correspondences exist).

    Expects ``source_indices`` / ``target_indices`` as flat point indices into
    ``src_xyz`` / ``tgt_xyz`` (same convention as Sable merge_pcd Kabsch).

    twoview3d loads: np.load(npz)['source_indices'], ['target_indices'].
    """
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

    # the npz is only meaningful when there are enough matched correspondences for Kabsch
    if len(source_indices) < 3 or len(source_indices) != len(target_indices):
        return

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
    masked_rgb_bgr_views: list[np.ndarray] | None,
    corr_left_xy: np.ndarray | None,
    corr_right_xy: np.ndarray | None,
) -> None:
    """Save side-by-side RGB (and optional masked) images plus correspondence overlay."""
    import cv2

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    n = len(rgb_bgr_views)

    if n >= 2:
        sep = np.full((rgb_bgr_views[0].shape[0], 8, 3), 255, dtype=np.uint8)
        cv2.imwrite(
            str(save_path / 'pair_rgb_sidebyside.png'),
            np.hstack([rgb_bgr_views[0], sep, rgb_bgr_views[1]]),
        )

    if masked_rgb_bgr_views is not None and len(masked_rgb_bgr_views) >= 2:
        sep = np.full((masked_rgb_bgr_views[0].shape[0], 8, 3), 255, dtype=np.uint8)
        cv2.imwrite(
            str(save_path / 'pair_rgb_masked_sidebyside.png'),
            np.hstack([masked_rgb_bgr_views[0], sep, masked_rgb_bgr_views[1]]),
        )

    if (
        corr_left_xy is not None
        and corr_right_xy is not None
        and len(corr_left_xy) > 0
        and len(corr_left_xy) == len(corr_right_xy)
        and n >= 2
    ):
        green = (0, 255, 0)
        o0 = _draw_xy_points_bgr(rgb_bgr_views[0], corr_left_xy, color_bgr=green, with_index=True)
        o1 = _draw_xy_points_bgr(rgb_bgr_views[1], corr_right_xy, color_bgr=green, with_index=True)
        sep = np.full((o0.shape[0], 8, 3), 255, dtype=np.uint8)
        cv2.imwrite(
            str(save_path / 'pair_rgb_sidebyside_corr_overlay.png'),
            np.hstack([o0, sep, o1]),
        )


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
    masked_views: list[np.ndarray] = []
    depth_gray_list: list[np.ndarray] = []
    depth_bgr_list: list[np.ndarray | None] = []
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
        if 'mask' in data:
            mask = data['mask'][batch_idx, view_global].float()
            mask_resized = torch.nn.functional.interpolate(
                mask.unsqueeze(0),
                size=(pcd_h, pcd_w),
                mode='nearest',
            )[0]
            masked_views.append(_chw01_to_bgr_u8(img_resized * mask_resized + (1 - mask_resized)))
        img_flat = rearrange(
            img_resized,
            'c (hh ph) (ww pw) -> (hh ww ph pw) c',
            hh=pcd_h // ph, ww=pcd_w // pw,
            ph=ph, pw=pw,
        )
        color_list.append(img_flat)

        if depth_maps is not None:
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
            d_bgr = depth_bgr_list[vi]
            if d_bgr is not None:
                Image.fromarray(d_bgr[..., ::-1]).save(
                    save_path / f'depth_view{vi}_turbo.png'
                )

    _save_pair_images_and_corr_overlays(
        save_dir=batch_save_dir,
        rgb_bgr_views=rgb_bgr_views,
        masked_rgb_bgr_views=masked_views if masked_views else None,
        corr_left_xy=corr_left_xy,
        corr_right_xy=corr_right_xy,
    )

    if icp_pre_kabsch_src_xyz is not None and icp_pre_kabsch_tgt_xyz is not None:
        n_per_view = pts_np.shape[0] // v_target
        src_c = colors_np[:n_per_view]
        tgt_c = colors_np[n_per_view : 2 * n_per_view]
        _save_icp_twoview3d_bundle(
            batch_save_dir,
            src_xyz=icp_pre_kabsch_src_xyz,
            tgt_xyz=icp_pre_kabsch_tgt_xyz,
            src_colors=src_c,
            tgt_colors=tgt_c,
            source_indices=src_corr_idx or [],
            target_indices=tgt_corr_idx or [],
        )


def save_debug_pcd(
    xyz_init: torch.Tensor,
    xyz_init_for_merge: np.ndarray,
    data: dict,
    target_idx: torch.Tensor,
    v_target: int,
    depth_output: torch.Tensor | None,
    debug_corrs: list[tuple[list[int], list[int], tuple[np.ndarray, np.ndarray] | None]],
    pcd_h: int,
    pcd_w: int,
    num_batches: int,
    ph: int,
    pw: int,
    save_dir: str = 'debug_ckpt/debug_pcd',
    filename: str = 'xyz_init.ply',
) -> None:
    """Save debug point clouds and overlays for each batch in debug_corrs.

    Args:
        xyz_init: [b, v_target*N, 3] world-space points after Kabsch alignment.
        xyz_init_for_merge: numpy copy of xyz_init before the loop, shape [b, v, N, 3].
        data: forward() data dict containing "image" [b, v_all, 3, H, W] in [0, 1].
        target_idx: [b, v_target] global view indices into data["image"].
        v_target: number of target views.
        depth_output: [b, v_target, H, W] VDA depth maps, or None.
        debug_corrs: one entry per saved batch — (src_idx, tgt_idx, corr_xy_from_pixels).
        pcd_h: depth-map height.
        pcd_w: depth-map width.
        num_batches: number of batches saved; controls whether subdirs are used.
        ph: patch height.
        pw: patch width.
        save_dir: root directory to write outputs.
        filename: output PLY filename.
    """
    use_subdirs = num_batches > 1
    for bi, (src_idx, tgt_idx, corr_xy) in enumerate(debug_corrs):
        batch_dir = str(Path(save_dir) / f'batch_{bi:03d}') if use_subdirs else save_dir
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
            src_corr_idx=src_idx,
            tgt_corr_idx=tgt_idx,
            depth_maps=depth_output,
            icp_pre_kabsch_src_xyz=np.ascontiguousarray(xyz_init_for_merge[bi, 0]),
            icp_pre_kabsch_tgt_xyz=np.ascontiguousarray(xyz_init_for_merge[bi, 1]),
            corr_left_xy=corr_xy[0] if corr_xy is not None else None,
            corr_right_xy=corr_xy[1] if corr_xy is not None else None,
        )
