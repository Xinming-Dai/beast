"""Video Depth Anything: sys.path/checkpoint helpers, frozen model load, depth forward, debug dumps.

Provides depth map normalization and pseudo pointcloud normalization utilities.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from einops import rearrange

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VDA_INPUT_SIZE = 518


def ensure_vda_on_path() -> None:
    """Ensure a local Video-Depth-Anything repo is on sys.path."""
    candidates = [
        os.environ.get('VDA_REPO_ROOT'),
        _REPO_ROOT / 'third_party' / 'VDA',
    ]
    for candidate in candidates:
        if not candidate:
            continue
        vda_root = Path(candidate).expanduser().resolve()
        if not vda_root.exists():
            continue
        if not (vda_root / 'video_depth_anything' / 'video_depth.py').exists():
            continue
        vda_root_str = str(vda_root)
        if vda_root_str not in sys.path:
            sys.path.insert(0, vda_root_str)
        vda_utils_root = str((vda_root / 'utils').resolve())
        existing_utils = sys.modules.get('utils')
        if existing_utils is not None and hasattr(existing_utils, '__path__'):
            if vda_utils_root not in existing_utils.__path__:
                existing_utils.__path__.append(vda_utils_root)
        return


def resolve_vda_checkpoint_path(vda_cfg: Mapping[str, Any], *, encoder: str) -> Path | None:
    """Resolve the VDA checkpoint path from config or default locations.

    Args:
        vda_cfg: VDA config mapping.
        encoder: encoder name (e.g. 'vitb').

    Returns:
        resolved Path if found, else None.
    """
    ckpt_path = vda_cfg.get('checkpoint_path')
    if ckpt_path:
        resolved = Path(ckpt_path).expanduser().resolve()
        if resolved.exists():
            return resolved

    candidates = [
        _REPO_ROOT / 'third_party' / 'VDA' / 'checkpoints' / f'video_depth_anything_{encoder}.pth',
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def _disable_vda_xformers_flags() -> None:
    for module_name in (
        'video_depth_anything.dinov2_layers.attention',
        'video_depth_anything.dinov2_layers.block',
        'video_depth_anything.dinov2_layers.swiglu_ffn',
        'video_depth_anything.motion_module.attention',
        'video_depth_anything.motion_module.motion_module',
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, 'XFORMERS_AVAILABLE'):
            module.XFORMERS_AVAILABLE = False


def load_frozen_video_depth_anything(vda_cfg: Mapping[str, Any]) -> nn.Module:
    """Build VideoDepthAnything, load checkpoint if present, freeze and set eval.

    Args:
        vda_cfg: VDA config mapping with optional keys 'encoder', 'metric', 'checkpoint_path'.

    Returns:
        frozen nn.Module in eval mode.
    """
    ensure_vda_on_path()
    _disable_vda_xformers_flags()
    from video_depth_anything.video_depth import VideoDepthAnything

    vda_encoder = vda_cfg.get('encoder', 'vitb')
    vda_metric = vda_cfg.get('metric', False)
    vda_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    vda_kwargs = {**vda_configs[vda_encoder], 'metric': vda_metric}
    model = VideoDepthAnything(**vda_kwargs)
    for module in model.modules():
        if hasattr(module, '_use_memory_efficient_attention_xformers'):
            module._use_memory_efficient_attention_xformers = False
    ckpt_path = resolve_vda_checkpoint_path(vda_cfg, encoder=vda_encoder)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state, strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def preprocess_vda_input_images(
    images: torch.Tensor,
    *,
    input_size: int = VDA_INPUT_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize and normalise images for VDA input.

    Args:
        images: float tensor of shape [b, num_views, 3, H, W] in [0, 1].
        input_size: target spatial resolution (default 518).

    Returns:
        tuple of (x_vda normalised [b, V, 3, S, S], resized_rgb unnormalised [b, V, 3, S, S]).
    """
    if images.ndim != 5:
        raise ValueError(
            f'Expected images to have shape [B, T, 3, H, W], got {tuple(images.shape)}'
        )

    b, num_views, c, h, w = images.shape
    resized_rgb = torch.nn.functional.interpolate(
        images.view(b * num_views, c, h, w),
        size=(int(input_size), int(input_size)),
        mode='bilinear',
        align_corners=True,
    ).view(b, num_views, c, int(input_size), int(input_size))

    mean = resized_rgb.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = resized_rgb.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    x_vda = (resized_rgb - mean) / std
    return x_vda, resized_rgb


def infer_vda_depth_maps(
    vda: nn.Module,
    x_vda: torch.Tensor,
    *,
    num_views: int,
) -> torch.Tensor:
    """Run VDA model inference and return depth maps.

    Args:
        vda: frozen VideoDepthAnything model.
        x_vda: normalised input of shape [B, T, 3, H, W].
        num_views: number of views to keep from the output.

    Returns:
        depth maps of shape [B, num_views, H, W].
    """
    if x_vda.ndim != 5:
        raise ValueError(
            f'Expected x_vda to have shape [B, T, 3, H, W], got {tuple(x_vda.shape)}'
        )
    if x_vda.shape[1] != int(num_views):
        raise ValueError(
            f'num_views={num_views} does not match x_vda.shape[1]={x_vda.shape[1]}'
        )

    with torch.no_grad():
        return vda(x_vda)[:, :num_views]


def extract_vda_depth(
    vda: nn.Module,
    images: torch.Tensor,
    num_views: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract VDA depth maps for a batch of images.

    Args:
        vda: frozen VideoDepthAnything model.
        images: float tensor of shape [b, num_views, 3, H, W] in [0, 1].
        num_views: number of input views.

    Returns:
        tuple of (depth_maps [b, num_views, H_vda, W_vda], resized_rgb [b, num_views, 3, S, S]).
    """
    _, v_input, _, _, _ = images.shape
    assert v_input == num_views

    x_vda, vda_in = preprocess_vda_input_images(images, input_size=VDA_INPUT_SIZE)
    depth_maps = infer_vda_depth_maps(vda, x_vda, num_views=v_input)
    return depth_maps, vda_in


def save_vda_debug_images(
    vda_in_orig: torch.Tensor,
    vda_in: torch.Tensor,
    depth_maps: torch.Tensor,
    debug_dir: Path | str,
) -> None:
    """Write one RGB/depth example under debug_dir.

    Args:
        vda_in_orig: original RGB tensor [b, V, 3, H, W].
        vda_in: resized RGB tensor [b, V, 3, S, S].
        depth_maps: depth map tensor [b, V, S, S].
        debug_dir: directory to write debug images into.
    """
    from PIL import Image

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    rgb_orig = vda_in_orig[0, 0].detach().float().cpu().clamp(0, 1)
    rgb_stretch = vda_in[0, 0].detach().float().cpu().clamp(0, 1)
    depth0 = depth_maps[0, 0].detach().float().cpu()

    def _to_uint8_rgb(x_chw: torch.Tensor):
        x_hwc = x_chw.permute(1, 2, 0).numpy()
        return (x_hwc * 255.0).round().clip(0, 255).astype('uint8')

    Image.fromarray(_to_uint8_rgb(rgb_orig)).save(debug_dir / 'vda_rgb_orig.png')
    Image.fromarray(_to_uint8_rgb(rgb_stretch)).save(debug_dir / 'vda_rgb_stretched_518.png')

    d = depth0.numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d_u8 = (d * 255.0).round().clip(0, 255).astype('uint8')
    Image.fromarray(d_u8).save(debug_dir / 'vda_depth_gray_518.png')
    try:
        import cv2

        d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
        Image.fromarray(d_color[..., ::-1]).save(debug_dir / 'vda_depth_turbo_518.png')
    except Exception:
        pass


def depth_map_normalized(depth: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise a depth map to [-0.5, 0.5], ignoring zero-valued pixels.

    Args:
        depth: depth tensor of any shape.
        eps: minimum range to avoid division by zero.

    Returns:
        normalised tensor with the same shape as depth.
    """
    z_flat = depth.reshape(-1)
    valid = z_flat > 0
    out = torch.full_like(z_flat, -0.5)
    if valid.any():
        zv = z_flat[valid]
        z_min = zv.min()
        z_max = zv.max()
        rng = (z_max - z_min).clamp_min(eps)
        out[valid] = (z_flat[valid] - z_min) / rng - 0.5
    return out.view_as(depth)


def pseudo_pointcloud_normalized(depth: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """Build a normalised pseudo point cloud from a single-view depth map.

    Points are arranged in patch-major order matching einops pattern
    ``(hh ph) (ww pw) -> (hh ww ph pw)``.

    Args:
        depth: depth tensor of shape (H, W).
        ph: patch height (must divide H).
        pw: patch width (must divide W).

    Returns:
        point cloud tensor of shape (H*W, 3).
    """
    if depth.ndim != 2:
        raise ValueError(f'Expected depth to have shape (H, W), got {tuple(depth.shape)}')

    H, W = depth.shape

    if H % ph != 0 or W % pw != 0:
        raise ValueError(
            f'depth (H,W)=({H},{W}) must be divisible by patch (ph,pw)=({ph},{pw})'
        )

    hh = H // ph
    ww = W // pw
    device = depth.device
    dtype = depth.dtype

    u = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)

    X = (u - W / 2) / W
    Y = (v - H / 2) / H
    Z = depth_map_normalized(depth)

    points_hw = torch.stack([-X, Y, Z], dim=-1)
    points = rearrange(
        points_hw,
        '(hh ph) (ww pw) c -> (hh ww ph pw) c',
        hh=hh,
        ww=ww,
        ph=ph,
        pw=pw,
    )
    return points
