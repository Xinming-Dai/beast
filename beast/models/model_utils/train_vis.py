"""Utilities for saving training visualizations (render vs. target grids)."""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_training_visuals(
    output_dir: Path | str,
    result: SimpleNamespace,
    batch: dict,
    step: int,
    max_samples: int = 1,
    max_views: int = 2,
) -> list[Path]:
    """Save a side-by-side render/target PNG for each sample in the batch.

    Args:
        output_dir: directory to write PNG files into (created if missing).
        result: model forward output with ``render`` and ``target_image`` attributes.
        batch: dataloader batch dict; ``scene_name`` key used for filenames.
        step: training step, used in filenames.
        max_samples: how many batch samples to save.
        max_views: how many target views to include per sample.

    Returns:
        list of saved file paths.
    """
    renders = getattr(result, 'render', None)
    targets = getattr(result, 'target_image', None)
    if renders is None or targets is None:
        return []

    depth_target = getattr(result, 'depth_output', None)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_names = batch.get('scene_name', [])
    if isinstance(scene_names, str):
        scene_names = [scene_names]

    sample_count = min(int(max_samples), int(renders.shape[0]), int(targets.shape[0]))
    saved_paths = []
    for sample_idx in range(sample_count):
        scene_name = (
            scene_names[sample_idx]
            if sample_idx < len(scene_names)
            else f'sample_{sample_idx:02d}'
        )
        depth_target_views = depth_target[sample_idx] if depth_target is not None else None
        image = build_render_target_visual(
            renders[sample_idx],
            targets[sample_idx],
            max_views=max_views,
            depth_target_views=depth_target_views,
        )
        filename = f'step_{step:06d}_{_sanitize_filename(scene_name)}_sample{sample_idx:02d}.png'
        path = output_dir / filename
        image.save(path)
        saved_paths.append(path)
    return saved_paths


def build_render_target_visual(
    render_views: torch.Tensor,
    target_views: torch.Tensor,
    max_views: int = 2,
    pad: int = 8,
    label_height: int = 18,
    depth_target_views: torch.Tensor | None = None,
) -> Image.Image:
    """Build a grid image with rendered and target views side by side.

    Args:
        render_views: ``[V, 3, H, W]`` float tensor in ``[0, 1]``.
        target_views: ``[V, 3, H, W]`` float tensor in ``[0, 1]``.
        max_views: maximum number of views to show.
        pad: pixel padding between cells.
        label_height: pixel height for row labels.
        depth_target_views: optional ``[V, H, W]`` or ``[V, 1, H, W]`` depth maps.

    Returns:
        PIL Image with rows: render, target, (optionally) depth.
    """
    view_count = min(int(max_views), int(render_views.shape[0]), int(target_views.shape[0]))
    if view_count < 1:
        raise ValueError('No target views available for visualization.')

    render_images = [_tensor_to_uint8(render_views[idx]) for idx in range(view_count)]
    target_images = [_tensor_to_uint8(target_views[idx]) for idx in range(view_count)]
    height, width = render_images[0].shape[:2]

    has_depth = (
        depth_target_views is not None
        and int(depth_target_views.shape[0]) >= view_count
    )
    if has_depth:
        depth_images = [
            _depth_to_uint8(depth_target_views[idx], apply_cmap=True)
            for idx in range(view_count)
        ]
        dh, dw = depth_images[0].shape[:2]
        if (dh, dw) != (height, width):
            depth_images = [
                np.array(Image.fromarray(x).resize((width, height)))
                for x in depth_images
            ]
        n_rows = 3
        row_labels = ['render', 'target', 'depth']
    else:
        n_rows = 2
        row_labels = ['render', 'target']

    canvas_height = pad * (n_rows + 1) + label_height * n_rows + height * n_rows
    canvas_width = pad + view_count * (width + pad)
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(246, 246, 246))
    draw = ImageDraw.Draw(canvas)

    row_data = [render_images, target_images]
    if has_depth:
        row_data.append(depth_images)

    y = pad
    for row_idx, label in enumerate(row_labels):
        draw.text((pad, y), label, fill=(24, 24, 24))
        y += label_height
        for idx, img in enumerate(row_data[row_idx]):
            x = pad + idx * (width + pad)
            canvas.paste(Image.fromarray(img), (x, y))
        y += height + pad

    return canvas


def _sanitize_filename(name: str) -> str:
    """Replace non-alphanumeric characters with underscores for safe filenames."""
    sanitized = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('._')
    return sanitized or 'scene'


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert ``[C, H, W]`` float tensor in ``[0, 1]`` to ``[H, W, C]`` uint8 array."""
    image = image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.asarray(image * 255.0 + 0.5, dtype=np.uint8)


def _depth_to_uint8(depth: torch.Tensor, apply_cmap: bool = False) -> np.ndarray:
    """Convert ``[H, W]`` or ``[1, H, W]`` depth tensor to ``[H, W, 3]`` uint8 array.

    Args:
        depth: depth map tensor.
        apply_cmap: if True, apply a turbo colormap via OpenCV.

    Returns:
        uint8 RGB array of shape ``[H, W, 3]``.
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)
    d = depth.detach().float().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    d_u8 = np.clip(np.round(d * 255.0), 0, 255).astype(np.uint8)
    if apply_cmap:
        try:
            import cv2
            d_u8 = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
            d_u8 = d_u8[:, :, ::-1]
        except Exception:
            d_u8 = np.stack([d_u8] * 3, axis=-1)
    else:
        d_u8 = np.stack([d_u8] * 3, axis=-1)
    return d_u8
