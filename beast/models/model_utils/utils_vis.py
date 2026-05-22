"""Visualization and camera utility functions for ERayZer rendering."""

import torch


@torch.no_grad()
def build_stepback_c2ws(frame_c2ws: torch.Tensor, step_back_distance: float) -> torch.Tensor:
    """Build camera-to-world transforms stepped back along each camera's local -Z axis.

    Args:
        frame_c2ws: camera-to-world (OpenCV-style) transforms of shape (..., 4, 4).
        step_back_distance: scalar distance to move along each camera's local -Z axis.

    Returns:
        stepback_c2ws with same shape as frame_c2ws.
    """
    R = frame_c2ws[..., :3, :3]
    t = frame_c2ws[..., :3, 3]

    z_world = R[..., :, 2]
    t_new = t - step_back_distance * z_world

    c2w_step = frame_c2ws.clone()
    c2w_step[..., :3, 3] = t_new
    return c2w_step
