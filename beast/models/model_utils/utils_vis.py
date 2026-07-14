"""Visualization and camera utility functions for Sable rendering."""

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

# converts an OpenCV-style c2w (camera looks down +Z) into the OpenGL convention
# (camera looks down -Z) used when building the frustum mesh below
_OPENGL_CORRECTION = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
])


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid/affine transform to a set of 3-D points.

    Args:
        transform: float array of shape (4, 4).
        points: float array of shape (N, 3).

    Returns:
        transformed points, float array of shape (N, 3).
    """
    return points @ transform[:3, :3].T + transform[:3, 3]


def add_scene_cam(
    scene: trimesh.Scene,
    c2w: np.ndarray,
    edge_color: np.ndarray,
    imsize: tuple[int, int] | None = None,
    screen_width: float = 0.03,
) -> None:
    """Add a small cone-shaped camera-frustum mesh to a trimesh scene, in place.

    Args:
        scene: trimesh scene to append the frustum mesh to.
        c2w: camera-to-world transform of shape (4, 4).
        edge_color: RGB color, uint8 array of shape (3,), used for the frustum faces.
        imsize: (width, height) used to derive the frustum's aspect ratio.  Defaults
            to a square frustum when ``None``.
        screen_width: physical size of the drawn frustum.
    """
    if imsize is not None:
        width_px, height_px = imsize
    else:
        width_px = height_px = 1

    focal = min(height_px, width_px) * 1.1

    # build a cone whose tip sits at the camera's optical center
    height = focal * screen_width / height_px
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = width_px / height_px
    transform = c2w @ _OPENGL_CORRECTION @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # duplicate/offset vertices to build a hollow pyramid-like frustum mesh with
    # pseudo-edges for a wireframe look
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(4)).as_matrix()
    vertices = cam.vertices
    vertices_offset = 0.9 * cam.vertices
    vertices = np.r_[vertices, vertices_offset, _transform_points(rot2, cam.vertices)]
    vertices = _transform_points(transform, vertices)

    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b2, a2))
        faces.append((a2, c, c2))
        faces.append((c2, b2, b))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    for i, face in enumerate(cam.faces):
        if 0 in face:
            continue
        if i == 1 or i == 5:
            a, b, c = face
            faces.append((a, b, c))

    vertices[:, [1, 2]] *= -1
    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color

    scene.add_geometry(cam)


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
