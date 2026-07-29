from __future__ import annotations

import numpy as np


def reconstruct_world_position(
    screen_uv: np.ndarray,
    depth_01: float,
    inverse_view_projection: np.ndarray,
) -> np.ndarray:
    """Reconstruct one world point from an OpenGL depth-buffer sample."""

    clip = np.array(
        [
            float(screen_uv[0]) * 2.0 - 1.0,
            float(screen_uv[1]) * 2.0 - 1.0,
            float(depth_01) * 2.0 - 1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    world_h = np.asarray(inverse_view_projection, dtype=np.float64) @ clip
    if abs(float(world_h[3])) < 1e-12:
        raise ValueError("depth sample reconstructs a point at infinity")
    return (world_h[:3] / world_h[3]).astype(np.float64)


def opaque_ray_limit(
    camera_position: np.ndarray,
    ray_direction: np.ndarray,
    opaque_world_position: np.ndarray,
    depth_bias_m: float,
) -> float:
    """Return the conservative ray distance immediately before an opaque hit."""

    ray = np.asarray(ray_direction, dtype=np.float64)
    ray /= np.linalg.norm(ray)
    hit_vector = (
        np.asarray(opaque_world_position, dtype=np.float64)
        - np.asarray(camera_position, dtype=np.float64)
    )
    return max(float(np.dot(hit_vector, ray)) - depth_bias_m, 0.0)


def box_vertices(
    minimum: np.ndarray | tuple[float, float, float],
    maximum: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    x0, y0, z0 = map(float, minimum)
    x1, y1, z1 = map(float, maximum)
    return np.array(
        [
            x0, y0, z0, x1, y0, z0,
            x1, y1, z0, x0, y1, z0,
            x0, y0, z1, x1, y0, z1,
            x1, y1, z1, x0, y1, z1,
        ],
        dtype=np.float32,
    ).reshape(-1, 3)


def active_volume_bounds(
    volume: np.ndarray,
    world_minimum: np.ndarray,
    world_maximum: np.ndarray,
    threshold: float = 1e-8,
    margin_cells: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a conservative world AABB for a z-y-x scalar volume."""

    active = np.argwhere(volume > threshold)
    if not len(active):
        return None
    lower_zyx = np.maximum(active.min(axis=0) - margin_cells, 0)
    upper_zyx = np.minimum(
        active.max(axis=0) + margin_cells + 1,
        np.asarray(volume.shape),
    )
    lower_xyz = lower_zyx[[2, 1, 0]]
    upper_xyz = upper_zyx[[2, 1, 0]]
    resolution_xyz = np.asarray(volume.shape)[[2, 1, 0]]
    extent = np.asarray(world_maximum) - np.asarray(world_minimum)
    minimum = np.asarray(world_minimum) + extent * lower_xyz / resolution_xyz
    maximum = np.asarray(world_minimum) + extent * upper_xyz / resolution_xyz
    return minimum.astype(np.float32), maximum.astype(np.float32)


def active_slice_bounds(
    field_yx: np.ndarray,
    xy_minimum: tuple[float, float],
    xy_maximum: tuple[float, float],
    z_minimum: float,
    z_maximum: float,
    threshold: float = 1e-8,
    margin_cells: int = 1,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a conservative 3D AABB for an analytic-depth 2D field."""

    active = np.argwhere(field_yx > threshold)
    if not len(active):
        return None
    lower_yx = np.maximum(active.min(axis=0) - margin_cells, 0)
    upper_yx = np.minimum(
        active.max(axis=0) + margin_cells + 1,
        np.asarray(field_yx.shape),
    )
    lower_xy = lower_yx[[1, 0]]
    upper_xy = upper_yx[[1, 0]]
    resolution_xy = np.asarray(field_yx.shape)[[1, 0]]
    minimum_xy = np.asarray(xy_minimum) + (
        np.asarray(xy_maximum) - np.asarray(xy_minimum)
    ) * lower_xy / resolution_xy
    maximum_xy = np.asarray(xy_minimum) + (
        np.asarray(xy_maximum) - np.asarray(xy_minimum)
    ) * upper_xy / resolution_xy
    return (
        np.asarray(
            [minimum_xy[0], minimum_xy[1], z_minimum], dtype=np.float32
        ),
        np.asarray(
            [maximum_xy[0], maximum_xy[1], z_maximum], dtype=np.float32
        ),
    )
