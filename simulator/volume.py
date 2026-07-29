from __future__ import annotations

import numpy as np


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

