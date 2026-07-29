from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .geodesy import LocalTangentPlane


@dataclass(frozen=True, slots=True)
class StaticScene:
    building_vertices: np.ndarray
    bridge_vertices: np.ndarray
    origin_latitude_deg: float
    origin_longitude_deg: float


def _height(tags: dict[str, str]) -> float:
    raw_height = tags.get("height", "").lower().replace("m", "").strip()
    try:
        return float(raw_height)
    except ValueError:
        pass
    try:
        levels = float(tags.get("building:levels", ""))
        return max(3.2, levels * 3.2)
    except ValueError:
        return 12.0


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - np.roll(points[:, 0], -1) * points[:, 1]
        )
    )


def _inside_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    ab = cross(b - a, point - a)
    bc = cross(c - b, point - b)
    ca = cross(a - c, point - c)
    return ab >= -1e-7 and bc >= -1e-7 and ca >= -1e-7


def _triangulate(points: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear-clip a simple counter-clockwise polygon."""

    if len(points) < 3:
        return []
    if _signed_area(points) < 0.0:
        points[:] = points[::-1]
    remaining = list(range(len(points)))
    triangles: list[tuple[int, int, int]] = []
    guard = len(points) ** 2
    while len(remaining) > 3 and guard:
        guard -= 1
        clipped = False
        for offset, current in enumerate(remaining):
            previous = remaining[offset - 1]
            following = remaining[(offset + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            ab, bc = b - a, c - b
            if ab[0] * bc[1] - ab[1] * bc[0] <= 1e-8:
                continue
            if any(
                _inside_triangle(points[index], a, b, c)
                for index in remaining
                if index not in (previous, current, following)
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[offset]
            clipped = True
            break
        if not clipped:
            break
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    return triangles


def _vertex(position: tuple[float, float, float], normal: tuple[float, float, float],
            material: float) -> list[float]:
    return [*position, *normal, material]


def _building_mesh(points: np.ndarray, height_m: float) -> list[list[float]]:
    vertices: list[list[float]] = []
    for index, a in enumerate(points):
        b = points[(index + 1) % len(points)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 0.05:
            continue
        normal = (edge[1] / length, 0.0, -edge[0] / length)
        p0, p1 = (a[0], 0.0, a[1]), (b[0], 0.0, b[1])
        p2, p3 = (b[0], height_m, b[1]), (a[0], height_m, a[1])
        vertices.extend(
            [
                _vertex(p0, normal, 0.0), _vertex(p1, normal, 0.0),
                _vertex(p2, normal, 0.0), _vertex(p0, normal, 0.0),
                _vertex(p2, normal, 0.0), _vertex(p3, normal, 0.0),
            ]
        )
    roof_normal = (0.0, 1.0, 0.0)
    for a, b, c in _triangulate(points.copy()):
        for index in (a, b, c):
            point = points[index]
            vertices.append(
                _vertex((point[0], height_m, point[1]), roof_normal, 1.0)
            )
    return vertices


def _bridge_mesh(points: np.ndarray, width_m: float, elevation_m: float) -> list[list[float]]:
    vertices: list[list[float]] = []
    for a, b in zip(points[:-1], points[1:]):
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 0.1:
            continue
        perpendicular = np.array([-edge[1], edge[0]]) / length * width_m * 0.5
        corners = (a - perpendicular, a + perpendicular, b + perpendicular, b - perpendicular)
        for indices in ((0, 1, 2), (0, 2, 3)):
            for index in indices:
                point = corners[index]
                vertices.append(
                    _vertex((point[0], elevation_m, point[1]), (0.0, 1.0, 0.0), 2.0)
                )
    return vertices


def build_scene(
    osm: dict[str, Any], origin_latitude_deg: float, origin_longitude_deg: float
) -> StaticScene:
    plane = LocalTangentPlane(origin_latitude_deg, origin_longitude_deg)
    buildings: list[list[float]] = []
    bridges: list[list[float]] = []
    for element in osm.get("elements", []):
        geometry = element.get("geometry", [])
        tags = element.get("tags", {})
        if len(geometry) < 2:
            continue
        local = np.array(
            [
                plane.to_local(node["lat"], node["lon"])[[0, 2]]
                for node in geometry
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(local[0] - local[-1]) < 0.05:
            local = local[:-1]
        if "building" in tags and len(local) >= 3:
            buildings.extend(_building_mesh(local, _height(tags)))
        if "bridge" in tags and len(local) >= 2:
            try:
                width = float(tags.get("width", "").replace("m", "").strip())
            except ValueError:
                width = 12.0
            bridges.extend(_bridge_mesh(local, width, 7.0))
    empty = np.empty((0, 7), dtype=np.float32)
    return StaticScene(
        np.asarray(buildings, dtype=np.float32) if buildings else empty,
        np.asarray(bridges, dtype=np.float32) if bridges else empty,
        origin_latitude_deg,
        origin_longitude_deg,
    )


def save_scene(scene: StaticScene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        building_vertices=scene.building_vertices,
        bridge_vertices=scene.bridge_vertices,
        origin=np.array(
            [scene.origin_latitude_deg, scene.origin_longitude_deg],
            dtype=np.float64,
        ),
    )


def load_scene(path: Path) -> StaticScene:
    with np.load(path) as data:
        origin = data["origin"]
        return StaticScene(
            data["building_vertices"].copy(),
            data["bridge_vertices"].copy(),
            float(origin[0]),
            float(origin[1]),
        )

