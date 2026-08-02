from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .geodesy import LocalTangentPlane

SURFACE_WALL = 0.0
SURFACE_ROOF = 1.0
SURFACE_BRIDGE = 2.0
SURFACE_ROAD = 3.0
SURFACE_VEGETATION = 4.0
SURFACE_FOOTWAY = 5.0
SURFACE_CYCLEWAY = 6.0
SURFACE_SPORT = 7.0
SURFACE_METAL = 8.0
SURFACE_WOOD = 9.0
SURFACE_LAMP = 10.0
SURFACE_FOLIAGE = 11.0
SURFACE_CONCRETE = 12.0
SURFACE_PLAYGROUND = 13.0
SURFACE_GARDEN = 14.0
SURFACE_TRAIL = 15.0
SURFACE_GRASS_BLADE = 16.0

# The final vertex channel is facade style for buildings and lightweight
# source semantics for linear batches.  Keeping the flag in the existing
# layout avoids another per-vertex attribute and draw call.
LINEAR_STYLE_DEFAULT = 0.0
LINEAR_STYLE_STEPS = 1.0
LINEAR_STYLE_EMBANKMENT = 2.0

FACADE_GENERIC = 0.0
FACADE_GLASS_BLUE = 1.0
FACADE_GOLD_63 = 2.0
FACADE_RESIDENTIAL = 3.0
FACADE_INSTITUTIONAL = 4.0
FACADE_PARC1 = 5.0
FACADE_HOTEL = 6.0
FACADE_FKI = 7.0
FACADE_NATIONAL_ASSEMBLY = 8.0
FACADE_PARC1_STRUCTURE = 9.0
FACADE_ASSEMBLY_COLUMN = 10.0
FACADE_ASSEMBLY_DOME = 12.0


@dataclass(frozen=True, slots=True)
class StaticScene:
    building_vertices: np.ndarray
    bridge_vertices: np.ndarray
    road_vertices: np.ndarray
    vegetation_vertices: np.ndarray
    detail_vertices: np.ndarray
    water_mask: np.ndarray
    water_mask_bounds: np.ndarray
    terrain_height_m: np.ndarray
    terrain_bounds: np.ndarray
    elevation_datum_m: float
    origin_latitude_deg: float
    origin_longitude_deg: float
    snapshot_utc: str


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


def _minimum_height(tags: dict[str, str]) -> float:
    raw_height = tags.get("min_height", "").lower().replace("m", "").strip()
    try:
        return max(float(raw_height), 0.0)
    except ValueError:
        pass
    try:
        return max(float(tags.get("building:min_level", "")) * 3.2, 0.0)
    except ValueError:
        return 0.0


def _names(tags: dict[str, str]) -> str:
    return " ".join(
        tags.get(key, "") for key in ("name", "name:en", "official_name")
    ).casefold()


def _landmark_kind(tags: dict[str, str]) -> str:
    names = _names(tags)
    if any(
        value in names
        for value in ("63시티", "63 city", "63square", "63스퀘어")
    ):
        return "63_city"
    if "parc.1 tower1" in names or "파크원 타워1" in names:
        return "parc1_tower1"
    if (
        ("tower2" in names and ("parc.1" in names or "nh financial" in names))
        or "nh금융타워" in names
    ):
        return "parc1_tower2"
    if "fairmont ambassador" in names or "페어몬트 앰배서더" in names:
        return "parc1_hotel"
    if "fki tow" in names or "전경련회관" in names:
        return "fki"
    if "national assembly of korea" in names or "국회의사당" in names:
        return "national_assembly"
    for key in ("one_ifc", "two_ifc", "three_ifc"):
        if key.replace("_", " ") in names:
            return key
    if "conrad seoul" in names or "콘래드 서울" in names:
        return "conrad"
    return ""


def _landmark_height(tags: dict[str, str], fallback_m: float) -> float:
    """Apply completed heights published by each building's architect."""

    return {
        "parc1_tower1": 318.0,
        "parc1_tower2": 246.0,
        "parc1_hotel": 101.0,
        "fki": 240.0,
    }.get(_landmark_kind(tags), fallback_m)


def _facade_style(tags: dict[str, str]) -> float:
    names = _names(tags)
    landmark = _landmark_kind(tags)
    building = tags.get("building", "").lower()
    material = tags.get("building:material", "").lower()
    colour = tags.get("building:colour", "").lower()
    if landmark == "63_city":
        return FACADE_GOLD_63
    if landmark in {"parc1_tower1", "parc1_tower2"}:
        return FACADE_PARC1
    if landmark == "fki":
        return FACADE_FKI
    if landmark == "national_assembly":
        return FACADE_NATIONAL_ASSEMBLY
    if landmark in {"one_ifc", "two_ifc", "three_ifc"} or "국제금융" in names:
        return FACADE_GLASS_BLUE
    if landmark in {"conrad", "parc1_hotel"} or building == "hotel":
        return FACADE_HOTEL
    if building in {"apartments", "residential", "officetel"}:
        return FACADE_RESIDENTIAL
    if building in {
        "government", "public", "civic", "school", "university", "church"
    } or tags.get("roof:shape") == "dome":
        return FACADE_INSTITUTIONAL
    if material == "glass" or colour in {"#547bbc", "#2240ca", "blue"}:
        return FACADE_GLASS_BLUE
    if colour in {"gold", "golden"}:
        return FACADE_GOLD_63
    return FACADE_GENERIC


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


def _vertex(
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    surface: float,
    surface_uv: tuple[float, float] = (0.0, 0.0),
    facade_style: float = FACADE_GENERIC,
) -> list[float]:
    return [*position, *normal, surface, *surface_uv, facade_style]


def _building_mesh(
    points: np.ndarray,
    height_m: float,
    minimum_height_m: float = 0.0,
    facade_style: float = FACADE_GENERIC,
) -> list[list[float]]:
    if _signed_area(points) < 0.0:
        points = points[::-1].copy()
    vertices: list[list[float]] = []
    base_height = min(max(minimum_height_m, 0.0), height_m - 0.1)
    band_count = 16 if facade_style == FACADE_GOLD_63 else 1
    centre = points.mean(axis=0)
    perimeter_u = np.concatenate(
        (
            np.array([0.0]),
            np.cumsum(
                np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            ),
        )
    )

    def ring(height: float) -> np.ndarray:
        if facade_style != FACADE_GOLD_63:
            return points
        alpha = (height - base_height) / max(height_m - base_height, 0.1)
        # 63 City has a visibly narrower, softly curved upper silhouette.
        scale = 1.0 - 0.14 * alpha * alpha
        return centre + (points - centre) * scale

    heights = np.linspace(base_height, height_m, band_count + 1)
    for band in range(band_count):
        lower, upper = ring(heights[band]), ring(heights[band + 1])
        for index, a in enumerate(lower):
            next_index = (index + 1) % len(points)
            b, c, d = lower[next_index], upper[next_index], upper[index]
            p0 = np.array((a[0], heights[band], a[1]))
            p1 = np.array((b[0], heights[band], b[1]))
            p2 = np.array((c[0], heights[band + 1], c[1]))
            p3 = np.array((d[0], heights[band + 1], d[1]))
            normal_vector = np.cross(p3 - p0, p1 - p0)
            normal_length = float(np.linalg.norm(normal_vector))
            if normal_length < 0.05:
                continue
            normal = tuple(normal_vector / normal_length)
            u0, u1 = float(perimeter_u[index]), float(perimeter_u[index + 1])
            v0, v1 = heights[band] - base_height, heights[band + 1] - base_height
            vertices.extend(
                [
                    _vertex(tuple(p0), normal, SURFACE_WALL, (u0, v0), facade_style),
                    _vertex(tuple(p1), normal, SURFACE_WALL, (u1, v0), facade_style),
                    _vertex(tuple(p2), normal, SURFACE_WALL, (u1, v1), facade_style),
                    _vertex(tuple(p0), normal, SURFACE_WALL, (u0, v0), facade_style),
                    _vertex(tuple(p2), normal, SURFACE_WALL, (u1, v1), facade_style),
                    _vertex(tuple(p3), normal, SURFACE_WALL, (u0, v1), facade_style),
                ]
            )
    roof_normal = (0.0, 1.0, 0.0)
    roof_points = ring(height_m)
    for a, b, c in _triangulate(roof_points.copy()):
        for index in (a, b, c):
            point = roof_points[index]
            vertices.append(
                _vertex(
                    (point[0], height_m, point[1]),
                    roof_normal,
                    SURFACE_ROOF,
                    (point[0], point[1]),
                    facade_style,
                )
            )
    return vertices


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return whether a 2D point lies inside a simple polygon."""

    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = previous
        x1, y1 = current
        if (y0 > y) != (y1 > y):
            crossing = (x1 - x0) * (y - y0) / (y1 - y0) + x0
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _principal_footprint(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    covariance = np.cov((points - centre).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_u = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis_v = np.array([-axis_u[1], axis_u[0]])
    projections = np.column_stack(
        ((points - centre) @ axis_u, (points - centre) @ axis_v)
    )
    extents = np.max(np.abs(projections), axis=0)
    return centre, axis_u, axis_v, extents


def _face_vertices(
    corners: list[np.ndarray], indices: tuple[int, int, int, int],
    surface: float, style: float,
) -> list[list[float]]:
    positions = [corners[index] for index in indices]
    normal_vector = np.cross(
        positions[1] - positions[0], positions[2] - positions[0]
    )
    length = float(np.linalg.norm(normal_vector))
    if length < 1e-6:
        return []
    normal = tuple(normal_vector / length)
    result: list[list[float]] = []
    for index in (0, 1, 2, 0, 2, 3):
        position = positions[index]
        result.append(
            _vertex(
                tuple(position), normal, surface,
                (float(position[0]), float(position[1])), style,
            )
        )
    return result


def _oriented_box_mesh(
    centre_xz: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray,
    size_u_m: float, size_v_m: float, base_m: float, top_m: float,
    style: float, surface: float = SURFACE_WALL,
) -> list[list[float]]:
    corners: list[np.ndarray] = []
    for height in (base_m, top_m):
        for sign_u, sign_v in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            xz = (
                centre_xz
                + axis_u * sign_u * size_u_m * 0.5
                + axis_v * sign_v * size_v_m * 0.5
            )
            corners.append(np.array([xz[0], height, xz[1]], dtype=np.float64))
    output: list[list[float]] = []
    for face in (
        (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
        (3, 7, 4, 0), (4, 7, 6, 5), (0, 1, 2, 3),
    ):
        output.extend(_face_vertices(corners, face, surface, style))
    return output


def _column_mesh(
    centre_xz: np.ndarray, radius_m: float, base_m: float, top_m: float,
    style: float, sides: int = 8,
) -> list[list[float]]:
    output: list[list[float]] = []
    for side in range(sides):
        angles = (
            2.0 * math.pi * side / sides,
            2.0 * math.pi * (side + 1) / sides,
        )
        xz0 = centre_xz + radius_m * np.array(
            [math.cos(angles[0]), math.sin(angles[0])]
        )
        xz1 = centre_xz + radius_m * np.array(
            [math.cos(angles[1]), math.sin(angles[1])]
        )
        corners = [
            np.array([xz0[0], base_m, xz0[1]]),
            np.array([xz1[0], base_m, xz1[1]]),
            np.array([xz1[0], top_m, xz1[1]]),
            np.array([xz0[0], top_m, xz0[1]]),
        ]
        output.extend(_face_vertices(corners, (0, 1, 2, 3), SURFACE_WALL, style))
    return output


def _skillion_mesh(
    points: np.ndarray, height_m: float, minimum_height_m: float,
    roof_height_m: float, facade_style: float,
) -> list[list[float]]:
    """Build the OSM Simple 3D Buildings single-slope roof profile."""

    if _signed_area(points) < 0.0:
        points = points[::-1].copy()
    base = min(max(minimum_height_m, 0.0), height_m - 0.1)
    eave = max(base, height_m - max(roof_height_m, 0.1))
    _, slope_axis, _, _ = _principal_footprint(points)
    projection = points @ slope_axis
    span = max(float(np.ptp(projection)), 0.1)
    top_heights = eave + (projection - projection.min()) / span * (height_m - eave)
    perimeter_u = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
            ),
        )
    )
    output: list[list[float]] = []
    for index, point in enumerate(points):
        following = (index + 1) % len(points)
        next_point = points[following]
        corners = [
            np.array([point[0], base, point[1]]),
            np.array([next_point[0], base, next_point[1]]),
            np.array([next_point[0], top_heights[following], next_point[1]]),
            np.array([point[0], top_heights[index], point[1]]),
        ]
        face = _face_vertices(corners, (0, 1, 2, 3), SURFACE_WALL, facade_style)
        u0 = float(perimeter_u[index])
        u1 = float(perimeter_u[index + 1])
        for vertex, u_coordinate in zip(
            face, (u0, u1, u1, u0, u1, u0)
        ):
            vertex[7] = u_coordinate
            vertex[8] = float(vertex[1] - base)
        output.extend(face)
    roof_positions = np.column_stack((points[:, 0], top_heights, points[:, 1]))
    for a, b, c in _triangulate(points.copy()):
        triangle = [roof_positions[a], roof_positions[b], roof_positions[c]]
        normal_vector = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        normal = tuple(
            normal_vector / max(float(np.linalg.norm(normal_vector)), 1e-6)
        )
        for position in triangle:
            output.append(
                _vertex(
                    tuple(position), normal, SURFACE_ROOF,
                    tuple(position[[0, 2]]), facade_style,
                )
            )
    return output


def _landmark_detail_mesh(
    points: np.ndarray, height_m: float, landmark: str,
) -> list[list[float]]:
    """Add only externally documented, silhouette-relevant landmark parts."""

    centre, axis_u, axis_v, extents = _principal_footprint(points)
    output: list[list[float]] = []
    if landmark in {"parc1_tower1", "parc1_tower2"}:
        corners = [
            centre + axis_u * sign_u * extents[0]
            + axis_v * sign_v * extents[1]
            for sign_u, sign_v in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        for corner in corners:
            output.extend(
                _oriented_box_mesh(
                    corner, axis_u, axis_v, 2.45, 2.45, 1.0, height_m - 0.8,
                    FACADE_PARC1_STRUCTURE,
                )
            )
        beam_heights = list(np.arange(30.0, height_m - 12.0, 32.0))
        beam_heights.append(height_m - 3.4)
        for beam_height in beam_heights:
            for first, second in zip(corners, corners[1:] + corners[:1]):
                edge = second - first
                edge_length = float(np.linalg.norm(edge))
                edge_axis = edge / max(edge_length, 1e-6)
                edge_normal = np.array([-edge_axis[1], edge_axis[0]])
                output.extend(
                    _oriented_box_mesh(
                        0.5 * (first + second), edge_axis, edge_normal,
                        edge_length, 1.55, beam_height - 0.72,
                        beam_height + 0.72, FACADE_PARC1_STRUCTURE,
                    )
                )
    elif landmark == "national_assembly":
        # The National Assembly's official description specifies 24 exterior
        # octagonal columns. The named outline includes the chamber wings, so
        # the colonnade is contracted to the central block rather than placed
        # around the full concave complex perimeter.
        column_top = min(height_m * 0.94, 18.0)
        column_extents = extents * np.array([0.58, 0.60])
        for sign_v in (-1.0, 1.0):
            for fraction in np.linspace(-0.78, 0.78, 8):
                position = (
                    centre + axis_u * column_extents[0] * fraction
                    + axis_v * sign_v * column_extents[1]
                )
                output.extend(
                    _column_mesh(
                        position, 0.82, 0.8, column_top,
                        FACADE_ASSEMBLY_COLUMN,
                    )
                )
        for sign_u in (-1.0, 1.0):
            for fraction in np.linspace(-0.58, 0.58, 4):
                position = (
                    centre + axis_u * sign_u * column_extents[0]
                    + axis_v * column_extents[1] * fraction
                )
                output.extend(
                    _column_mesh(
                        position, 0.82, 0.8, column_top,
                        FACADE_ASSEMBLY_COLUMN,
                    )
                )
    elif landmark == "fki":
        # The architect documents a 10-degree photovoltaic rooftop canopy.
        half_u, half_v = extents * np.array([0.72, 0.66])
        rise = math.tan(math.radians(10.0)) * half_v
        xz = [
            centre - axis_u * half_u - axis_v * half_v,
            centre + axis_u * half_u - axis_v * half_v,
            centre + axis_u * half_u + axis_v * half_v,
            centre - axis_u * half_u + axis_v * half_v,
        ]
        top = [
            np.array(
                [
                    point[0],
                    height_m + 1.5 + (-rise if index < 2 else rise),
                    point[1],
                ]
            )
            for index, point in enumerate(xz)
        ]
        corners = [point - np.array([0.0, 0.55, 0.0]) for point in top] + top
        for face in (
            (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
            (3, 7, 4, 0), (4, 7, 6, 5),
        ):
            output.extend(_face_vertices(corners, face, SURFACE_ROOF, FACADE_FKI))
    return output


def _dome_mesh(
    points: np.ndarray,
    height_m: float,
    minimum_height_m: float,
    facade_style: float,
) -> list[list[float]]:
    if _signed_area(points) < 0.0:
        points = points[::-1].copy()
    centre = points.mean(axis=0)
    base_height = min(max(minimum_height_m, 0.0), height_m - 0.1)
    rings = 8
    ring_points: list[np.ndarray] = []
    ring_heights: list[float] = []
    for ring_index in range(rings):
        angle = (ring_index / rings) * math.pi * 0.5
        ring_points.append(centre + (points - centre) * math.cos(angle))
        ring_heights.append(
            base_height + (height_m - base_height) * math.sin(angle)
        )
    vertices: list[list[float]] = []
    for ring_index in range(rings - 1):
        lower, upper = ring_points[ring_index], ring_points[ring_index + 1]
        for index, a in enumerate(lower):
            following = (index + 1) % len(points)
            p0 = np.array((a[0], ring_heights[ring_index], a[1]))
            p1 = np.array(
                (
                    lower[following, 0],
                    ring_heights[ring_index],
                    lower[following, 1],
                )
            )
            p2 = np.array(
                (
                    upper[following, 0],
                    ring_heights[ring_index + 1],
                    upper[following, 1],
                )
            )
            p3 = np.array(
                (
                    upper[index, 0],
                    ring_heights[ring_index + 1],
                    upper[index, 1],
                )
            )
            normal_vector = np.cross(p3 - p0, p1 - p0)
            normal = tuple(normal_vector / np.linalg.norm(normal_vector))
            vertices.extend(
                [
                    _vertex(tuple(p0), normal, SURFACE_ROOF, tuple(p0[[0, 2]]), facade_style),
                    _vertex(tuple(p1), normal, SURFACE_ROOF, tuple(p1[[0, 2]]), facade_style),
                    _vertex(tuple(p2), normal, SURFACE_ROOF, tuple(p2[[0, 2]]), facade_style),
                    _vertex(tuple(p0), normal, SURFACE_ROOF, tuple(p0[[0, 2]]), facade_style),
                    _vertex(tuple(p2), normal, SURFACE_ROOF, tuple(p2[[0, 2]]), facade_style),
                    _vertex(tuple(p3), normal, SURFACE_ROOF, tuple(p3[[0, 2]]), facade_style),
                ]
            )
    top_ring = ring_points[-1]
    top = np.array((centre[0], height_m, centre[1]))
    for index, point in enumerate(top_ring):
        following = top_ring[(index + 1) % len(top_ring)]
        p0 = np.array((point[0], ring_heights[-1], point[1]))
        p1 = np.array((following[0], ring_heights[-1], following[1]))
        normal_vector = np.cross(top - p0, p1 - p0)
        normal = tuple(normal_vector / np.linalg.norm(normal_vector))
        vertices.extend(
            [
                _vertex(tuple(p0), normal, SURFACE_ROOF, tuple(p0[[0, 2]]), facade_style),
                _vertex(tuple(p1), normal, SURFACE_ROOF, tuple(p1[[0, 2]]), facade_style),
                _vertex(tuple(top), normal, SURFACE_ROOF, tuple(top[[0, 2]]), facade_style),
            ]
        )
    return vertices


def _bridge_mesh(
    points: np.ndarray,
    width_m: float,
    elevation_m: float,
    material: float = 2.0,
) -> list[list[float]]:
    return _linear_feature_mesh(points, width_m, elevation_m, material)


def _subdivide_polyline(
    points: np.ndarray, maximum_segment_length_m: float
) -> np.ndarray:
    """Split long plan segments without changing the surveyed centreline."""

    if len(points) < 2 or maximum_segment_length_m <= 0.0:
        return np.asarray(points, dtype=np.float64)
    samples = [np.asarray(points[0], dtype=np.float64)]
    for start, end in zip(points[:-1], points[1:]):
        edge = np.asarray(end - start, dtype=np.float64)
        divisions = max(
            1,
            int(
                math.ceil(
                    float(np.linalg.norm(edge)) / maximum_segment_length_m
                )
            ),
        )
        samples.extend(
            start + edge * (index / divisions)
            for index in range(1, divisions + 1)
        )
    return np.asarray(samples, dtype=np.float64)


def _linear_feature_mesh(
    points: np.ndarray,
    width_m: float,
    elevation_m: float,
    material: float,
    maximum_segment_length_m: float = 0.0,
    linear_style: float = LINEAR_STYLE_DEFAULT,
) -> list[list[float]]:
    """Create a crack-free strip with bounded miter joins.

    Each segment remains a six-vertex quad for the renderer's longitudinal UV
    and kerb passes, but adjacent quads use exactly the same join corners.
    Long road spans can be subdivided so the GPU terrain sampler follows the
    official height field instead of bridging over it with a single plane.
    """

    points = _subdivide_polyline(points, maximum_segment_length_m)
    if len(points) < 2:
        return []
    edges = points[1:] - points[:-1]
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths >= 0.1
    if not np.all(valid):
        keep = np.concatenate(([True], valid))
        points = points[keep]
        if len(points) < 2:
            return []
        edges = points[1:] - points[:-1]
        lengths = np.linalg.norm(edges, axis=1)
    directions = edges / lengths[:, None]
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    half_width = width_m * 0.5
    offsets = np.empty_like(points)
    offsets[0] = normals[0] * half_width
    offsets[-1] = normals[-1] * half_width
    for index in range(1, len(points) - 1):
        summed = normals[index - 1] + normals[index]
        summed_length = float(np.linalg.norm(summed))
        if summed_length < 1e-6:
            offsets[index] = normals[index] * half_width
            continue
        miter = summed / summed_length
        denominator = float(np.dot(miter, normals[index]))
        if abs(denominator) < 1e-4:
            offsets[index] = normals[index] * half_width
            continue
        miter_length = np.clip(
            half_width / denominator,
            -half_width * 2.5,
            half_width * 2.5,
        )
        offsets[index] = miter * miter_length

    vertices: list[list[float]] = []
    for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
        corners = (
            a - offsets[index],
            a + offsets[index],
            b + offsets[index + 1],
            b - offsets[index + 1],
        )
        for indices in ((0, 1, 2), (0, 2, 3)):
            for index in indices:
                point = corners[index]
                vertices.append(
                    _vertex(
                        (point[0], elevation_m, point[1]),
                        (0.0, 1.0, 0.0),
                        material,
                        (point[0], point[1]),
                        linear_style,
                    )
                )
    return vertices


def _road_surface(tags: dict[str, str]) -> float:
    highway = tags.get("highway", "")
    if highway == "steps" and tags.get("surface") == "asphalt":
        return SURFACE_ROAD
    if highway == "cycleway":
        return SURFACE_CYCLEWAY
    if highway in {"footway", "pedestrian", "steps"}:
        return SURFACE_FOOTWAY
    if highway in {"path", "bridleway"}:
        return SURFACE_TRAIL
    return SURFACE_ROAD


def _linear_style(tags: dict[str, str]) -> float:
    if tags.get("highway") == "steps":
        return LINEAR_STYLE_STEPS
    if tags.get("embankment") == "yes" or tags.get("man_made") == "embankment":
        return LINEAR_STYLE_EMBANKMENT
    return LINEAR_STYLE_DEFAULT


def _surface_mesh(
    points: np.ndarray, elevation_m: float, material: float
) -> list[list[float]]:
    vertices: list[list[float]] = []
    normal = (0.0, 1.0, 0.0)
    for a, b, c in _triangulate(points.copy()):
        for index in (a, b, c):
            point = points[index]
            vertices.append(
                _vertex(
                    (point[0], elevation_m, point[1]),
                    normal,
                    material,
                    (point[0], point[1]),
                )
            )
    return vertices


def _road_width(tags: dict[str, str]) -> float:
    raw_width = tags.get("width", "").lower().replace("m", "").strip()
    try:
        return max(float(raw_width), 1.0)
    except ValueError:
        pass
    try:
        return max(float(tags.get("lanes", "")) * 3.2, 2.0)
    except ValueError:
        pass
    return {
        "motorway": 24.0,
        "trunk": 22.0,
        "primary": 18.0,
        "secondary": 14.0,
        "tertiary": 11.0,
        "residential": 7.0,
        "service": 4.0,
        "cycleway": 2.5,
        "footway": 2.2,
        "pedestrian": 4.0,
        "steps": 2.0,
        "path": 1.5,
        "bridleway": 2.0,
    }.get(tags.get("highway", ""), 5.0)


def build_scene(
    osm: dict[str, Any],
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    snapshot_utc: str = "",
) -> StaticScene:
    plane = LocalTangentPlane(origin_latitude_deg, origin_longitude_deg)
    buildings: list[list[float]] = []
    bridges: list[list[float]] = []
    roads: list[list[float]] = []
    vegetation: list[list[float]] = []
    landmark_parents: list[tuple[np.ndarray, str, float]] = []
    building_part_centres: list[np.ndarray] = []
    for element in osm.get("elements", []):
        geometry = element.get("geometry", [])
        tags = element.get("tags", {})
        if len(geometry) < 3:
            continue
        local_polygon = np.array(
            [
                plane.to_local(node["lat"], node["lon"])[[0, 2]]
                for node in geometry
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(local_polygon[0] - local_polygon[-1]) < 0.05:
            local_polygon = local_polygon[:-1]
        if len(local_polygon) < 3:
            continue
        if "building:part" in tags and tags.get("building:part") != "no":
            building_part_centres.append(local_polygon.mean(axis=0))
        landmark = _landmark_kind(tags)
        if landmark and "building" in tags and tags.get("building") != "roof":
            landmark_parents.append(
                (local_polygon, landmark, _facade_style(tags))
            )
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
        if (
            "building" in tags or "building:part" in tags
        ) and len(local) >= 3:
            height = _landmark_height(tags, _height(tags))
            minimum_height = _minimum_height(tags)
            style = _facade_style(tags)
            landmark = _landmark_kind(tags)
            if "building:part" in tags and tags.get("building:part") != "no":
                for (
                    parent_polygon,
                    parent_landmark,
                    parent_style,
                ) in landmark_parents:
                    if _point_in_polygon(local.mean(axis=0), parent_polygon):
                        landmark = parent_landmark
                        style = parent_style
                        break
                if (
                    landmark == "national_assembly"
                    and tags.get("roof:shape") == "dome"
                ):
                    style = FACADE_ASSEMBLY_DOME
            is_named_parent = (
                bool(_landmark_kind(tags))
                and "building" in tags
                and tags.get("building") != "roof"
            )
            parent_has_parts = is_named_parent and any(
                _point_in_polygon(centre, local)
                for centre in building_part_centres
            )
            if is_named_parent:
                buildings.extend(
                    _landmark_detail_mesh(local, height, _landmark_kind(tags))
                )
            if parent_has_parts:
                # OSM Simple 3D Buildings parents describe total occupancy;
                # rendering them together with their parts creates duplicate
                # coplanar walls and hides the documented stepped silhouette.
                continue
            if tags.get("roof:shape") == "dome":
                buildings.extend(
                    _dome_mesh(local, height, minimum_height, style)
                )
            elif tags.get("roof:shape") == "skillion":
                raw_roof_height = (
                    tags.get("roof:height", "")
                    .lower()
                    .replace("m", "")
                    .strip()
                )
                try:
                    roof_height = float(raw_roof_height)
                except ValueError:
                    roof_height = min(height * 0.18, 12.0)
                buildings.extend(
                    _skillion_mesh(
                        local, height, minimum_height, roof_height, style
                    )
                )
            else:
                buildings.extend(
                    _building_mesh(local, height, minimum_height, style)
                )
        if "bridge" in tags and len(local) >= 2:
            try:
                width = float(tags.get("width", "").replace("m", "").strip())
            except ValueError:
                width = 12.0
            bridges.extend(_bridge_mesh(local, width, 7.0))
        elif "highway" in tags and len(local) >= 2:
            roads.extend(
                _linear_feature_mesh(
                    local,
                    _road_width(tags),
                    0.06,
                    _road_surface(tags),
                    maximum_segment_length_m=12.0,
                    linear_style=_linear_style(tags),
                )
            )
        is_green = (
            tags.get("leisure") == "park"
            or tags.get("landuse")
            in {"grass", "forest", "recreation_ground"}
            or tags.get("natural") in {"wood", "grassland"}
        )
        if is_green and len(local) >= 3:
            vegetation.extend(_surface_mesh(local, 0.035, 4.0))
    empty = np.empty((0, 10), dtype=np.float32)
    return StaticScene(
        np.asarray(buildings, dtype=np.float32) if buildings else empty,
        np.asarray(bridges, dtype=np.float32) if bridges else empty,
        np.asarray(roads, dtype=np.float32) if roads else empty,
        np.asarray(vegetation, dtype=np.float32) if vegetation else empty,
        empty,
        np.full((1, 1), 255, dtype=np.uint8),
        np.array([-10_000.0, -10_000.0, 10_000.0, 10_000.0], dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.array([-10_000.0, -10_000.0, 10_000.0, 10_000.0], dtype=np.float32),
        0.0,
        origin_latitude_deg,
        origin_longitude_deg,
        snapshot_utc,
    )


def _assemble_rings(
    relation_data: dict[str, Any], role: str
) -> list[list[dict[str, float]]]:
    elements = relation_data.get("elements", [])
    nodes = {
        element["id"]: {"lat": element["lat"], "lon": element["lon"]}
        for element in elements
        if element.get("type") == "node"
    }
    ways = {
        element["id"]: element.get("nodes", [])
        for element in elements
        if element.get("type") == "way"
    }
    relation = next(
        element
        for element in elements
        if element.get("type") == "relation" and element.get("id") == 152336
    )
    remaining = [
        list(ways[member["ref"]])
        for member in relation.get("members", [])
        if member.get("type") == "way"
        and member.get("role") == role
        and member.get("ref") in ways
    ]
    rings: list[list[dict[str, float]]] = []
    while remaining:
        chain = remaining.pop(0)
        while chain[0] != chain[-1]:
            match = next(
                (
                    index
                    for index, candidate in enumerate(remaining)
                    if candidate[0] == chain[-1] or candidate[-1] == chain[-1]
                ),
                None,
            )
            if match is None:
                break
            candidate = remaining.pop(match)
            if candidate[-1] == chain[-1]:
                candidate.reverse()
            chain.extend(candidate[1:])
        if len(chain) >= 4 and chain[0] == chain[-1]:
            rings.append([nodes[node_id] for node_id in chain if node_id in nodes])
    return rings


def build_water_mask(
    relation_data: dict[str, Any],
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    bounds: tuple[float, float, float, float] = (-2_500.0, -2_000.0, 2_500.0, 2_000.0),
    resolution: tuple[int, int] = (1024, 1024),
) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image, ImageDraw

    plane = LocalTangentPlane(origin_latitude_deg, origin_longitude_deg)
    minimum_x, minimum_z, maximum_x, maximum_z = bounds
    width, height = resolution

    def raster_points(ring: list[dict[str, float]]) -> list[tuple[float, float]]:
        result = []
        for point in ring:
            local = plane.to_local(point["lat"], point["lon"])
            pixel_x = (local[0] - minimum_x) / (maximum_x - minimum_x) * (width - 1)
            pixel_y = (local[2] - minimum_z) / (maximum_z - minimum_z) * (height - 1)
            result.append((pixel_x, pixel_y))
        return result

    image = Image.new("L", resolution, 0)
    draw = ImageDraw.Draw(image)
    for ring in _assemble_rings(relation_data, "outer"):
        draw.polygon(raster_points(ring), fill=255)
    for ring in _assemble_rings(relation_data, "inner"):
        draw.polygon(raster_points(ring), fill=0)
    return np.asarray(image, dtype=np.uint8), np.asarray(bounds, dtype=np.float32)


def save_scene(scene: StaticScene, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        building_vertices=scene.building_vertices,
        bridge_vertices=scene.bridge_vertices,
        road_vertices=scene.road_vertices,
        vegetation_vertices=scene.vegetation_vertices,
        detail_vertices=scene.detail_vertices,
        water_mask=scene.water_mask,
        water_mask_bounds=scene.water_mask_bounds,
        terrain_height_m=scene.terrain_height_m,
        terrain_bounds=scene.terrain_bounds,
        elevation_datum_m=np.array([scene.elevation_datum_m], dtype=np.float32),
        origin=np.array(
            [scene.origin_latitude_deg, scene.origin_longitude_deg],
            dtype=np.float64,
        ),
        snapshot_utc=np.array([scene.snapshot_utc]),
    )


def _upgrade_vertex_layout(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.shape[1] == 10:
        return vertices.copy()
    if vertices.shape[1] != 7:
        raise ValueError(f"unsupported static-scene vertex width: {vertices.shape[1]}")
    upgraded = np.zeros((len(vertices), 10), dtype=np.float32)
    upgraded[:, :7] = vertices
    upgraded[:, 7] = vertices[:, 0]
    upgraded[:, 8] = np.where(
        vertices[:, 6] < 0.5, vertices[:, 1], vertices[:, 2]
    )
    return upgraded


def load_scene(path: Path) -> StaticScene:
    with np.load(path) as data:
        origin = data["origin"]
        return StaticScene(
            _upgrade_vertex_layout(data["building_vertices"]),
            _upgrade_vertex_layout(data["bridge_vertices"]),
            _upgrade_vertex_layout(data["road_vertices"])
            if "road_vertices" in data else np.empty((0, 10), dtype=np.float32),
            _upgrade_vertex_layout(data["vegetation_vertices"])
            if "vegetation_vertices" in data
            else np.empty((0, 10), dtype=np.float32),
            _upgrade_vertex_layout(data["detail_vertices"])
            if "detail_vertices" in data
            else np.empty((0, 10), dtype=np.float32),
            data["water_mask"].copy(),
            data["water_mask_bounds"].copy(),
            data["terrain_height_m"].copy(),
            data["terrain_bounds"].copy(),
            float(data["elevation_datum_m"][0]),
            float(origin[0]),
            float(origin[1]),
            str(data["snapshot_utc"][0])
            if "snapshot_utc" in data else "",
        )
