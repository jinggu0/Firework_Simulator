from __future__ import annotations

import math
from typing import Any

import numpy as np

from .geodesy import LocalTangentPlane
from .scene import (
    SURFACE_CONCRETE,
    SURFACE_CYCLEWAY,
    SURFACE_FOLIAGE,
    SURFACE_FOOTWAY,
    SURFACE_GRASS_BLADE,
    SURFACE_GARDEN,
    SURFACE_LAMP,
    SURFACE_METAL,
    SURFACE_PLAYGROUND,
    SURFACE_SPORT,
    SURFACE_TRAIL,
    SURFACE_WOOD,
    StaticScene,
    _surface_mesh,
    _vertex,
)
from .vegetation import EVENT_SITE_DETAIL_RADIUS_M


def _box(
    centre: np.ndarray,
    size: tuple[float, float, float],
    surface: float,
    yaw_rad: float = 0.0,
) -> list[list[float]]:
    half = np.asarray(size, dtype=np.float64) * 0.5
    corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.array(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]]
    )
    corners = corners @ rotation.T + centre
    faces = (
        ((0, 1, 2, 3), (0.0, 0.0, -1.0)),
        ((5, 4, 7, 6), (0.0, 0.0, 1.0)),
        ((4, 0, 3, 7), (-1.0, 0.0, 0.0)),
        ((1, 5, 6, 2), (1.0, 0.0, 0.0)),
        ((3, 2, 6, 7), (0.0, 1.0, 0.0)),
        ((4, 5, 1, 0), (0.0, -1.0, 0.0)),
    )
    vertices: list[list[float]] = []
    for indices, local_normal in faces:
        normal = tuple(rotation @ np.asarray(local_normal))
        for triangle in ((0, 1, 2), (0, 2, 3)):
            for offset in triangle:
                point = corners[indices[offset]]
                vertices.append(
                    _vertex(
                        tuple(point),
                        normal,
                        surface,
                        (point[0], point[2]),
                    )
                )
    return vertices


def _cylinder(
    centre_xz: np.ndarray,
    radius_m: float,
    height_m: float,
    surface: float,
    sides: int = 8,
    base_y: float = 0.0,
) -> list[list[float]]:
    vertices: list[list[float]] = []
    for index in range(sides):
        a0 = index / sides * math.tau
        a1 = (index + 1) / sides * math.tau
        p0 = np.array(
            [
                centre_xz[0] + math.cos(a0) * radius_m,
                base_y,
                centre_xz[1] + math.sin(a0) * radius_m,
            ]
        )
        p1 = np.array(
            [
                centre_xz[0] + math.cos(a1) * radius_m,
                base_y,
                centre_xz[1] + math.sin(a1) * radius_m,
            ]
        )
        p2, p3 = p1.copy(), p0.copy()
        p2[1] += height_m
        p3[1] += height_m
        normals = (
            (math.cos(a0), 0.0, math.sin(a0)),
            (math.cos(a1), 0.0, math.sin(a1)),
        )
        vertices.extend(
            [
                _vertex(tuple(p0), normals[0], surface),
                _vertex(tuple(p1), normals[1], surface),
                _vertex(tuple(p2), normals[1], surface),
                _vertex(tuple(p0), normals[0], surface),
                _vertex(tuple(p2), normals[1], surface),
                _vertex(tuple(p3), normals[0], surface),
            ]
        )
    return vertices


def _beam(
    start_xz: np.ndarray,
    end_xz: np.ndarray,
    centre_y: float,
    thickness_m: float,
    surface: float,
) -> list[list[float]]:
    delta = end_xz - start_xz
    length = float(np.linalg.norm(delta))
    if length < 0.05:
        return []
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    centre = np.array(
        [
            (start_xz[0] + end_xz[0]) * 0.5,
            centre_y,
            (start_xz[1] + end_xz[1]) * 0.5,
        ]
    )
    return _box(
        centre,
        (length, thickness_m, thickness_m),
        surface,
        yaw,
    )


def _lamp(position: np.ndarray, yaw: float) -> list[list[float]]:
    vertices = _cylinder(position, 0.09, 6.8, SURFACE_METAL, sides=8)
    head = np.array([position[0], 6.88, position[1]])
    vertices.extend(_box(head, (0.85, 0.18, 0.32), SURFACE_LAMP, yaw))
    return vertices


def _grass_blade(
    position: np.ndarray, height_m: float, width_m: float, yaw: float
) -> list[list[float]]:
    """Two crossed tapered cards with tip weights stored in surface UV.y."""

    vertices: list[list[float]] = []
    for angle in (yaw, yaw + math.pi * 0.5):
        across = np.array([math.cos(angle), 0.0, math.sin(angle)])
        facing = np.array([-math.sin(angle), 0.0, math.cos(angle)])
        left = np.array([position[0], 0.025, position[1]]) - across * width_m
        right = np.array([position[0], 0.025, position[1]]) + across * width_m
        tip = np.array([position[0], height_m, position[1]])
        vertices.extend(
            [
                _vertex(tuple(left), tuple(facing), SURFACE_GRASS_BLADE, (0.0, 0.0)),
                _vertex(tuple(right), tuple(facing), SURFACE_GRASS_BLADE, (1.0, 0.0)),
                _vertex(tuple(tip), tuple(facing), SURFACE_GRASS_BLADE, (0.5, 1.0)),
                _vertex(tuple(right), tuple(-facing), SURFACE_GRASS_BLADE, (1.0, 0.0)),
                _vertex(tuple(left), tuple(-facing), SURFACE_GRASS_BLADE, (0.0, 0.0)),
                _vertex(tuple(tip), tuple(-facing), SURFACE_GRASS_BLADE, (0.5, 1.0)),
            ]
        )
    return vertices


def _bench(position: np.ndarray, yaw: float) -> list[list[float]]:
    centre = np.array([position[0], 0.55, position[1]])
    vertices = _box(centre, (1.8, 0.12, 0.48), SURFACE_WOOD, yaw)
    back_offset = np.array([-math.sin(yaw), 0.0, math.cos(yaw)]) * 0.22
    back = centre + back_offset + np.array([0.0, 0.48, 0.0])
    vertices.extend(_box(back, (1.8, 0.72, 0.10), SURFACE_WOOD, yaw))
    along = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
    for direction in (-0.65, 0.65):
        support = centre + along * direction
        support[1] = 0.27
        vertices.extend(_box(support, (0.09, 0.54, 0.35), SURFACE_METAL, yaw))
    return vertices


def _tree(
    position: np.ndarray, height_m: float, radius_m: float
) -> list[list[float]]:
    trunk_height = height_m * 0.38
    vertices = _cylinder(
        position, max(radius_m * 0.11, 0.16), trunk_height, SURFACE_WOOD, 7
    )
    rings = (
        (0.0, 0.55),
        (0.32, 0.95),
        (0.68, 0.72),
        (1.0, 0.10),
    )
    sides = 8
    crown_base = trunk_height * 0.82
    for ring_index in range(len(rings) - 1):
        y0f, r0f = rings[ring_index]
        y1f, r1f = rings[ring_index + 1]
        y0 = crown_base + y0f * (height_m - crown_base)
        y1 = crown_base + y1f * (height_m - crown_base)
        for index in range(sides):
            a0, a1 = index / sides * math.tau, (index + 1) / sides * math.tau
            points = (
                np.array(
                    [
                        position[0] + math.cos(a0) * radius_m * r0f,
                        y0,
                        position[1] + math.sin(a0) * radius_m * r0f,
                    ]
                ),
                np.array(
                    [
                        position[0] + math.cos(a1) * radius_m * r0f,
                        y0,
                        position[1] + math.sin(a1) * radius_m * r0f,
                    ]
                ),
                np.array(
                    [
                        position[0] + math.cos(a1) * radius_m * r1f,
                        y1,
                        position[1] + math.sin(a1) * radius_m * r1f,
                    ]
                ),
                np.array(
                    [
                        position[0] + math.cos(a0) * radius_m * r1f,
                        y1,
                        position[1] + math.sin(a0) * radius_m * r1f,
                    ]
                ),
            )
            for triangle in ((0, 1, 2), (0, 2, 3)):
                for offset in triangle:
                    point = points[offset]
                    normal = point - np.array(
                        [position[0], (y0 + y1) * 0.5, position[1]]
                    )
                    normal /= max(float(np.linalg.norm(normal)), 1e-6)
                    vertices.append(
                        _vertex(
                            tuple(point),
                            tuple(normal),
                            SURFACE_FOLIAGE,
                            (point[0], point[2]),
                        )
                    )
    return vertices


def _inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, z = point
    result = False
    previous = polygon[-1]
    for current in polygon:
        if (current[1] > z) != (previous[1] > z):
            crossing_x = (
                (previous[0] - current[0])
                * (z - current[1])
                / (previous[1] - current[1])
                + current[0]
            )
            if x < crossing_x:
                result = not result
        previous = current
    return result


def classify_path_surfaces(road_vertices: np.ndarray) -> np.ndarray:
    """Recover imported way widths and assign visible surface families."""

    result = np.asarray(road_vertices, dtype=np.float32).copy()
    for start in range(0, len(result) - 5, 6):
        width = float(np.linalg.norm(result[start, :3] - result[start + 1, :3]))
        if width <= 1.8:
            surface = SURFACE_TRAIL
        elif width <= 2.35:
            surface = SURFACE_FOOTWAY
        elif width <= 2.8:
            surface = SURFACE_CYCLEWAY
        else:
            continue
        result[start : start + 6, 6] = surface
    return result


def _road_furniture(road_vertices: np.ndarray) -> list[list[float]]:
    vertices: list[list[float]] = []
    occupied: set[tuple[int, int]] = set()
    for start in range(0, len(road_vertices) - 5, 6):
        quad = road_vertices[start : start + 6, :3]
        width = float(np.linalg.norm(quad[0] - quad[1]))
        if width > 3.0:
            continue
        a = (quad[0, [0, 2]] + quad[1, [0, 2]]) * 0.5
        b = (quad[2, [0, 2]] + quad[5, [0, 2]]) * 0.5
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 8.0:
            continue
        midpoint = (a + b) * 0.5
        if float(np.linalg.norm(midpoint)) > 1_450.0:
            continue
        key = (round(float(midpoint[0]) / 32), round(float(midpoint[1]) / 32))
        if key in occupied:
            continue
        occupied.add(key)
        direction = edge / length
        perpendicular = np.array([-direction[1], direction[0]])
        yaw = math.atan2(float(direction[1]), float(direction[0]))
        lamp_position = midpoint + perpendicular * (width * 0.5 + 0.65)
        vertices.extend(_lamp(lamp_position, yaw))
        hashed = (key[0] * 73856093) ^ (key[1] * 19349663)
        if hashed % 5 == 0:
            bench_position = midpoint - perpendicular * (width * 0.5 + 0.75)
            vertices.extend(_bench(bench_position, yaw))
    return vertices


def _shoreline_railing(scene: StaticScene) -> list[list[float]]:
    mask = scene.water_mask > 127
    bounds = scene.water_mask_bounds
    x_samples: list[tuple[float, float]] = []
    for pixel_x in range(0, mask.shape[1], 2):
        x = bounds[0] + pixel_x / (mask.shape[1] - 1) * (
            bounds[2] - bounds[0]
        )
        if not -700.0 <= x <= 550.0:
            continue
        column = mask[:, pixel_x]
        transitions = np.flatnonzero(column[:-1] & ~column[1:])
        candidates = [
            bounds[1]
            + (index + 0.5)
            / (mask.shape[0] - 1)
            * (bounds[3] - bounds[1])
            for index in transitions
        ]
        candidates = [z for z in candidates if -800.0 <= z <= 800.0]
        if candidates:
            x_samples.append((float(x), float(max(candidates))))
    if len(x_samples) < 5:
        return []
    points = np.asarray(x_samples)
    smoothed = points[:, 1].copy()
    for index in range(2, len(points) - 2):
        smoothed[index] = float(np.median(points[index - 2 : index + 3, 1]))
    points[:, 1] = smoothed
    vertices: list[list[float]] = []
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        if float(np.linalg.norm(end - start)) > 28.0:
            continue
        if index % 2 == 0:
            vertices.extend(_cylinder(start, 0.045, 1.15, SURFACE_METAL, 6))
        vertices.extend(_beam(start, end, 0.48, 0.055, SURFACE_METAL))
        vertices.extend(_beam(start, end, 1.02, 0.055, SURFACE_METAL))
    return vertices


def _facility_mesh(
    name: str, position: np.ndarray
) -> list[list[float]]:
    if "음수대" in name:
        return _cylinder(position, 0.22, 1.05, SURFACE_METAL, 8)
    if "화장실" in name:
        return _box(
            np.array([position[0], 1.45, position[1]]),
            (4.8, 2.9, 3.2),
            SURFACE_CONCRETE,
        )
    if "매점" in name:
        return _box(
            np.array([position[0], 1.8, position[1]]),
            (7.5, 3.6, 5.5),
            SURFACE_CONCRETE,
        )
    if "안내센터" in name or "119수난구조대" in name:
        return _box(
            np.array([position[0], 2.1, position[1]]),
            (11.0, 4.2, 8.0),
            SURFACE_CONCRETE,
        )
    if any(
        keyword in name
        for keyword in ("흡연부스", "자전거대여점", "샤워장", "물빛초소")
    ):
        return _box(
            np.array([position[0], 1.45, position[1]]),
            (4.8, 2.9, 3.5),
            SURFACE_METAL,
        )
    if "어린이놀이터" in name:
        vertices = _box(
            np.array([position[0], 0.7, position[1]]),
            (4.0, 1.4, 1.2),
            SURFACE_PLAYGROUND,
        )
        vertices.extend(
            _cylinder(position + np.array([-2.4, 0.0]), 0.10, 2.8, SURFACE_METAL)
        )
        return vertices
    if any(keyword in name for keyword in ("카페", "생태체험관", "철새조망대")):
        return _box(
            np.array([position[0], 2.0, position[1]]),
            (9.0, 4.0, 7.0),
            SURFACE_WOOD,
        )
    return []


def _covered_by_building(
    position: np.ndarray, roof_triangles: np.ndarray
) -> bool:
    if not len(roof_triangles):
        return False
    minimum = roof_triangles.min(axis=1)
    maximum = roof_triangles.max(axis=1)
    candidates = roof_triangles[
        (position[0] >= minimum[:, 0])
        & (position[0] <= maximum[:, 0])
        & (position[1] >= minimum[:, 1])
        & (position[1] <= maximum[:, 1])
    ]
    for triangle in candidates:
        a, b, c = triangle
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        signs = (
            cross(b - a, position - a),
            cross(c - b, position - b),
            cross(a - c, position - c),
        )
        if all(value >= -1e-6 for value in signs) or all(
            value <= 1e-6 for value in signs
        ):
            return True
    return False


def build_site_detail_mesh(
    scene: StaticScene,
    detail_osm: dict[str, Any],
    official_facilities: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    plane = LocalTangentPlane(
        scene.origin_latitude_deg, scene.origin_longitude_deg
    )
    vertices: list[list[float]] = []
    counts = {
        "historical_surfaces": 0,
        "inferred_trees": 0,
        "official_facilities": 0,
        "grass_blades": 0,
    }
    tree_limit = 900
    grass_blade_limit = 2_500
    roof_vertices = scene.building_vertices[
        (scene.building_vertices[:, 6] > 0.5)
        & (scene.building_vertices[:, 6] < 1.5),
        :,
    ]
    roof_triangles = roof_vertices[:, [0, 2]].reshape(-1, 3, 2)
    for element in detail_osm.get("elements", []):
        geometry = element.get("geometry", [])
        tags = element.get("tags", {})
        if len(geometry) < 4:
            continue
        polygon = np.asarray(
            [
                plane.to_local(node["lat"], node["lon"])[[0, 2]]
                for node in geometry
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(polygon[0] - polygon[-1]) < 0.05:
            polygon = polygon[:-1]
        leisure = tags.get("leisure")
        if leisure in {"pitch", "track"}:
            vertices.extend(_surface_mesh(polygon.copy(), 0.075, SURFACE_SPORT))
            counts["historical_surfaces"] += 1
        elif leisure == "playground":
            vertices.extend(
                _surface_mesh(polygon.copy(), 0.08, SURFACE_PLAYGROUND)
            )
            counts["historical_surfaces"] += 1
        elif leisure == "garden" or tags.get("natural") == "scrub":
            vertices.extend(_surface_mesh(polygon.copy(), 0.065, SURFACE_GARDEN))
            counts["historical_surfaces"] += 1
        is_grass = (
            tags.get("landuse") == "grass"
            or tags.get("natural") == "grassland"
        )
        if is_grass and counts["grass_blades"] < grass_blade_limit:
            minimum, maximum = polygon.min(axis=0), polygon.max(axis=0)
            seed = int(element.get("id", 0))
            spacing = 6.0
            offset = np.array(
                [(seed % 29) / 29.0, (seed % 31) / 31.0]
            ) * spacing
            for x in np.arange(minimum[0] + offset[0], maximum[0], spacing):
                for z in np.arange(minimum[1] + offset[1], maximum[1], spacing):
                    if counts["grass_blades"] >= grass_blade_limit:
                        break
                    point = np.array([x, z])
                    # A budget on where blades are authored, not a level of
                    # detail: whether authored geometry is drawn is decided at
                    # runtime by observation distance in simulator/vegetation.py.
                    if np.linalg.norm(point) > EVENT_SITE_DETAIL_RADIUS_M or (
                        not _inside_polygon(point, polygon)
                    ):
                        continue
                    hashed = (
                        seed * 73856093
                        ^ round(x * 10.0) * 19349663
                        ^ round(z * 10.0) * 83492791
                    ) & 0x7FFFFFFF
                    jitter = np.array(
                        [
                            ((hashed & 255) / 255.0 - 0.5) * spacing * 0.72,
                            (((hashed >> 8) & 255) / 255.0 - 0.5)
                            * spacing
                            * 0.72,
                        ]
                    )
                    blade_position = point + jitter
                    if not _inside_polygon(blade_position, polygon):
                        continue
                    height = 0.26 + ((hashed >> 16) & 255) / 255.0 * 0.24
                    width = 0.028 + ((hashed >> 12) & 15) / 15.0 * 0.022
                    yaw = (hashed % 6283) / 1000.0
                    vertices.extend(
                        _grass_blade(blade_position, height, width, yaw)
                    )
                    counts["grass_blades"] += 1
        if tags.get("natural") != "wood" or counts["inferred_trees"] >= tree_limit:
            continue
        minimum, maximum = polygon.min(axis=0), polygon.max(axis=0)
        seed = int(element.get("id", 0))
        spacing = 25.0
        offset = np.array([(seed % 17) / 17.0, (seed % 23) / 23.0]) * spacing
        for x in np.arange(minimum[0] + offset[0], maximum[0], spacing):
            for z in np.arange(minimum[1] + offset[1], maximum[1], spacing):
                point = np.array([x, z])
                if (
                    counts["inferred_trees"] >= tree_limit
                    or np.linalg.norm(point) > 1_900.0
                    or not _inside_polygon(point, polygon)
                ):
                    continue
                hashed = (
                    seed * 73856093
                    ^ round(x) * 19349663
                    ^ round(z) * 83492791
                ) & 0x7FFFFFFF
                height = 6.0 + (hashed % 500) / 100.0
                radius = 2.1 + ((hashed // 11) % 160) / 100.0
                vertices.extend(_tree(point, height, radius))
                counts["inferred_trees"] += 1
    for facility in official_facilities.get("facilities", []):
        local = plane.to_local(
            float(facility["latitude"]), float(facility["longitude"])
        )[[0, 2]]
        if _covered_by_building(local, roof_triangles):
            continue
        mesh = _facility_mesh(str(facility["name"]), local)
        if mesh:
            vertices.extend(mesh)
            counts["official_facilities"] += 1
    furniture = _road_furniture(scene.road_vertices)
    railing = _shoreline_railing(scene)
    vertices.extend(furniture)
    vertices.extend(railing)
    counts["furniture_vertices"] = len(furniture)
    counts["railing_vertices"] = len(railing)
    return np.asarray(vertices, dtype=np.float32), counts
