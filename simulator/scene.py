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

FACADE_GENERIC = 0.0
FACADE_GLASS_BLUE = 1.0
FACADE_GOLD_63 = 2.0
FACADE_RESIDENTIAL = 3.0
FACADE_INSTITUTIONAL = 4.0
FACADE_PARC1 = 5.0
FACADE_HOTEL = 6.0


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


def _facade_style(tags: dict[str, str]) -> float:
    names = " ".join(
        tags.get(key, "") for key in ("name", "name:en", "official_name")
    ).lower()
    building = tags.get("building", "").lower()
    material = tags.get("building:material", "").lower()
    colour = tags.get("building:colour", "").lower()
    if "63시티" in names or "63 city" in names or "63square" in names:
        return FACADE_GOLD_63
    if (
        "parc.1" in names
        or "파크원" in names
        or "nh financial tower" in names
        or "nh금융타워" in names
    ):
        return FACADE_PARC1
    if (
        "ifc" in names
        or "국제금융" in names
        or "전경련회관" in names
        or "fki tow" in names
    ):
        return FACADE_GLASS_BLUE
    if "conrad" in names or building == "hotel":
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
                    _vertex(
                        (point[0], elevation_m, point[1]),
                        (0.0, 1.0, 0.0),
                        material,
                        (point[0], point[1]),
                    )
                )
    return vertices


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
        "path": 1.5,
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
            height = _height(tags)
            minimum_height = _minimum_height(tags)
            style = _facade_style(tags)
            if tags.get("roof:shape") == "dome":
                buildings.extend(
                    _dome_mesh(local, height, minimum_height, style)
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
                _bridge_mesh(local, _road_width(tags), 0.06, 3.0)
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
