"""Static city geometry: buildings, bridges, roads, vegetation, site detail."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import moderngl
import numpy as np

from .. import shaders
from ..config import LightingConfig, PhysicalCameraConfig, RenderConfig
from ..lighting import led_energy_budget
from ..material_textures import (
    SCANNED_MATERIAL_UNIT,
    ScannedMaterialTextures,
    texture_widths_m,
)
from ..materials import MATERIAL_LIBRARY, MaterialLibrary
from ..road_semantics import (
    RoadSemanticFilterStats,
    filter_occluded_road_segments,
    load_road_structure_semantics,
)
from ..scene import LINEAR_STYLE_STEPS, SURFACE_GRASS_BLADE, load_scene
from ..site_details import GRASS_VERTICES_PER_TUFT
from ..terrain import sample_heightmap_array
from ..terrain_detail import (
    AdaptiveTessellationStats,
    adaptive_terrain_tessellate,
    load_priority_area_bounds,
)
from ..vegetation import VegetationLod

VERTEX_LAYOUT = (
    "3f 3f 1f 2f 1f",
    "in_position",
    "in_normal",
    "in_surface",
    "in_surface_uv",
    "in_facade_style",
)

SURFACE_LAMP = 10.0
SURFACE_CONCRETE = 12.0
LAMP_VERTICES_PER_FIXTURE = 36
GRASS_CHUNK_SIZE_M = 64.0


def linear_feature_uv(vertices: np.ndarray) -> np.ndarray:
    """Map six-vertex road/bridge segments to longitudinal and edge UVs.

    The historical asset stores one independent six-vertex quad per linear
    segment. Its old UV channel is world X/Z, which cannot describe deck/road
    edges, longitudinal joints, or lane markings. This conversion is performed
    once while uploading the static mesh: ``u`` is accumulated distance along
    each connected way and ``v`` is -1/+1 at its two edges.

    The source geometry is not changed and no fixture position is presented as
    surveyed.  The resulting spacing is an appearance calibration against the
    2024 event photographs, while every lit segment still comes from the
    2024-10-05 OSM bridge geometry.
    """

    converted = np.array(vertices, dtype=np.float32, copy=True)
    if len(converted) % 6:
        return converted
    distance_along = 0.0
    previous_end: np.ndarray | None = None
    for offset in range(0, len(converted), 6):
        quad = converted[offset : offset + 6]
        start = 0.5 * (quad[0, [0, 2]] + quad[1, [0, 2]])
        end = 0.5 * (quad[2, [0, 2]] + quad[5, [0, 2]])
        if previous_end is None or np.linalg.norm(start - previous_end) > 0.5:
            distance_along = 0.0
        segment_length = float(np.linalg.norm(end - start))
        quad[:, 7] = distance_along + np.array(
            [0.0, 0.0, segment_length, 0.0, segment_length, segment_length],
            dtype=np.float32,
        )
        quad[:, 8] = np.array(
            [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=np.float32
        )
        distance_along += segment_length
        previous_end = end
    return converted


def bridge_lighting_uv(vertices: np.ndarray) -> np.ndarray:
    """Compatibility name for the longitudinal bridge coordinate transform."""

    return linear_feature_uv(vertices)


def road_edge_detail_vertices(road_vertices: np.ndarray) -> np.ndarray:
    """Give mapped asphalt edges a raised concrete kerb and visible reveal.

    OSM centreline width is authoritative for the plan footprint but carries
    no vertical kerb profile.  A bounded 140 mm reveal and 180 mm top strip are
    derived only for mapped asphalt segments inside the central 1.7 km.  Their
    dimensions are generic urban construction values, not a Yeouido survey.
    """

    if not len(road_vertices) or len(road_vertices) % 6:
        return np.empty((0, 10), dtype=np.float32)
    output: list[list[float]] = []
    up = np.array([0.0, 0.14, 0.0], dtype=np.float64)
    kerb_width_m = 0.18
    for offset in range(0, len(road_vertices), 6):
        quad = road_vertices[offset : offset + 6]
        if not np.isclose(quad[0, 6], 3.0):
            continue
        left_start = quad[0, :3].astype(np.float64)
        right_start = quad[1, :3].astype(np.float64)
        right_end = quad[2, :3].astype(np.float64)
        left_end = quad[5, :3].astype(np.float64)
        width_m = float(np.linalg.norm(right_start - left_start))
        centre_start = 0.5 * (left_start + right_start)
        centre_end = 0.5 * (left_end + right_end)
        length_m = float(np.linalg.norm(centre_end - centre_start))
        if not 3.5 <= width_m <= 32.0 or length_m < 0.35:
            continue
        midpoint_xz = 0.5 * (centre_start + centre_end)[[0, 2]]
        if float(np.linalg.norm(midpoint_xz)) > 1_700.0:
            continue
        for edge_start, edge_end in (
            (left_start, left_end),
            (right_start, right_end),
        ):
            outward = edge_start - centre_start
            outward[1] = 0.0
            outward /= max(float(np.linalg.norm(outward)), 1e-6)
            outer_start = edge_start + outward * kerb_width_m
            outer_end = edge_end + outward * kerb_width_m
            # The road-facing vertical reveal catches grazing lamp light.
            _append_quad(
                output,
                (edge_start, edge_end, edge_end + up, edge_start + up),
                SURFACE_CONCRETE,
            )
            top = (
                edge_start + up,
                edge_end + up,
                outer_end + up,
                outer_start + up,
            )
            if np.cross(top[1] - top[0], top[2] - top[0])[1] < 0.0:
                top = (top[0], top[3], top[2], top[1])
            _append_quad(output, top, SURFACE_CONCRETE)
    return np.asarray(output, dtype=np.float32).reshape(-1, 10)


def grass_detail_chunks(
    detail_vertices: np.ndarray,
    chunk_size_m: float = GRASS_CHUNK_SIZE_M,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Partition complete grass tufts into independently cullable cells."""

    grass = detail_vertices[
        np.isclose(detail_vertices[:, 6], SURFACE_GRASS_BLADE)
    ]
    if not len(grass):
        return []
    if len(grass) % GRASS_VERTICES_PER_TUFT:
        raise ValueError("grass detail ends with an incomplete tuft")
    tufts = grass.reshape(-1, GRASS_VERTICES_PER_TUFT, 10)
    centres = tufts[:, :, [0, 2]].mean(axis=1)
    cells = np.floor(centres / chunk_size_m).astype(np.int32)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault((int(cell[0]), int(cell[1])), []).append(index)

    chunks: list[tuple[np.ndarray, np.ndarray, float]] = []
    for key in sorted(groups):
        indices = np.asarray(groups[key], dtype=np.int32)
        vertices = np.ascontiguousarray(
            tufts[indices].reshape(-1, 10), dtype=np.float32
        )
        minimum = vertices[:, [0, 2]].min(axis=0)
        maximum = vertices[:, [0, 2]].max(axis=0)
        centre = (minimum + maximum) * 0.5
        radius = float(np.linalg.norm(maximum - minimum) * 0.5)
        chunks.append((vertices, centre.astype(np.float32), radius))
    return chunks


def _mesh_vertex(
    position: np.ndarray, normal: np.ndarray, surface: float,
    uv: tuple[float, float] = (0.0, 0.0),
    linear_style: float = 0.0,
) -> list[float]:
    return [
        float(position[0]), float(position[1]), float(position[2]),
        float(normal[0]), float(normal[1]), float(normal[2]),
        surface, uv[0], uv[1], linear_style,
    ]


def _append_quad(
    output: list[list[float]], points: tuple[np.ndarray, ...], surface: float,
    uv: tuple[tuple[float, float], ...] | None = None,
    normal: np.ndarray | None = None,
    linear_style: float = 0.0,
) -> None:
    a, b, c, d = points
    resolved_normal = (
        np.asarray(normal, dtype=np.float64)
        if normal is not None
        else np.cross(b - a, c - a)
    )
    length = float(np.linalg.norm(resolved_normal))
    if length < 1e-6:
        return
    resolved_normal /= length
    coordinates = uv or ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for index in (0, 1, 2, 0, 2, 3):
        output.append(
            _mesh_vertex(
                points[index],
                resolved_normal,
                surface,
                coordinates[index],
                linear_style,
            )
        )


def stair_structure_vertices(
    step_vertices: np.ndarray,
    terrain_height_m: np.ndarray,
    terrain_bounds: np.ndarray,
) -> np.ndarray:
    """Terrace mapped stair ways only where the official terrain resolves rise.

    The dated OSM line fixes plan position and width while the official height
    field fixes both endpoint elevations. Tread count targets the documented
    ``2 * riser + tread = 0.63 m`` relation and is bounded by the Korean
    accessibility dimensions (tread >= 0.28 m, riser <= 0.18 m). A rise below
    0.12 m is beneath the terrain evidence used here, so that segment retains
    its original draped deck rather than receiving invented vertical geometry.
    """

    vertices = np.asarray(step_vertices, dtype=np.float32)
    if not len(vertices) or len(vertices) % 6:
        return np.empty((0, 10), dtype=np.float32)
    output: list[list[float]] = []
    minimum_resolved_rise_m = 0.12
    minimum_tread_m = 0.28
    maximum_riser_m = 0.18
    preferred_stride_m = 0.63

    def terrain_at(points: np.ndarray) -> np.ndarray:
        return sample_heightmap_array(
            terrain_height_m, terrain_bounds, points[:, [0, 2]]
        )

    for offset in range(0, len(vertices), 6):
        quad = vertices[offset : offset + 6].astype(np.float64)
        left_start, right_start = quad[0, :3], quad[1, :3]
        right_end, left_end = quad[2, :3], quad[5, :3]
        centre_start = 0.5 * (left_start + right_start)
        centre_end = 0.5 * (left_end + right_end)
        plan_edge = centre_end[[0, 2]] - centre_start[[0, 2]]
        length_m = float(np.linalg.norm(plan_edge))
        centre_terrain = sample_heightmap_array(
            terrain_height_m,
            terrain_bounds,
            np.vstack((centre_start[[0, 2]], centre_end[[0, 2]])),
        )
        start_world_y = float(centre_terrain[0] + centre_start[1])
        end_world_y = float(centre_terrain[1] + centre_end[1])
        signed_rise_m = end_world_y - start_world_y
        rise_m = abs(signed_rise_m)
        minimum_steps = max(1, int(math.ceil(rise_m / maximum_riser_m)))
        maximum_steps = int(math.floor(length_m / minimum_tread_m))
        if (
            length_m < minimum_tread_m
            or rise_m < minimum_resolved_rise_m
            or minimum_steps > maximum_steps
        ):
            fallback = quad.copy()
            fallback[:, 9] = 0.0
            output.extend(fallback.tolist())
            continue

        preferred_steps = max(
            1, int(round((length_m + 2.0 * rise_m) / preferred_stride_m))
        )
        step_count = min(max(preferred_steps, minimum_steps), maximum_steps)
        if signed_rise_m >= 0.0:
            low_left, low_right = left_start, right_start
            high_left, high_right = left_end, right_end
            low_world_y = start_world_y
            u_origin = float(quad[0, 7])
        else:
            low_left, low_right = right_end, left_end
            high_left, high_right = right_start, left_start
            low_world_y = end_world_y
            u_origin = float(quad[5, 7])
        direction = np.array(
            [plan_edge[0], 0.0, plan_edge[1]], dtype=np.float64
        )
        if signed_rise_m < 0.0:
            direction *= -1.0
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        riser_normal = -direction
        surface = float(quad[0, 6])
        style = LINEAR_STYLE_STEPS

        for index in range(step_count):
            fraction0 = index / step_count
            fraction1 = (index + 1) / step_count
            left0 = low_left * (1.0 - fraction0) + high_left * fraction0
            right0 = low_right * (1.0 - fraction0) + high_right * fraction0
            left1 = low_left * (1.0 - fraction1) + high_left * fraction1
            right1 = low_right * (1.0 - fraction1) + high_right * fraction1
            bottom_world_y = low_world_y + rise_m * fraction0
            top_world_y = low_world_y + rise_m * fraction1

            tread = np.vstack((left0, right0, right1, left1))
            tread[:, 1] = top_world_y - terrain_at(tread)
            u0 = u_origin + length_m * fraction0
            u1 = u_origin + length_m * fraction1
            _append_quad(
                output,
                tuple(tread),
                surface,
                ((u0, -1.0), (u0, 1.0), (u1, 1.0), (u1, -1.0)),
                normal=np.array([0.0, 1.0, 0.0]),
                linear_style=style,
            )

            riser = np.vstack((left0, right0, right0, left0))
            riser[:2, 1] = bottom_world_y - terrain_at(riser[:2])
            riser[2:, 1] = top_world_y - terrain_at(riser[2:])
            _append_quad(
                output,
                tuple(riser),
                surface,
                ((-1.0, bottom_world_y), (1.0, bottom_world_y),
                 (1.0, top_world_y), (-1.0, top_world_y)),
                normal=riser_normal,
                linear_style=style,
            )
    return np.asarray(output, dtype=np.float32).reshape(-1, 10)


def bridge_structure_vertices(deck_vertices: np.ndarray) -> np.ndarray:
    """Derive fascia, underside and sparse piers from the mapped deck mesh.

    OSM gives bridge centre lines and widths but the stored scene previously
    rendered them as infinitely thin sheets. The derived parts make the same
    surveyed footprint read as a structure from shore and in reflection. Pier
    spacing and fascia depth are grade-D appearance calibrations, not a survey.
    """

    if not len(deck_vertices) or len(deck_vertices) % 6:
        return np.empty((0, 10), dtype=np.float32)
    output: list[list[float]] = []
    pier_spacing_m = 85.0
    fascia_depth_m = 1.35
    for offset in range(0, len(deck_vertices), 6):
        quad = deck_vertices[offset : offset + 6]
        left_start = quad[0, :3].astype(np.float64)
        right_start = quad[1, :3].astype(np.float64)
        right_end = quad[2, :3].astype(np.float64)
        left_end = quad[5, :3].astype(np.float64)
        down = np.array([0.0, -fascia_depth_m, 0.0])
        bottom_left_start = left_start + down
        bottom_right_start = right_start + down
        bottom_right_end = right_end + down
        bottom_left_end = left_end + down
        u0, u1 = float(quad[0, 7]), float(quad[2, 7])
        _append_quad(
            output,
            (left_start, left_end, bottom_left_end, bottom_left_start),
            2.0,
            ((u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)),
        )
        _append_quad(
            output,
            (right_end, right_start, bottom_right_start, bottom_right_end),
            2.0,
            ((u1, 0.0), (u0, 0.0), (u0, 1.0), (u1, 1.0)),
        )
        _append_quad(
            output,
            (
                bottom_left_start, bottom_left_end,
                bottom_right_end, bottom_right_start,
            ),
            2.0,
        )

        width_m = float(np.linalg.norm(right_start - left_start))
        segment_length_m = max(u1 - u0, 0.0)
        if width_m < 8.0 or segment_length_m < 0.1:
            continue
        first_pier = (np.floor(u0 / pier_spacing_m) + 1.0) * pier_spacing_m
        for along in np.arange(first_pier, u1 + 1e-4, pier_spacing_m):
            alpha = float((along - u0) / segment_length_m)
            centre_start = 0.5 * (left_start + right_start)
            centre_end = 0.5 * (left_end + right_end)
            centre = centre_start * (1.0 - alpha) + centre_end * alpha
            across = right_start - left_start
            across /= max(float(np.linalg.norm(across)), 1e-6)
            forward = centre_end - centre_start
            forward /= max(float(np.linalg.norm(forward)), 1e-6)
            across *= min(width_m * 0.16, 2.2)
            forward *= 0.9
            low_y = 0.15
            high_y = centre[1] - fascia_depth_m
            if high_y <= low_y:
                continue
            low = [
                centre + sx * across + sz * forward
                for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            high = [point.copy() for point in low]
            for point in low:
                point[1] = low_y
            for point in high:
                point[1] = high_y
            for side in range(4):
                following = (side + 1) % 4
                _append_quad(
                    output,
                    (low[side], low[following], high[following], high[side]),
                    SURFACE_CONCRETE,
                )
    return np.asarray(output, dtype=np.float32).reshape(-1, 10)


def _append_box(
    output: list[list[float]], centre: np.ndarray,
    dimensions: tuple[float, float, float], surface: float,
) -> None:
    hx, hy, hz = (value * 0.5 for value in dimensions)
    corners = [
        centre + np.array([sx * hx, sy * hy, sz * hz])
        for sx, sy, sz in (
            (-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1),
            (-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1),
        )
    ]
    for indices in (
        (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
        (3, 7, 4, 0), (4, 7, 6, 5),
    ):
        _append_quad(output, tuple(corners[index] for index in indices), surface)


def rooftop_detail_vertices(building_vertices: np.ndarray) -> np.ndarray:
    """Add bounded mechanical penthouses to large OSM roof triangles.

    The footprint and roof height remain the historical OSM geometry. Public
    data does not locate individual HVAC units, so this deliberately sparse
    derived layer is grade D and exists to break the implausibly perfect flat
    silhouette, not to claim surveyed equipment placement.
    """

    roof = building_vertices[np.isclose(building_vertices[:, 6], 1.0)]
    if not len(roof) or len(roof) % 3:
        return np.empty((0, 10), dtype=np.float32)
    output: list[list[float]] = []
    for triangle in roof.reshape(-1, 3, 10):
        # Published landmark silhouettes are reconstructed in scene.py.
        # Do not place confidence-grade-D HVAC boxes on top of them.
        if float(triangle[0, 9]) in {2.0, 5.0, 7.0, 8.0, 12.0}:
            continue
        positions = triangle[:, :3].astype(np.float64)
        area_m2 = 0.5 * float(np.linalg.norm(np.cross(
            positions[1] - positions[0], positions[2] - positions[0]
        )))
        if area_m2 < 500.0:
            continue
        centre = positions.mean(axis=0)
        random = np.sin(centre[0] * 12.9898 + centre[2] * 78.233) * 43758.5453
        random -= np.floor(random)
        footprint = min(max(np.sqrt(area_m2) * .16, 3.0), 7.5)
        height = 1.1 + random * 1.9
        centre[1] += height * 0.5 + 0.05
        _append_box(
            output,
            centre,
            (footprint, height, footprint * (.58 + random * .24)),
            SURFACE_CONCRETE,
        )
    return np.asarray(output, dtype=np.float32).reshape(-1, 10)


@dataclass(frozen=True, slots=True)
class SceneData:
    """CPU-side scene arrays shared with the water and land passes."""

    water_mask: np.ndarray
    water_mask_bounds: np.ndarray
    terrain_height_m: np.ndarray
    terrain_bounds: np.ndarray
    static_light_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )

    @classmethod
    def empty(cls) -> "SceneData":
        bounds = np.array(
            [-10_000.0, -10_000.0, 10_000.0, 10_000.0], dtype=np.float32
        )
        return cls(
            water_mask=np.full((1, 1), 255, dtype=np.uint8),
            water_mask_bounds=bounds,
            terrain_height_m=np.zeros((1, 1), dtype=np.float32),
            terrain_bounds=bounds.copy(),
        )


class ScenePass:
    """Batched static geometry, drawn in one pass with a facade-family shader.

    Buildings keep individual meshes in the same depth buffer as water, smoke,
    and fireworks; the facade family travels as a vertex attribute so the batch
    does not become one draw call per building.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        lighting_config: LightingConfig,
        scene_path: Path,
        render_config: RenderConfig | None = None,
        camera_config: PhysicalCameraConfig | None = None,
        materials: MaterialLibrary = MATERIAL_LIBRARY,
    ) -> None:
        self.ctx = ctx
        self.materials = materials
        self.lod = VegetationLod.from_camera(
            camera_config or PhysicalCameraConfig(),
            render_config or RenderConfig(),
        )
        self.program = shaders.program(ctx, "scene.vert", "scene.frag")
        budget = led_energy_budget(lighting_config)
        self.program["window_radiance_w_m2_sr"] = budget.window_radiance_w_m2_sr
        # Surface appearance is a uniform table rather than a shader branch
        # chain, so adding a material is a row in simulator/materials.py.
        materials.upload(self.program)
        self.scanned_materials = ScannedMaterialTextures(ctx)
        self.program["scanned_material_texture"] = SCANNED_MATERIAL_UNIT
        self.program["scanned_diffuse_mean"].write(
            self.scanned_materials.diffuse_means.tobytes()
        )
        self.program["scanned_texture_width_m"].write(
            texture_widths_m().tobytes()
        )
        # Vegetation detail bands follow from the camera optics, so a change of
        # sensor, focal length, or resolution moves them automatically.
        self.program["blade_full_detail_m"] = self.lod.blade_full_detail_m
        self.program["blade_cutoff_m"] = self.lod.blade_cutoff_m
        self.program["tree_sway_cutoff_m"] = self.lod.tree_sway_cutoff_m
        self.vaos: list[tuple[moderngl.VertexArray, int]] = []
        self.grass_vaos: list[
            tuple[moderngl.VertexArray, int, np.ndarray, float]
        ] = []
        self.reflection_vaos: list[tuple[moderngl.VertexArray, int]] = []
        self.buffers: list[moderngl.Buffer] = []
        self.camera_position_xz = np.zeros(2, dtype=np.float32)
        self.data = SceneData.empty()
        self.road_tessellation_stats: AdaptiveTessellationStats | None = None
        self.road_edge_tessellation_stats: AdaptiveTessellationStats | None = None
        self.road_semantic_filter_stats: RoadSemanticFilterStats | None = None
        if scene_path.exists():
            self._build(load_scene(scene_path))

    def _build(self, scene) -> None:
        static_light_positions = np.empty((0, 3), dtype=np.float32)
        lamp_vertices = scene.detail_vertices[
            np.isclose(scene.detail_vertices[:, 6], SURFACE_LAMP)
        ]
        complete = len(lamp_vertices) // LAMP_VERTICES_PER_FIXTURE
        if complete:
            # Each lamp head is a fixed-size box; its centroid is the luminaire
            # position the lighting model needs.
            static_light_positions = (
                lamp_vertices[: complete * LAMP_VERTICES_PER_FIXTURE, :3]
                .reshape(-1, LAMP_VERTICES_PER_FIXTURE, 3)
                .mean(axis=1)
                .astype(np.float32)
            )
        self.data = SceneData(
            water_mask=scene.water_mask,
            water_mask_bounds=scene.water_mask_bounds,
            terrain_height_m=scene.terrain_height_m,
            terrain_bounds=scene.terrain_bounds,
            static_light_positions=static_light_positions,
        )
        rooftop_detail = rooftop_detail_vertices(scene.building_vertices)
        building_batch = (
            np.concatenate((scene.building_vertices, rooftop_detail), axis=0)
            if len(rooftop_detail) else scene.building_vertices
        )
        bridge_deck = bridge_lighting_uv(scene.bridge_vertices)
        bridge_structure = bridge_structure_vertices(bridge_deck)
        bridge_batch = (
            np.concatenate((bridge_deck, bridge_structure), axis=0)
            if len(bridge_structure) else bridge_deck
        )
        visible_roads, self.road_semantic_filter_stats = (
            filter_occluded_road_segments(
                scene.road_vertices,
                load_road_structure_semantics(),
            )
        )
        road_deck = linear_feature_uv(visible_roads)
        stair_mask = np.isclose(road_deck[:, 9], LINEAR_STYLE_STEPS)
        ordinary_road = road_deck[~stair_mask]
        stair_geometry = stair_structure_vertices(
            road_deck[stair_mask],
            scene.terrain_height_m,
            scene.terrain_bounds,
        )
        road_edges = road_edge_detail_vertices(ordinary_road)
        priority_bounds = load_priority_area_bounds()
        terrain_road, self.road_tessellation_stats = adaptive_terrain_tessellate(
            ordinary_road,
            scene.terrain_height_m,
            scene.terrain_bounds,
            scene.water_mask,
            scene.water_mask_bounds,
            priority_bounds,
        )
        terrain_edges, self.road_edge_tessellation_stats = (
            adaptive_terrain_tessellate(
                road_edges,
                scene.terrain_height_m,
                scene.terrain_bounds,
                scene.water_mask,
                scene.water_mask_bounds,
                priority_bounds,
            )
        )
        road_parts = [terrain_road]
        if len(stair_geometry):
            road_parts.append(stair_geometry)
        if len(terrain_edges):
            road_parts.append(terrain_edges)
        road_batch = np.concatenate(road_parts, axis=0)
        non_grass_detail = scene.detail_vertices[
            ~np.isclose(scene.detail_vertices[:, 6], SURFACE_GRASS_BLADE)
        ]
        batches = (
            building_batch,
            bridge_batch,
            road_batch,
            scene.vegetation_vertices,
            non_grass_detail,
            scene.structure_vertices,
        )
        for index, vertices in enumerate(batches):
            if not len(vertices):
                continue
            buffer = self.ctx.buffer(vertices.tobytes())
            self.buffers.append(buffer)
            vao = self.ctx.vertex_array(
                self.program, [(buffer, *VERTEX_LAYOUT)]
            )
            self.vaos.append((vao, len(vertices)))
            if index < 2:
                # Buildings and bridges dominate the reflected image.
                self.reflection_vaos.append((vao, len(vertices)))
            elif index == 4:
                # Of the site detail only lamp heads and concrete facilities
                # are visually significant at reflection distance; dark tree
                # crowns and benches reuse the surrounding land reflection.
                mask = np.isin(vertices[:, 6], (SURFACE_LAMP, SURFACE_CONCRETE))
                reflection_vertices = vertices[mask]
                if len(reflection_vertices):
                    reflection_buffer = self.ctx.buffer(
                        reflection_vertices.tobytes()
                    )
                    self.buffers.append(reflection_buffer)
                    self.reflection_vaos.append(
                        (
                            self.ctx.vertex_array(
                                self.program,
                                [(reflection_buffer, *VERTEX_LAYOUT)],
                            ),
                            len(reflection_vertices),
                        )
                    )
            elif index == 5:
                # Survey-qualified riverside walls and slopes are part of the
                # reflected shoreline silhouette. They remain one batch and
                # one reflection draw regardless of source feature count.
                self.reflection_vaos.append((vao, len(vertices)))
        for vertices, centre, radius in grass_detail_chunks(
            scene.detail_vertices
        ):
            buffer = self.ctx.buffer(vertices.tobytes())
            self.buffers.append(buffer)
            vao = self.ctx.vertex_array(
                self.program, [(buffer, *VERTEX_LAYOUT)]
            )
            self.grass_vaos.append((vao, len(vertices), centre, radius))

    # -- per-frame state ---------------------------------------------------

    def set_terrain(self, terrain_unit: int) -> None:
        self.program["terrain_height"] = terrain_unit
        self.program["terrain_bounds"].value = tuple(self.data.terrain_bounds)

    def set_view_projection(
        self, matrix_bytes: bytes, camera_position
    ) -> None:
        self.program["view_projection"].write(matrix_bytes)
        self.program["camera_position"].value = tuple(camera_position)
        self.camera_position_xz[:] = camera_position[[0, 2]]

    def set_environment(self, time_s: float, wind_xz, wind_speed_mps: float) -> None:
        self.program["time_s"] = time_s
        self.program["wind_xz"].value = tuple(wind_xz)
        self.program["wind_speed_mps"] = wind_speed_mps

    def set_ambient_irradiance(self, irradiance_w_m2: float) -> None:
        self.program["ambient_irradiance_w_m2"] = irradiance_w_m2

    # -- drawing -----------------------------------------------------------

    def draw(self) -> None:
        self.scanned_materials.bind()
        for vao, vertex_count in self.vaos:
            vao.render(moderngl.TRIANGLES, vertices=vertex_count)
        cutoff_m = self.lod.blade_cutoff_m
        for vao, vertex_count, centre, radius in self.grass_vaos:
            distance_m = float(
                np.linalg.norm(centre - self.camera_position_xz)
            )
            if distance_m <= cutoff_m + radius:
                vao.render(moderngl.TRIANGLES, vertices=vertex_count)

    def draw_reflection(self) -> None:
        self.scanned_materials.bind()
        for vao, vertex_count in self.reflection_vaos:
            vao.render(moderngl.TRIANGLES, vertices=vertex_count)
