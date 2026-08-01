"""Static city geometry: buildings, bridges, roads, vegetation, site detail."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import moderngl
import numpy as np

from .. import shaders
from ..config import LightingConfig, PhysicalCameraConfig, RenderConfig
from ..lighting import led_energy_budget
from ..materials import MATERIAL_LIBRARY, MaterialLibrary
from ..scene import load_scene
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
        # Vegetation detail bands follow from the camera optics, so a change of
        # sensor, focal length, or resolution moves them automatically.
        self.program["blade_full_detail_m"] = self.lod.blade_full_detail_m
        self.program["blade_cutoff_m"] = self.lod.blade_cutoff_m
        self.program["tree_sway_cutoff_m"] = self.lod.tree_sway_cutoff_m
        self.vaos: list[tuple[moderngl.VertexArray, int]] = []
        self.reflection_vaos: list[tuple[moderngl.VertexArray, int]] = []
        self.buffers: list[moderngl.Buffer] = []
        self.data = SceneData.empty()
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
        batches = (
            scene.building_vertices,
            scene.bridge_vertices,
            scene.road_vertices,
            scene.vegetation_vertices,
            scene.detail_vertices,
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

    # -- per-frame state ---------------------------------------------------

    def set_terrain(self, terrain_unit: int) -> None:
        self.program["terrain_height"] = terrain_unit
        self.program["terrain_bounds"].value = tuple(self.data.terrain_bounds)

    def set_view_projection(
        self, matrix_bytes: bytes, camera_position
    ) -> None:
        self.program["view_projection"].write(matrix_bytes)
        self.program["camera_position"].value = tuple(camera_position)

    def set_environment(self, time_s: float, wind_xz, wind_speed_mps: float) -> None:
        self.program["time_s"] = time_s
        self.program["wind_xz"].value = tuple(wind_xz)
        self.program["wind_speed_mps"] = wind_speed_mps

    def set_ambient_irradiance(self, irradiance_w_m2: float) -> None:
        self.program["ambient_irradiance_w_m2"] = irradiance_w_m2

    # -- drawing -----------------------------------------------------------

    def draw(self) -> None:
        for vao, vertex_count in self.vaos:
            vao.render(moderngl.TRIANGLES, vertices=vertex_count)

    def draw_reflection(self) -> None:
        for vao, vertex_count in self.reflection_vaos:
            vao.render(moderngl.TRIANGLES, vertices=vertex_count)
