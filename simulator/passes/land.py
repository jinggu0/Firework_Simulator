"""Terrain-displaced land surface, clipped against the river mask."""

from __future__ import annotations

from dataclasses import replace

import moderngl
import numpy as np

from .. import shaders
from ..config import LightingConfig
from ..lighting import led_energy_budget
from ..water import WaterConfig, build_water_mesh

WATER_MASK_UNIT = 1
TERRAIN_UNIT = 2

STREET_LAMP_COLOR = np.array([1.0, 0.63, 0.31], dtype=np.float32)
"""Warm high-pressure-sodium-like appearance of the imported lamp heads."""


class LandPass:
    """The 5 x 4 km ground plane sharing the water grid's tessellation."""

    def __init__(
        self,
        ctx: moderngl.Context,
        lighting_config: LightingConfig,
        water_config: WaterConfig,
        water_mask_bounds: np.ndarray,
        terrain_bounds: np.ndarray,
    ) -> None:
        self.ctx = ctx
        self.program = shaders.program(ctx, "land.vert", "land.frag")
        street_lamp_budget = led_energy_budget(
            replace(
                lighting_config,
                led_input_power_w=lighting_config.street_lamp_input_power_w,
            )
        )
        self.static_light_power_w = street_lamp_budget.luminaire_radiant_w
        self.static_light_color = STREET_LAMP_COLOR

        vertices, indices = build_water_mesh(
            water_config,
            water_config.far_grid_size,
            water_config.far_extent_m,
        )
        self.vertex_buffer = ctx.buffer(vertices.tobytes())
        self.index_buffer = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vertex_buffer, "2f", "in_xz")],
            self.index_buffer,
            index_element_size=4,
        )
        self.program["water_mask"] = WATER_MASK_UNIT
        self.program["water_mask_bounds"].value = tuple(water_mask_bounds)
        self.program["terrain_height"] = TERRAIN_UNIT
        self.program["terrain_bounds"].value = tuple(terrain_bounds)

    def set_view_projection(self, matrix_bytes: bytes) -> None:
        self.program["view_projection"].write(matrix_bytes)

    def set_ambient_scale(self, scale: float) -> None:
        self.program["sky_ambient_scale"] = scale

    def draw(self) -> None:
        self.vao.render(moderngl.TRIANGLES)


def initialise_static_lights(
    programs, color: np.ndarray, power_w: float
) -> None:
    """Zero the luminaire arrays on every program that reads them."""

    for program in programs:
        program["static_light_count"] = 0
        program["static_light_position"].write(
            np.zeros((4, 3), dtype=np.float32).tobytes()
        )
        program["static_light_color"].value = tuple(color)
        program["static_light_power_w"] = power_w
