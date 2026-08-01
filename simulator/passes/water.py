"""Han River surface: JONSWAP spectrum, mask clipping, planar reflection."""

from __future__ import annotations

from dataclasses import replace
import math

import moderngl
import numpy as np

from .. import shaders
from ..config import AtmosphereConfig
from ..water import (
    WaterConfig,
    build_directional_spectrum,
    build_water_mesh,
    estimate_fetch_length_m,
    relax_wave_spectrum,
)

WATER_MASK_UNIT = 1
REFLECTION_UNIT = 7


def wind_from_bearing_deg(wind_xz: np.ndarray) -> float:
    """Meteorological bearing the wind blows *from*, from an East/South vector."""

    return (
        math.degrees(math.atan2(-float(wind_xz[0]), float(wind_xz[1]))) % 360.0
    )


class WaterPass:
    """Near and far water grids sharing one spectrum.

    The spectrum relaxes toward the current wind rather than being rebuilt, so
    crossing a weather sample does not pop the surface.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        atmosphere: AtmosphereConfig,
        water_mask: np.ndarray,
        water_mask_bounds: np.ndarray,
    ) -> None:
        self.ctx = ctx
        self.water_mask_cpu = water_mask
        self.water_mask_bounds_cpu = water_mask_bounds

        wind = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float32)
        wind_xz = wind[[0, 2]]
        wind_speed = float(np.linalg.norm(wind_xz))
        self.config = WaterConfig(
            wind_speed_mps=max(wind_speed, 0.1),
            wind_direction_deg=wind_from_bearing_deg(wind_xz),
            fetch_length_m=estimate_fetch_length_m(
                water_mask, water_mask_bounds, wind_xz
            ),
        )
        self.spectrum = build_directional_spectrum(self.config)
        self.significant_wave_height_m = self.spectrum.significant_wave_height_m
        self.atmosphere_accumulator_s = 0.0

        near_vertices, near_indices = build_water_mesh(self.config)
        far_vertices, far_indices = build_water_mesh(
            self.config,
            self.config.far_grid_size,
            self.config.far_extent_m,
            # The far grid omits the near patch, preventing overlap and
            # z-fighting where the two meshes meet.
            self.config.extent_m,
        )
        self.program = shaders.program(ctx, "water.vert", "water.frag")
        self.vaos: list[moderngl.VertexArray] = []
        self.buffers: list[moderngl.Buffer] = []
        for vertices, indices in (
            (far_vertices, far_indices),
            (near_vertices, near_indices),
        ):
            vertex_buffer = ctx.buffer(vertices.tobytes())
            index_buffer = ctx.buffer(indices.tobytes())
            self.buffers.extend((vertex_buffer, index_buffer))
            self.vaos.append(
                ctx.vertex_array(
                    self.program,
                    [(vertex_buffer, "2f", "in_xz")],
                    index_buffer,
                    index_element_size=4,
                )
            )
        self.program["waves"].write(self.spectrum.components.tobytes())
        self.program["phases"].write(self.spectrum.phases.tobytes())
        self.program["choppiness"] = self.config.choppiness
        self.program["reflection_texture"] = REFLECTION_UNIT
        self.program["water_mask"] = WATER_MASK_UNIT
        self.program["water_mask_bounds"].value = tuple(water_mask_bounds)

    # -- per-frame state ---------------------------------------------------

    def set_view_projection(self, matrix_bytes: bytes, camera_position) -> None:
        self.program["view_projection"].write(matrix_bytes)
        self.program["camera_position"].value = tuple(camera_position)

    def set_reflection_view_projection(self, matrix_bytes: bytes) -> None:
        self.program["reflection_view_projection"].write(matrix_bytes)

    def set_wind_speed(self, wind_speed_mps: float) -> None:
        self.program["wind_speed_mps"] = wind_speed_mps

    def set_ambient_scale(self, scale: float) -> None:
        self.program["sky_ambient_scale"] = scale

    def update_forcing(
        self, atmosphere: AtmosphereConfig, frame_dt_s: float
    ) -> None:
        """Relax the spectrum toward the current wind and fetch."""

        self.atmosphere_accumulator_s += frame_dt_s
        interval_s = self.config.atmosphere_update_interval_s
        if self.atmosphere_accumulator_s < interval_s:
            return
        elapsed_s = self.atmosphere_accumulator_s
        self.atmosphere_accumulator_s %= interval_s
        wind = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float32)
        wind_xz = wind[[0, 2]]
        target_config = replace(
            self.config,
            wind_speed_mps=max(float(np.linalg.norm(wind_xz)), 0.1),
            wind_direction_deg=wind_from_bearing_deg(wind_xz),
            fetch_length_m=estimate_fetch_length_m(
                self.water_mask_cpu, self.water_mask_bounds_cpu, wind_xz
            ),
        )
        self.spectrum = relax_wave_spectrum(
            self.spectrum,
            build_directional_spectrum(target_config),
            elapsed_s,
            self.config.wind_response_time_s,
        )
        self.program["waves"].write(self.spectrum.components.tobytes())
        self.significant_wave_height_m = self.spectrum.significant_wave_height_m

    # -- drawing -----------------------------------------------------------

    def draw(self, time_s: float, reflection_texture: moderngl.Texture) -> None:
        self.program["time_s"] = time_s
        reflection_texture.use(REFLECTION_UNIT)
        for vao in self.vaos:
            vao.render(moderngl.TRIANGLES)
