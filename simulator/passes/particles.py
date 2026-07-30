"""Burning stars, drawn as exposure-integrated trails."""

from __future__ import annotations

import moderngl
import numpy as np

from .. import shaders
from ..config import PhysicalCameraConfig, RenderConfig
from ..physics import FireworkWorld

STRIDE = 10
"""Floats per star: position, trail start, linear colour, radiant power."""


class ParticlePass:
    """Additive star pass with a geometry-shader motion trail.

    Each primitive spans from the star's current position back along its
    velocity by the configured shutter time, so the trail length is an exposure
    property rather than an authored effect.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        config: RenderConfig,
        camera_config: PhysicalCameraConfig,
    ) -> None:
        self.ctx = ctx
        self.camera_config = camera_config
        self.program = shaders.program(
            ctx, "particle.vert", "particle.frag", "particle.geom"
        )
        self.program["viewport_size"].value = (
            float(config.width),
            float(config.height),
        )
        self.buffer = ctx.buffer(
            reserve=config.max_particles * STRIDE * 4, dynamic=True
        )
        self.vao = ctx.vertex_array(
            self.program,
            [
                (
                    self.buffer,
                    "3f 3f 3f 1f",
                    "in_position",
                    "in_trail_start",
                    "in_color",
                    "in_power",
                )
            ],
        )
        # Reused every frame so a 250,000-particle capacity does not allocate
        # a fresh staging array per frame.
        self._staging = np.empty((config.max_particles, STRIDE), dtype=np.float32)

    def set_view_projection(self, matrix_bytes: bytes) -> None:
        self.program["view_projection"].write(matrix_bytes)

    def draw(
        self,
        world: FireworkWorld,
        radiant_power_w: np.ndarray,
        time_s: float,
    ) -> None:
        count = world.stars.count
        if not count:
            return
        data = self._staging[:count]
        data[:, :3] = world.stars.position_m[:count]
        data[:, 3:6] = (
            world.stars.position_m[:count]
            - world.stars.velocity_mps[:count]
            * self.camera_config.shutter_time_s
        )
        data[:, 6:9] = world.stars.current_color_linear()
        data[:, 9] = radiant_power_w
        self.buffer.write(data.tobytes())
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.program["time_s"] = time_s
        # Stars are drawn twice: once mirrored under the water datum, once
        # directly. Both are emissive and neither writes depth.
        for reflection in (1.0, 0.0):
            self.program["reflection"] = reflection
            self.vao.render(moderngl.POINTS, vertices=count)
