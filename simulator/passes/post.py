"""Bloom and the display transform.

Kept as one unit because bloom is evaluated *before* the sensor stage: it is a
compact optical point-spread approximation of the lens, not a post-display
effect. Everything upstream of this module works in linear scene-referred
radiance; everything this module emits is display-referred and no longer valid
input to a colour metric.
"""

from __future__ import annotations

import moderngl

from .. import shaders
from ..camera_optics import analog_gain, photon_to_electron_scale
from ..config import PhysicalCameraConfig, RenderConfig
from .targets import RenderTargets

HDR_UNIT = 0
BLOOM_UNIT = 3


class PostProcessPass:
    """Half-resolution bloom followed by the physical sensor response."""

    def __init__(
        self,
        ctx: moderngl.Context,
        config: RenderConfig,
        camera_config: PhysicalCameraConfig,
        quad_buffer: moderngl.Buffer,
        tan_half_fov: float,
    ) -> None:
        self.ctx = ctx
        self.frame_index = 0

        self.tonemap_program = shaders.program(ctx, "quad.vert", "tonemap.frag")
        self.tonemap_vao = ctx.simple_vertex_array(
            self.tonemap_program, quad_buffer, "in_position"
        )
        self.tonemap_program["hdr_texture"] = HDR_UNIT
        self.tonemap_program["bloom_texture"] = BLOOM_UNIT
        self.tonemap_program["bloom_strength"] = config.bloom_strength
        self.tonemap_program["photon_to_electron"].value = tuple(
            photon_to_electron_scale(camera_config)
        )
        self.tonemap_program["analog_gain"] = analog_gain(camera_config)
        self.tonemap_program["full_well_electrons"] = (
            camera_config.full_well_electrons
        )
        self.tonemap_program["read_noise_electrons"] = (
            camera_config.read_noise_electrons
        )
        self.tonemap_program["tan_half_fov"] = tan_half_fov
        self.tonemap_program["aspect"] = config.width / config.height
        self.tonemap_program["sensor_noise_enabled"] = int(
            camera_config.enable_sensor_noise
        )

        self.prefilter_program = shaders.program(
            ctx, "quad.vert", "bloom_prefilter.frag"
        )
        self.prefilter_vao = ctx.simple_vertex_array(
            self.prefilter_program, quad_buffer, "in_position"
        )
        self.prefilter_program["hdr_texture"] = HDR_UNIT

        self.blur_program = shaders.program(ctx, "quad.vert", "bloom_blur.frag")
        self.blur_vao = ctx.simple_vertex_array(
            self.blur_program, quad_buffer, "in_position"
        )
        self.blur_program["source_texture"] = BLOOM_UNIT

    def run(self, targets: RenderTargets) -> None:
        """Prefilter, blur separably, then tone map to the default framebuffer."""

        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        targets.bloom_fbos[0].use()
        targets.hdr_texture.use(HDR_UNIT)
        self.prefilter_vao.render(moderngl.TRIANGLE_STRIP)

        targets.bloom_fbos[1].use()
        targets.bloom_textures[0].use(BLOOM_UNIT)
        self.blur_program["direction"].value = (1.0, 0.0)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        targets.bloom_fbos[0].use()
        targets.bloom_textures[1].use(BLOOM_UNIT)
        self.blur_program["direction"].value = (0.0, 1.0)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)
        targets.hdr_texture.use(HDR_UNIT)
        targets.bloom_textures[0].use(BLOOM_UNIT)
        # The frame index drives the sensor's shot-noise sequence; it wraps so
        # a long capture cannot lose float precision in the shader.
        self.frame_index = (self.frame_index + 1) % 1_000_000
        self.tonemap_program["frame_index"] = float(self.frame_index)
        self.tonemap_vao.render(moderngl.TRIANGLE_STRIP)
