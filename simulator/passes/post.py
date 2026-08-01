"""Bloom and the display transform.

Kept as one unit because bloom is evaluated *before* the sensor stage: it is a
compact optical point-spread approximation of the lens, not a post-display
effect. Everything upstream of this module works in linear scene-referred
radiance; everything this module emits is display-referred and no longer valid
input to a colour metric.
"""

from __future__ import annotations

from enum import Enum
import math

import moderngl

from .. import shaders
from ..camera_optics import (
    LensDistortion,
    analog_gain,
    photon_to_electron_scale,
    white_balance_gains,
)
from ..config import PhysicalCameraConfig, RenderConfig
from ..human_vision import (
    CHROMATIC_ADAPTATION_TIME_S,
    DARK_ADAPTATION_TIME_S,
    LIGHT_ADAPTATION_TIME_S,
    HumanVisionState,
)
from .targets import RenderTargets

HDR_UNIT = 0
BLOOM_UNIT = 3
ADAPTATION_UNIT = 10
PREVIOUS_ADAPTATION_UNIT = 11

ADAPTATION_POOLING_LOD = 3.0
"""Mip level the adaptation pass reads, standing in for the pooling area."""


class DisplayMode(Enum):
    """Which observer the linear HDR buffer is presented through."""

    PHYSICAL_CAMERA = "physical_camera"
    """Sensor response: aperture, shutter, quantum efficiency, noise, full well."""

    HUMAN_VISION = "human_vision"
    """Observer response: pupil, adaptation, mesopic mixing, glare, acuity."""


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
        # The sensor's spectral response was applied and never balanced out.
        # These gains are the von Kries correction for it, referenced to a
        # Planckian at the configured temperature.
        self.tonemap_program["white_balance_gain"].value = tuple(
            float(value) for value in white_balance_gains(camera_config)
        )
        # Identity unless a measured lens calibration has been loaded into the
        # config; the shader's inversion is exact in that case.
        self.distortion = LensDistortion.from_config(camera_config)
        for name, value in self.distortion.uniforms().items():
            self.tonemap_program[name].value = value
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

        # -- human vision path ---------------------------------------------
        self.mode = DisplayMode.PHYSICAL_CAMERA
        self.vision = HumanVisionState()
        self.vision_program = shaders.program(
            ctx, "quad.vert", "human_vision.frag"
        )
        self.vision_vao = ctx.simple_vertex_array(
            self.vision_program, quad_buffer, "in_position"
        )
        self.vision_program["hdr_texture"] = HDR_UNIT
        self.vision_program["bloom_texture"] = BLOOM_UNIT
        self.vision_program["adaptation_texture"] = ADAPTATION_UNIT
        self.vision_program["bloom_strength"] = config.bloom_strength
        self.vision_program["tan_half_fov"] = tan_half_fov
        self.vision_program["aspect"] = config.width / config.height

        self.adaptation_program = shaders.program(
            ctx, "quad.vert", "adaptation.frag"
        )
        self.adaptation_vao = ctx.simple_vertex_array(
            self.adaptation_program, quad_buffer, "in_position"
        )
        self.adaptation_program["hdr_texture"] = HDR_UNIT
        self.adaptation_program["previous_adaptation"] = (
            PREVIOUS_ADAPTATION_UNIT
        )
        self.adaptation_program["pooling_lod"] = ADAPTATION_POOLING_LOD
        # The adapting white is a property of the whole field, so it is pooled
        # from the last mip level — the one texel that covers everything.
        # floor, not ceil: the chain for 1280x720 ends at level 10, and asking
        # for 11 samples a level that was never allocated. Drivers clamp, so
        # the mistake is invisible until a readback of that level returns
        # rubbish, which is how it was found.
        self.adaptation_program["global_pooling_lod"] = float(
            math.floor(math.log2(max(config.width, config.height)))
        )

    def set_mode(self, mode: DisplayMode) -> None:
        self.mode = mode

    def toggle_mode(self) -> DisplayMode:
        self.mode = (
            DisplayMode.HUMAN_VISION
            if self.mode is DisplayMode.PHYSICAL_CAMERA
            else DisplayMode.PHYSICAL_CAMERA
        )
        return self.mode

    def _run_bloom(self, targets: RenderTargets) -> None:
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

    def _advance_adaptation(
        self, targets: RenderTargets, frame_dt_s: float
    ) -> None:
        """Update the local retinal adaptation buffer for this frame."""

        targets.swap_adaptation()
        self.adaptation_program["light_response"] = 1.0 - math.exp(
            -max(frame_dt_s, 0.0) / LIGHT_ADAPTATION_TIME_S
        )
        self.adaptation_program["dark_response"] = 1.0 - math.exp(
            -max(frame_dt_s, 0.0) / DARK_ADAPTATION_TIME_S
        )
        self.adaptation_program["chromatic_response"] = 1.0 - math.exp(
            -max(frame_dt_s, 0.0) / CHROMATIC_ADAPTATION_TIME_S
        )
        targets.adaptation_fbos[targets.adaptation_index].use()
        targets.hdr_texture.use(HDR_UNIT)
        targets.previous_adaptation.use(PREVIOUS_ADAPTATION_UNIT)
        self.adaptation_vao.render(moderngl.TRIANGLE_STRIP)

    def run(
        self,
        targets: RenderTargets,
        frame_dt_s: float = 1.0 / 60.0,
        scene_illuminance_lux: float = 0.0,
    ) -> None:
        """Bloom, then present through the selected observer model."""

        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._run_bloom(targets)

        if self.mode is DisplayMode.HUMAN_VISION:
            # Mip chains: peripheral acuity samples a coarser level of the
            # scene, and both the glare tail and adaptation pooling read
            # reduced levels of their sources.
            targets.hdr_texture.build_mipmaps()
            targets.bloom_textures[0].build_mipmaps()
            self._advance_adaptation(targets, frame_dt_s)
            self.vision.update(scene_illuminance_lux, frame_dt_s)
            for name, value in self.vision.uniforms().items():
                self.vision_program[name] = value

            self.ctx.screen.use()
            self.ctx.disable(moderngl.BLEND)
            targets.hdr_texture.use(HDR_UNIT)
            targets.bloom_textures[0].use(BLOOM_UNIT)
            targets.current_adaptation.use(ADAPTATION_UNIT)
            self.vision_vao.render(moderngl.TRIANGLE_STRIP)
            return

        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)
        targets.hdr_texture.use(HDR_UNIT)
        targets.bloom_textures[0].use(BLOOM_UNIT)
        # The frame index drives the sensor's shot-noise sequence; it wraps so
        # a long capture cannot lose float precision in the shader.
        self.frame_index = (self.frame_index + 1) % 1_000_000
        self.tonemap_program["frame_index"] = float(self.frame_index)
        self.tonemap_vao.render(moderngl.TRIANGLE_STRIP)
