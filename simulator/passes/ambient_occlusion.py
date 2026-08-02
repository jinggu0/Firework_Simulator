"""Fixed-cost, half-resolution ambient obscurance for contact depth.

This is an OpenGL 3.3 adaptation of two public real-time strategies rather
than a port of their API-specific code: CACAO's recommended downsampled path
and Scalable Ambient Obscurance's fixed sample budget. The shader estimates the
local depth plane before classifying a neighbour, avoiding the classic SSAO
darkening of a uniformly sloped road.
"""

from __future__ import annotations

import moderngl

from .. import shaders
from ..config import RenderConfig
from .targets import RenderTargets

SCENE_DEPTH_UNIT = 6
AMBIENT_OCCLUSION_UNIT = 13


class AmbientOcclusionPass:
    """Generate contact obscurance and multiply it into opaque radiance."""

    def __init__(
        self,
        ctx: moderngl.Context,
        config: RenderConfig,
        quad_buffer: moderngl.Buffer,
        tan_half_fov: float,
        near_plane_m: float,
        far_plane_m: float,
    ) -> None:
        self.ctx = ctx
        self.generate_program = shaders.program(
            ctx, "quad.vert", "ambient_occlusion.frag"
        )
        self.generate_vao = ctx.simple_vertex_array(
            self.generate_program, quad_buffer, "in_position"
        )
        self.generate_program["scene_depth"] = SCENE_DEPTH_UNIT
        self.generate_program["tan_half_fov"] = tan_half_fov
        self.generate_program["aspect"] = config.width / config.height
        self.generate_program["effect_radius_m"] = (
            config.ambient_occlusion_radius_m
        )
        self.generate_program["strength"] = config.ambient_occlusion_strength
        self.generate_program["near_plane_m"] = near_plane_m
        self.generate_program["far_plane_m"] = far_plane_m
        self.generate_program["occlusion_resolution"].value = (
            max(int(config.width * config.ambient_occlusion_scale), 1),
            max(int(config.height * config.ambient_occlusion_scale), 1),
        )

        self.apply_program = shaders.program(
            ctx, "quad.vert", "ambient_occlusion_apply.frag"
        )
        self.apply_vao = ctx.simple_vertex_array(
            self.apply_program, quad_buffer, "in_position"
        )
        self.apply_program["ambient_occlusion"] = AMBIENT_OCCLUSION_UNIT
        self.apply_program["scene_depth"] = SCENE_DEPTH_UNIT
        self.apply_program["near_plane_m"] = near_plane_m
        self.apply_program["far_plane_m"] = far_plane_m

    def draw(self, targets: RenderTargets) -> None:
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        targets.ambient_occlusion_fbo.use()
        targets.scene_depth_texture.use(SCENE_DEPTH_UNIT)
        self.generate_vao.render(moderngl.TRIANGLE_STRIP)

        # Multiply only the already-rendered opaque target. The apply shader
        # emits one for sky pixels, and this runs before haze so in-scattered
        # airlight is never incorrectly occluded.
        targets.composite_fbo.use()
        targets.ambient_occlusion_texture.use(AMBIENT_OCCLUSION_UNIT)
        targets.scene_depth_texture.use(SCENE_DEPTH_UNIT)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ZERO, moderngl.SRC_COLOR
        self.apply_vao.render(moderngl.TRIANGLE_STRIP)
