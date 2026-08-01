"""Aerial perspective: the visibility field applied to the rendered scene.

Visibility was computed by :mod:`simulator.environment_field` and consumed by
nothing — the renderer's only air extinction was a single unsourced scalar on
the point-light paths, and the view path had none at all, so the far bank of
the Han River was rendered as if it were in vacuum. This pass makes the
Koschmieder range an appearance the frame actually shows.

It is deferred: one full-screen composite over the opaque depth buffer rather
than a term inside each surface shader. A blend on the geometry would depend on
the order the buildings happened to be drawn in, and adding the term to five
shaders would put five copies of the integral in play. The stars and the plume
are attenuated where they are drawn instead, because they are additive and must
not receive the airlight a second time.
"""

from __future__ import annotations

import moderngl

from .. import shaders
from ..atmosphere import SurfaceExtinction

SCENE_DEPTH_UNIT = 6
AIRLIGHT_UNIT = 12

EXTINCTION_STAGE = 0.0
"""Multiplies the target by the transmittance. Must precede the airlight."""

AIRLIGHT_STAGE = 1.0
"""Adds the in-scattered airlight the transmittance left room for."""


def set_air_extinction(
    programs, extinction: SurfaceExtinction
) -> None:
    """Push one atmospheric state to every program that consumes extinction.

    One call site rather than five means a renamed or dropped uniform raises
    here instead of leaving a pass quietly rendering last week's weather.
    """

    uniforms = extinction.uniforms()
    for program in programs:
        for name, value in uniforms.items():
            member = program[name]
            if isinstance(value, tuple):
                member.value = value
            else:
                member.value = float(value)


class HazePass:
    """Per-channel extinction and airlight over the opaque scene."""

    def __init__(self, ctx: moderngl.Context, quad_buffer: moderngl.Buffer) -> None:
        self.ctx = ctx
        self.program = shaders.program(ctx, "quad.vert", "haze.frag")
        self.vao = ctx.simple_vertex_array(
            self.program, quad_buffer, "in_position"
        )
        self.program["scene_depth"] = SCENE_DEPTH_UNIT
        self.program["airlight"] = AIRLIGHT_UNIT

    def set_camera(self, inverse_view_projection: bytes, camera_position) -> None:
        self.program["inverse_view_projection"].write(inverse_view_projection)
        self.program["camera_position"].value = tuple(
            float(value) for value in camera_position
        )

    def draw(
        self,
        framebuffer: moderngl.Framebuffer,
        depth_texture: moderngl.Texture,
        airlight_texture: moderngl.Texture,
        reflected_path: bool = False,
    ) -> None:
        """Attenuate, then add airlight. The order is not interchangeable.

        ``framebuffer`` must be the colour target *without* its depth
        attachment: this samples the depth it would otherwise be bound to.
        ``reflected_path`` marks the planar-reflection pre-pass, whose camera
        is mirrored below the water datum — see ``haze.frag``.
        """

        framebuffer.use()
        depth_texture.use(SCENE_DEPTH_UNIT)
        airlight_texture.use(AIRLIGHT_UNIT)
        self.program["reflected_path"] = int(reflected_path)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)

        # dst *= 1 - src, with src the scattered fraction, so dst *= T.
        self.program["haze_stage"] = EXTINCTION_STAGE
        self.ctx.blend_func = moderngl.ZERO, moderngl.ONE_MINUS_SRC_COLOR
        self.vao.render(moderngl.TRIANGLE_STRIP)

        self.program["haze_stage"] = AIRLIGHT_STAGE
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.vao.render(moderngl.TRIANGLE_STRIP)
