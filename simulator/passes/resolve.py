"""Resolving the multisampled scene buffer.

Multisample antialiasing samples geometry coverage more finely than it samples
shading: the fragment shader still runs once per covered pixel, so every
derivative-based filter in the scene shaders keeps the width it has at the
presented pixel. That is what separates geometry edge sampling from shading
filter width — a finer render resolution cannot, because it halves `fwidth`
and sharpens every analytic filter along with the geometry (V2-3h).

Colour resolves with a framebuffer blit. Depth cannot, so it is drawn.
"""

from __future__ import annotations

import moderngl

from .. import shaders
from .targets import RenderTargets

DEPTH_MULTISAMPLE_UNIT = 0


class MultisampleResolvePass:
    """Copy the multisampled scene into the buffers every later pass reads."""

    def __init__(self, ctx: moderngl.Context, quad_buffer: moderngl.Buffer) -> None:
        self.ctx = ctx
        self.program = shaders.program(ctx, "quad.vert", "depth_resolve.frag")
        self.vao = ctx.simple_vertex_array(self.program, quad_buffer, "in_position")
        self.program["depth_multisample"] = DEPTH_MULTISAMPLE_UNIT

    def resolve(self, targets: RenderTargets) -> None:
        if targets.multisample_fbo is None:
            raise ValueError("nothing to resolve: these targets are single-sample")
        self.ctx.copy_framebuffer(targets.hdr_fbo, targets.multisample_fbo)

        # A depth write only happens while the depth test is enabled, so the
        # test is enabled and made to always pass rather than turned off. The
        # function is put back to the renderer's own default afterwards.
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "1"
        targets.depth_resolve_fbo.use()
        targets.multisample_depth.use(DEPTH_MULTISAMPLE_UNIT)
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.depth_func = "<"
