"""Framebuffers the render passes share.

Targets are separated from passes because several passes write the same HDR
buffer and the validation harness reads it directly. Keeping them in one object
means a resolution or format change happens in a single place, rather than in
whichever constructor happened to allocate the texture.
"""

from __future__ import annotations

import moderngl
import numpy as np

from ..config import RenderConfig


class RenderTargets:
    """Owns the HDR, reflection, and bloom attachments for one resolution."""

    __slots__ = (
        "ctx",
        "hdr_texture",
        "scene_depth_texture",
        "hdr_fbo",
        "smoke_fbo",
        "reflection_texture",
        "reflection_depth",
        "reflection_fbo",
        "reflection_size",
        "bloom_textures",
        "bloom_fbos",
        "bloom_size",
        "adaptation_size",
        "adaptation_textures",
        "adaptation_fbos",
        "adaptation_index",
    )

    def __init__(self, ctx: moderngl.Context, config: RenderConfig) -> None:
        self.ctx = ctx
        size = (config.width, config.height)
        # RGBA16F: the linear scene-referred buffer every colour metric reads
        # before exposure and tone mapping.
        self.hdr_texture = ctx.texture(size, components=4, dtype="f2")
        # A sampleable depth texture rather than a renderbuffer, so the smoke
        # ray marcher can terminate against opaque geometry.
        self.scene_depth_texture = ctx.depth_texture(size)
        self.scene_depth_texture.compare_func = ""
        self.scene_depth_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.scene_depth_texture.repeat_x = False
        self.scene_depth_texture.repeat_y = False
        self.hdr_fbo = ctx.framebuffer([self.hdr_texture], self.scene_depth_texture)
        # The same colour target without the depth attachment. Compositing
        # smoke needs to sample the depth it would otherwise be bound to,
        # which would be a texture feedback loop.
        self.smoke_fbo = ctx.framebuffer([self.hdr_texture])

        self.reflection_size = (
            max(int(config.width * config.reflection_scale), 1),
            max(int(config.height * config.reflection_scale), 1),
        )
        self.reflection_texture = ctx.texture(
            self.reflection_size, components=4, dtype="f2"
        )
        self.reflection_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.reflection_texture.repeat_x = False
        self.reflection_texture.repeat_y = False
        self.reflection_depth = ctx.depth_renderbuffer(self.reflection_size)
        self.reflection_fbo = ctx.framebuffer(
            [self.reflection_texture], self.reflection_depth
        )

        self.bloom_size = (
            max(config.width // 2, 1),
            max(config.height // 2, 1),
        )
        self.bloom_textures = [
            ctx.texture(self.bloom_size, components=4, dtype="f2")
            for _ in range(2)
        ]
        for texture in self.bloom_textures:
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.repeat_x = False
            texture.repeat_y = False
        self.bloom_fbos = [
            ctx.framebuffer(color_attachments=[texture])
            for texture in self.bloom_textures
        ]

        # Local retinal adaptation, ping-ponged so a frame can read the
        # previous state while writing the next. Quarter resolution: adaptation
        # pools over about a degree of visual angle, so per-pixel state would
        # be finer than the process it models.
        self.adaptation_size = (
            max(config.width // 4, 1),
            max(config.height // 4, 1),
        )
        self.adaptation_textures = [
            ctx.texture(self.adaptation_size, components=1, dtype="f2")
            for _ in range(2)
        ]
        for texture in self.adaptation_textures:
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.repeat_x = False
            texture.repeat_y = False
        self.adaptation_fbos = [
            ctx.framebuffer(color_attachments=[texture])
            for texture in self.adaptation_textures
        ]
        for framebuffer in self.adaptation_fbos:
            framebuffer.clear(0.0, 0.0, 0.0, 1.0)
        self.adaptation_index = 0

    @property
    def current_adaptation(self) -> moderngl.Texture:
        return self.adaptation_textures[self.adaptation_index]

    @property
    def previous_adaptation(self) -> moderngl.Texture:
        return self.adaptation_textures[1 - self.adaptation_index]

    def swap_adaptation(self) -> None:
        self.adaptation_index = 1 - self.adaptation_index

    def read_linear_hdr(self) -> np.ndarray:
        """Linear float32 RGBA in image order, for validation capture."""

        width, height = self.hdr_texture.size
        raw = np.frombuffer(self.hdr_texture.read(), dtype=np.float16)
        return np.flipud(raw.reshape(height, width, 4).astype(np.float32)).copy()
