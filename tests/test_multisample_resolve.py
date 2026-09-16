from __future__ import annotations

from dataclasses import replace

import moderngl
import numpy as np
import pytest

from simulator import shaders
from simulator.config import RenderConfig
from simulator.passes.resolve import MultisampleResolvePass
from simulator.passes.targets import RenderTargets


SMALL = RenderConfig(width=64, height=32)
SINGLE = replace(SMALL, msaa_samples=1)


@pytest.fixture
def context():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:  # pragma: no cover - depends on the machine
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        yield ctx
    finally:
        ctx.release()


def test_eight_samples_ship() -> None:
    # Adopted in V2-3i against the A/B rule; four was measured first and left
    # water_reflection less stable, so the count is part of what was judged.
    assert RenderConfig().msaa_samples == 8


@pytest.mark.opengl
def test_single_sample_targets_allocate_nothing_extra(context) -> None:
    targets = RenderTargets(context, SINGLE)

    assert targets.samples == 1
    assert targets.multisample_fbo is None
    assert targets.multisample_colour is None
    assert targets.multisample_depth is None
    assert targets.depth_resolve_fbo is None


@pytest.mark.opengl
def test_the_scene_is_drawn_into_the_hdr_buffer_itself_without_multisampling(
    context,
) -> None:
    # Identical, not merely equivalent: the single-sample path binds the same
    # framebuffer object it bound before multisampling existed.
    targets = RenderTargets(context, SINGLE)

    assert targets.scene_fbo is targets.hdr_fbo


@pytest.mark.opengl
def test_multisample_targets_carry_the_configured_sample_count(context) -> None:
    targets = RenderTargets(context, SMALL)

    assert targets.samples == 8
    assert targets.multisample_colour.samples == 8
    assert targets.multisample_depth.samples == 8
    assert targets.scene_fbo is targets.multisample_fbo


@pytest.mark.opengl
def test_more_samples_than_the_context_allows_are_refused(context) -> None:
    with pytest.raises(ValueError, match="above this context's maximum"):
        RenderTargets(context, replace(SMALL, msaa_samples=context.max_samples * 2))


@pytest.mark.opengl
def test_resolving_single_sample_targets_is_refused(context) -> None:
    # Silently doing nothing would let a caller believe a resolve happened.
    quad = context.buffer(
        np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
    )
    targets = RenderTargets(context, SINGLE)
    resolver = MultisampleResolvePass(context, quad)

    with pytest.raises(ValueError, match="single-sample"):
        resolver.resolve(targets)


@pytest.mark.opengl
def test_the_resolve_copies_colour_and_depth_into_the_readable_buffers(
    context,
) -> None:
    quad = context.buffer(
        np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
    )
    targets = RenderTargets(context, SMALL)
    resolver = MultisampleResolvePass(context, quad)
    targets.hdr_fbo.use()
    targets.hdr_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)
    targets.multisample_fbo.use()
    targets.multisample_fbo.clear(0.25, 0.5, 0.75, 1.0, depth=0.5)

    resolver.resolve(targets)

    colour = np.frombuffer(targets.hdr_texture.read(), dtype=np.float16)
    depth = np.frombuffer(targets.scene_depth_texture.read(), dtype=np.float32)
    assert colour.reshape(-1, 4)[0] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert depth == pytest.approx(np.full(depth.shape, 0.5), abs=1e-6)


@pytest.mark.opengl
def test_the_resolve_leaves_the_depth_function_where_the_renderer_expects_it(
    context,
) -> None:
    # The resolve has to turn the depth test on and make it always pass, or no
    # depth would be written at all. Every later pass assumes the default.
    quad = context.buffer(
        np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()
    )
    targets = RenderTargets(context, SMALL)
    resolver = MultisampleResolvePass(context, quad)
    targets.multisample_fbo.use()
    targets.multisample_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

    resolver.resolve(targets)

    near, far = 0.25, 0.75
    targets.hdr_fbo.use()
    targets.hdr_fbo.clear(0.0, 0.0, 0.0, 1.0, depth=far)
    context.enable(moderngl.DEPTH_TEST)
    program = context.program(
        vertex_shader=(
            "#version 330\nin vec2 in_position; uniform float z;\n"
            "void main(){ gl_Position = vec4(in_position, z, 1); }"
        ),
        fragment_shader=(
            "#version 330\nout vec4 colour;\n"
            "void main(){ colour = vec4(1, 0, 0, 1); }"
        ),
    )
    vao = context.simple_vertex_array(program, quad, "in_position")
    program["z"] = near * 2.0 - 1.0
    vao.render(moderngl.TRIANGLE_STRIP)
    program["z"] = 0.9 * 2.0 - 1.0
    vao.render(moderngl.TRIANGLE_STRIP)

    depth = np.frombuffer(targets.scene_depth_texture.read(), dtype=np.float32)
    assert depth == pytest.approx(np.full(depth.shape, near), abs=1e-5)


@pytest.mark.opengl
def test_the_depth_resolve_shader_compiles_against_the_shared_quad(context) -> None:
    program = shaders.program(context, "quad.vert", "depth_resolve.frag")
    try:
        assert "depth_multisample" in program
    finally:
        program.release()
