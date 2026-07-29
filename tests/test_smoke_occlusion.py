from __future__ import annotations

import moderngl
import numpy as np
import pygame
import pytest

from simulator.renderer import (
    SMOKE_FRAGMENT,
    SMOKE_VERTEX,
    _look_at,
    _perspective,
)
from simulator.volume import box_vertices


@pytest.fixture(scope="module")
def gl_context() -> moderngl.Context:
    try:
        pygame.init()
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MAJOR_VERSION, 3
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MINOR_VERSION, 3
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE,
        )
        pygame.display.set_mode(
            (64, 64), pygame.OPENGL | pygame.HIDDEN, vsync=0
        )
        context = moderngl.create_context()
    except (pygame.error, moderngl.Error) as error:
        pytest.skip(f"OpenGL 3.3 unavailable: {error}")
    yield context
    pygame.quit()


def _projected_depth(
    view_projection: np.ndarray, world_position: tuple[float, float, float]
) -> float:
    clip = view_projection @ np.array([*world_position, 1.0])
    return float((clip[2] / clip[3]) * 0.5 + 0.5)


def _render_centre_alpha(
    context: moderngl.Context, opaque_depth: float
) -> float:
    size = (64, 64)
    program = context.program(
        vertex_shader=SMOKE_VERTEX,
        fragment_shader=SMOKE_FRAGMENT,
    )
    vertices = box_vertices((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    indices = np.array(
        [
            0, 3, 2, 0, 2, 1,
            4, 5, 6, 4, 6, 7,
            0, 4, 7, 0, 7, 3,
            1, 2, 6, 1, 6, 5,
            0, 1, 5, 0, 5, 4,
            3, 7, 6, 3, 6, 2,
        ],
        dtype=np.uint32,
    )
    vertex_buffer = context.buffer(vertices.tobytes())
    index_buffer = context.buffer(indices.tobytes())
    vao = context.vertex_array(
        program,
        [(vertex_buffer, "3f", "in_position")],
        index_buffer,
    )
    camera = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    view_projection = _perspective(50.0, 1.0, 0.1, 100.0) @ _look_at(
        camera, np.zeros(3, dtype=np.float32)
    )
    program["view_projection"].write(
        view_projection.T.astype(np.float32).tobytes()
    )
    program["inverse_view_projection"].write(
        np.linalg.inv(view_projection).T.astype(np.float32).tobytes()
    )
    program["camera_position"].value = tuple(camera)
    program["volume_min"].value = (-1.0, -1.0, -1.0)
    program["volume_max"].value = (1.0, 1.0, 1.0)
    program["smoke_field_min"].value = (-1.0, -1.0, -1.0)
    program["smoke_field_max"].value = (1.0, 1.0, 1.0)
    program["smoke_xy_bounds"].value = (-1.0, -1.0, 1.0, 1.0)
    program["depth_profile_sigma_m"] = 1.0
    program["depth_profile_scale"] = 1.0
    program["camera_inside"] = 0
    program["ray_steps"] = 8
    program["depth_bias_m"] = 0.2
    program["smoke_is_3d"] = 1
    program["smoke_state"] = 4
    program["smoke_state_3d"] = 5
    program["scene_depth"] = 6

    dummy_2d = context.texture((2, 2), 2, dtype="f4")
    state = np.zeros((4, 4, 4, 2), dtype=np.float32)
    state[..., 0] = 0.0002
    smoke_3d = context.texture3d((4, 4, 4), 2, state.tobytes(), dtype="f4")
    depth = context.texture(
        size,
        1,
        np.full(size[::-1], opaque_depth, dtype=np.float32).tobytes(),
        dtype="f4",
    )
    dummy_2d.use(4)
    smoke_3d.use(5)
    depth.use(6)

    color = context.texture(size, 4, dtype="f4")
    framebuffer = context.framebuffer([color])
    framebuffer.use()
    framebuffer.clear(0.0, 0.0, 0.0, 0.0)
    context.disable(moderngl.BLEND)
    context.disable(moderngl.DEPTH_TEST)
    vao.render(moderngl.TRIANGLES)
    pixels = np.frombuffer(color.read(), dtype=np.float32).reshape(64, 64, 4)
    return float(pixels[32, 32, 3])


def test_scene_depth_terminates_smoke_before_opaque_surface(
    gl_context: moderngl.Context,
) -> None:
    camera = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    view_projection = _perspective(50.0, 1.0, 0.1, 100.0) @ _look_at(
        camera, np.zeros(3, dtype=np.float32)
    )
    clear_alpha = _render_centre_alpha(gl_context, 1.0)
    foreground_depth = _projected_depth(
        view_projection, (0.0, 0.0, 2.0)
    )
    interior_depth = _projected_depth(
        view_projection, (0.0, 0.0, 0.0)
    )
    background_depth = _projected_depth(
        view_projection, (0.0, 0.0, -2.0)
    )
    occluded_alpha = _render_centre_alpha(gl_context, foreground_depth)
    interior_alpha = _render_centre_alpha(gl_context, interior_depth)
    background_alpha = _render_centre_alpha(gl_context, background_depth)
    assert clear_alpha > 0.7
    assert occluded_alpha == 0.0
    assert 0.0 < interior_alpha < clear_alpha
    assert background_alpha == pytest.approx(clear_alpha, abs=1e-6)
