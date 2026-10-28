from __future__ import annotations

from pathlib import Path

import moderngl
import numpy as np
import pytest

from simulator import shaders
from tools.shader_candidate import apply_replacements, load_candidate


PATH = Path("docs/validation/shimmer_ab_v2/diagnostics/diagnostic_bridge_fixtures_at_mean.json")


def test_fixture_diagnostic_changes_only_the_bridge_fixture_and_adds_helper():
    candidate = load_candidate(PATH)
    source = shaders.source("scene.frag")
    patched = apply_replacements(source, candidate.replacements)
    assert not candidate.render_overrides
    # Emission's two facade paths must remain byte-for-byte identical.
    assert patched[patched.index("    if (surface > .5) {"):] == source[source.index("    if (surface > .5) {"):]
    assert "(.025 + fixture * .42)" in patched
    assert "float edge = smoothstep(.72, .94, abs(surface_uv.y));" in patched
    assert "max(fwidth(surface_uv.x / period), .002)" in patched


@pytest.mark.opengl
def test_actual_glsl_period_mean_matches_numerical_integral_including_wide_pulses():
    candidate = load_candidate(PATH)
    helper = candidate.replacements[0].new.partition("float hash21")[0]
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    program = ctx.program(vertex_shader="""#version 330
        const vec2 p[3] = vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));
        void main(){gl_Position=vec4(p[gl_VertexID],0,1);}
        """, fragment_shader="#version 330\nuniform float footprint;\nout float result;\n" + helper + """
        void main(){result=diagnostic_fixture_period_mean(.018+footprint,.050+2.0*footprint);}
        """)
    texture = ctx.texture((1, 1), 1, dtype="f4")
    target = ctx.framebuffer([texture])
    vao = ctx.vertex_array(program, [])
    try:
        target.use()
        phase = np.abs((np.arange(200_000) + .5) / 200_000 - .5)
        for footprint in (.002, .01, .10, .225, .30, .482, .60, 2.0):
            lower, upper = .018 + footprint, .050 + 2.0 * footprint
            t = np.clip((phase - lower) / (upper - lower), 0.0, 1.0)
            expected = np.mean(1.0 - t * t * (3.0 - 2.0 * t))
            program["footprint"].value = footprint
            vao.render(moderngl.TRIANGLES, vertices=3)
            measured = np.frombuffer(target.read(components=1, dtype="f4"), np.float32)[0]
            assert measured == pytest.approx(expected, abs=2e-6)
        assert ctx.error == "GL_NO_ERROR"
    finally:
        vao.release()
        target.release()
        texture.release()
        program.release()
        ctx.release()
