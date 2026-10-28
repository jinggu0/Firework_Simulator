from __future__ import annotations

import moderngl
import numpy as np
import pytest

from simulator import shaders
from tools.shader_candidate import DEFAULT_CANDIDATE_DIRECTORY, apply_replacements, load_candidate


def candidate():
    return load_candidate(DEFAULT_CANDIDATE_DIRECTORY / "window_cell_coverage_integral.json")


def patched_source():
    return apply_replacements(shaders.source("scene.frag"), candidate().replacements)


def test_candidate_only_changes_helper_not_near_path_or_footprint():
    source, patched = shaders.source("scene.frag"), patched_source()
    begin, end = "// Window light at one point", "// Derivative cotangent frame:"
    assert source[:source.index(begin)] == patched[:patched.index(begin)]
    assert source[source.index(end):] == patched[patched.index(end):]
    assert "vec2 within = grid - cell;" in patched
    assert "first_cell = floor(grid - antialias)" in patched
    assert "last_cell = floor(grid + antialias)" in patched
    assert candidate().render_overrides == {}


def test_authored_bounds_make_interval_product_equal_kernel_cdf_difference():
    import re

    bounds = [tuple(map(float, match)) for match in re.findall(
        r"pane_bounds = vec4\((\.[\d]+), (\.[\d]+), (\.[\d]+), (\.[\d]+)\)", shaders.source("scene.frag"))]
    assert len(bounds) == 9
    for x0, x1, y0, y1 in bounds:
        assert min(x1 - x0, y1 - y0) > 2 * .12
        assert 0 <= min(x0, y0) and max(x1, y1) <= 1


@pytest.fixture(scope="module")
def context():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    yield ctx
    ctx.release()


VERTEX = """#version 330
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


def render(ctx, body, *, original=False, aa=.12, threshold=.5, width=601, start=.85, step=.0005, y=.4,
           hash_body="return fract(p.x * .37 + p.y * .53 + .27);"):
    source = shaders.source("scene.frag") if original else patched_source()
    interval = source[source.index("float interval("):source.index("// Window light at one point")]
    begin = "vec3 window_emission_at(" if original else "vec3 window_emission_in_cell("
    helpers = source[source.index(begin):source.index("// Derivative cotangent frame:")]
    # Controlled room hashes isolate coverage integration from GPU-dependent
    # floating point hash rounding. Production linkage is tested separately.
    fragment = """#version 330
out vec4 frag_color;
uniform float aa, threshold, start_x, step_x, grid_y;
""" + "float hash21(vec2 p) { " + hash_body + " }\n" + interval + helpers + """
void main() {
    vec2 grid = vec2(start_x + (gl_FragCoord.x - .5) * step_x, grid_y);
    vec2 antialias = vec2(aa);
    frag_color = vec4(""" + body + """, 1.0);
}
"""
    program = ctx.program(vertex_shader=VERTEX, fragment_shader=fragment)
    target = ctx.simple_framebuffer((width, 1), components=4, dtype="f4")
    vao = ctx.vertex_array(program, [])
    try:
        for name, value in {"aa": aa, "threshold": threshold, "start_x": start,
                            "step_x": step, "grid_y": y}.items():
            if name in program:
                program[name].value = value
        target.use()
        vao.render(vertices=3)
        data = np.frombuffer(target.read(components=3, dtype="f4"), np.float32).reshape(width, 3).copy()
        assert ctx.error == "GL_NO_ERROR"
        return data
    finally:
        vao.release()
        target.release()
        program.release()


CALL = "window_emission_at(grid, vec2(0.0), vec4(.04, .96, .08, .92), antialias, threshold, 0.0)"


@pytest.mark.opengl
def test_cell_coverage_matches_numeric_kernel_integral_of_each_actual_room(context):
    actual = render(context, CALL)
    # Query each actual room's modulation independently with a fully open pane.
    room = [render(context, f"window_emission_in_cell(grid, vec2({cell}.0, 0.0), vec2(0.0), "
                   "vec4(-2.0, 3.0, -2.0, 3.0), antialias, threshold, 0.0)") for cell in (0, 1)]
    assert np.max(room[0]) == 0 and np.max(room[1]) > .01
    grid = .85 + np.arange(601) * .0005
    # Independent midpoint integration of the normalized Epanechnikov density.
    count = 16384
    t = ((np.arange(count) + .5) / count * 2 - 1) * .12
    density = 3 / (4 * .12) * (1 - (t / .12) ** 2)
    expected = np.zeros_like(actual, dtype=float)
    for cell in (0, 1):
        inside = (grid[:, None] + t >= cell + .04) & (grid[:, None] + t <= cell + .96)
        weight = (inside * density).sum(axis=1) * (.24 / count)
        expected += room[cell] * weight[:, None]
    assert actual == pytest.approx(expected, abs=6e-5)


@pytest.mark.opengl
def test_cell_seam_is_continuous_instead_of_switching_to_the_next_occupied_room(context):
    actual = render(context, CALL, width=2, start=.99999, step=.00002)
    old = render(context, CALL, original=True, width=2, start=.99999, step=.00002)
    old_jump = np.linalg.norm(old[1] - old[0])
    assert old_jump > .01
    assert np.linalg.norm(actual[1] - actual[0]) < old_jump / 100


@pytest.mark.opengl
def test_no_cell_overlap_matches_original_helper_and_unoccupied_rooms_stay_dark(context):
    old = render(context, CALL, original=True, aa=.01, start=.2, step=.001, width=601)
    actual = render(context, CALL, aa=.01, start=.2, step=.001, width=601)
    assert actual == pytest.approx(old, abs=1e-7)
    dark = render(context, CALL, threshold=1.1)
    assert np.all(dark == 0)


@pytest.mark.opengl
def test_negative_cells_and_two_axis_overlap_use_all_four_contributions(context):
    body = " + ".join(
        f"window_emission_in_cell(grid, vec2({x}.0, {y}.0), vec2(0.0), "
        "vec4(.04, .96, .08, .92), antialias, threshold, 0.0)"
        for y in (-1, 0) for x in (-1, 0))
    expected = render(context, body, start=-.02, step=.0001, width=101, y=-.01, threshold=0.)
    actual = render(context, CALL, start=-.02, step=.0001, width=101, y=-.01, threshold=0.)
    assert np.max(expected) > 0
    assert actual == pytest.approx(expected, abs=1e-7)


@pytest.mark.opengl
def test_gpu_phase_integral_preserves_aperture_area_including_tails_across_cells(context):
    width = 16384
    settings = dict(width=width, start=.5 / width, step=1 / width, threshold=0., hash_body="return .5;")
    # Identical rooms let us divide out the nonzero interior modulation in
    # the SAME shader and integrate coverage alone. Separately compiled light
    # probes need not have identical constant folding. Not a scene-mean test.
    light = "window_emission_in_cell(grid, vec2(0.0), vec2(0.0), " \
            "vec4(-2.0, 3.0, -2.0, 3.0), antialias, threshold, 0.0)"
    old_light = "window_emission_at(grid, vec2(0.0), vec4(-2.0, 3.0, -2.0, 3.0), " \
                "antialias, threshold, 0.0)"
    actual = render(context, CALL + " / " + light, **settings)
    old = render(context, CALL + " / " + old_light, original=True, **settings)
    assert np.isfinite(actual).all() and np.min(actual) >= 0
    assert np.mean(actual) == pytest.approx(.96 - .04, abs=2e-6)
    assert np.mean(old) < .96 - .04 - .001


@pytest.mark.opengl
def test_candidate_links_with_production_vertex_shader(context):
    program = context.program(vertex_shader=shaders.source("scene.vert"), fragment_shader=patched_source())
    try:
        assert "window_radiance_w_m2_sr" in program
        assert context.error == "GL_NO_ERROR"
    finally:
        program.release()
