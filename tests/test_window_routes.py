from __future__ import annotations

from types import SimpleNamespace

import moderngl
import numpy as np
import pytest

from simulator import shaders
from simulator.passes.scene import ScenePass
from tools.measure_window_routes import (
    REGIONS, REPLACEMENTS, reflection_difference, receiver_statistics, verify_receiver_baseline, window_route,
)
from tools.shader_candidate import apply_replacements


@pytest.mark.parametrize("route", [None, "direct", "reflection"])
def test_draw_route_sets_explicit_flags_and_restores_methods(monkeypatch, route):
    observed = []

    def draw(name):
        def run(scene):
            observed.append((name, scene.program["audit_suppress_windows"].value,
                             scene.program["audit_real_eye"].value))
        return run

    direct, reflection = draw("direct"), draw("reflection")
    monkeypatch.setattr(ScenePass, "draw", direct)
    monkeypatch.setattr(ScenePass, "draw_reflection", reflection)
    scene = SimpleNamespace(program={"audit_real_eye": SimpleNamespace(),
                                     "audit_suppress_windows": SimpleNamespace()})
    with pytest.raises(RuntimeError):
        with window_route(route, (1, 2, 3)) as calls:
            ScenePass.draw_reflection(scene)
            ScenePass.draw(scene)
            raise RuntimeError("fixture failure")
    assert observed == [("reflection", route == "reflection", (1., 2., 3.)),
                        ("direct", route == "direct", (1., 2., 3.))]
    assert calls["direct"]["suppressed"] == int(route == "direct")
    assert calls["reflection"]["suppressed"] == int(route == "reflection")
    assert ScenePass.draw is direct and ScenePass.draw_reflection is reflection


def test_same_receiver_samples_and_hdr_scale_are_used_after_dimming():
    data = np.stack([np.full((16, 20), value, dtype=float) for value in (2., 3., 4.)])
    scored = np.zeros((16, 20), bool)
    scored[7:9, 7:13] = True
    labels = np.zeros((2, 12), dtype=np.uint8)
    labels[:, 6:] = REGIONS.index("water_interior")
    pan = {"display_luma": data * 10, "linear_hdr": data, "matrices": [np.eye(4)] * 3}
    scale = np.full(12, 4.)
    control, retained = receiver_statistics(pan, scored, scale, labels)
    dimmed, _ = receiver_statistics({**pan, "linear_hdr": data * .5}, scored, scale, retained)
    assert retained is labels
    water = control["water_interior"]
    assert water["samples"] == 12
    assert water["mean_display_code"] == 35.
    assert water["display_energy"] == pytest.approx(1200.)
    assert water["hdr_energy"] == pytest.approx(12 / 16)
    assert dimmed["water_interior"]["hdr_energy"] == pytest.approx(water["hdr_energy"] / 4)
    assert dimmed["water_interior"]["mean_linear_hdr"] == 1.75
    assert control["facade_water_mixed"]["mean_display_code"] is None
    verify_receiver_baseline(control, control)
    with pytest.raises(ValueError, match="hdr_energy"):
        verify_receiver_baseline(dimmed, control)


def test_reflection_comparison_counts_pixels_not_channels():
    control = np.ones((2, 3, 3), dtype=np.float16)
    edited = control.copy()
    edited[1, 0] = 0
    report = reflection_difference(control, edited)
    assert report["changed_pixels"] == 1
    assert report["mean_absolute_rgb_delta"] == pytest.approx(1 / 6)
    assert report["max_absolute_rgb_delta"] == 1


@pytest.mark.opengl
def test_diagnostic_shader_links_and_only_suppresses_final_window_emission():
    original = shaders.source("scene.frag")
    patched = apply_replacements(original, REPLACEMENTS)
    assert patched.count("emission = vec3(0.0);") == 1
    assert patched.count("frag_color =") == original.count("frag_color =")
    assert patched.count("discard;") == original.count("discard;")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        program = ctx.program(vertex_shader=shaders.source("scene.vert"), fragment_shader=patched)
        try:
            assert "audit_real_eye" in program and "audit_suppress_windows" in program
        finally:
            program.release()
    finally:
        ctx.release()
