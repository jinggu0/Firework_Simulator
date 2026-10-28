from __future__ import annotations

import re

import moderngl
import numpy as np
import pytest

from simulator import shaders
from tools.shader_candidate import DEFAULT_CANDIDATE_DIRECTORY, apply_replacements, load_candidate


def candidate():
    return load_candidate(DEFAULT_CANDIDATE_DIRECTORY / "window_boundary_moment_filter.json")


def offsets():
    text = candidate().replacements[0].new
    matches = re.findall(r"vec2\(\s*([-.\d]+),\s*([-.\d]+)\)", text)
    points = np.asarray(matches, dtype=float)
    assert points.shape == (8, 2)
    return np.concatenate([points, -points])


def test_kernel_is_centred_normalized_and_matches_declared_variance():
    points = offsets()
    weights = np.full(16, 1 / 16)
    assert weights.sum() == 1 and np.all(weights > 0)
    assert points.mean(axis=0) == pytest.approx([0, 0], abs=1e-15)
    covariance = points.T @ points / 16
    assert covariance == pytest.approx(np.eye(2) * (75 / 256), abs=1e-8)
    # Numeric quadrature of the actual smoothstep derivative, not just a
    # restatement of the candidate's variance formula.
    x = np.linspace(-1.25, 1.25, 100001)
    kernel = 3 / (4 * 1.25) * (1 - (x / 1.25) ** 2)
    variance = np.trapezoid(x * x * kernel, x)
    assert variance == pytest.approx(covariance[0, 0] + .3125 ** 2 / 5, abs=1e-8)


def test_candidate_changes_only_offsets_and_leaves_near_emission_and_fade_untouched():
    change, = candidate().replacements
    source = shaders.source("scene.frag")
    patched = apply_replacements(source, candidate().replacements)
    assert source.replace(change.old, "OFFSETS") == patched.replace(change.new, "OFFSETS")
    assert "averaged / 16.0" in patched
    assert "i % 8" in patched and "i < 8" in patched
    assert not candidate().render_overrides


@pytest.mark.parametrize("frequency", [1, 3, 11])
def test_normalized_periodic_filter_preserves_phase_mean_without_an_expected_value(frequency):
    # Integer phase shifts on a dense periodic grid conserve the mean exactly.
    # This property is not a promise about finite, view-dependent facades.
    phase = np.arange(65536) / 65536
    signal = ((phase > .13) & (phase < .82)) * (.52 + .48 * ((phase * frequency) % 1 > .25))
    filtered = sum(np.roll(signal, round(float(point[0]) * len(signal) * .1)) for point in offsets()) / 16
    assert filtered.mean() == pytest.approx(signal.mean(), abs=1e-15)
    assert np.all(filtered >= 0) and np.all(filtered <= 1)


@pytest.mark.opengl
def test_candidate_links_with_production_vertex_shader():
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        patched = apply_replacements(shaders.source("scene.frag"), candidate().replacements)
        program = context.program(vertex_shader=shaders.source("scene.vert"), fragment_shader=patched)
        assert "window_radiance_w_m2_sr" in program
        program.release()
        assert context.error == "GL_NO_ERROR"
    finally:
        context.release()
