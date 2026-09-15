from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import moderngl
import pytest

from simulator import shaders
from tools.shader_candidate import (
    DEFAULT_CANDIDATE_DIRECTORY,
    ShaderReplacement,
    apply_replacements,
    load_candidate,
    shader_variant,
)


WINDOW_FADE = DEFAULT_CANDIDATE_DIRECTORY / "window_grid_expectation_fade.json"
FOOTPRINT_AVERAGE = DEFAULT_CANDIDATE_DIRECTORY / "window_emission_footprint_average.json"
GRID_LINE = "    vec2 grid = surface_uv / vec2(bay_width, floor_height);\n"


def test_a_replacement_changes_only_the_matched_text() -> None:
    result = apply_replacements("keep swap keep", [ShaderReplacement("x.frag", "swap", "done")])

    assert result == "keep done keep"


def test_replacements_apply_in_order() -> None:
    result = apply_replacements(
        "first",
        [
            ShaderReplacement("x.frag", "first", "second"),
            ShaderReplacement("x.frag", "second", "third"),
        ],
    )

    assert result == "third"


def test_text_that_is_absent_is_refused() -> None:
    with pytest.raises(ValueError, match="0 times"):
        apply_replacements("abc", [ShaderReplacement("x.frag", "zzz", "y")])


def test_text_that_occurs_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="2 times"):
        apply_replacements("ab ab", [ShaderReplacement("x.frag", "ab", "y")])


def test_the_variant_is_visible_only_inside_the_context() -> None:
    original = shaders.source("scene.frag")
    marked = GRID_LINE + "    // candidate marker\n"

    with shader_variant([ShaderReplacement("scene.frag", GRID_LINE, marked)]) as digests:
        patched = shaders.source("scene.frag")
        assert "// candidate marker" in patched
        assert digests["scene.frag"] == sha256(patched.encode("utf-8")).hexdigest()

    assert shaders.source("scene.frag") == original


def test_the_original_returns_even_when_the_context_raises() -> None:
    original = shaders.source("scene.frag")
    marked = GRID_LINE + "    // candidate marker\n"

    with pytest.raises(RuntimeError, match="inside"):
        with shader_variant([ShaderReplacement("scene.frag", GRID_LINE, marked)]):
            raise RuntimeError("inside")

    assert shaders.source("scene.frag") == original


def test_a_variant_that_does_not_apply_fails_before_the_context_opens() -> None:
    entered = False

    with pytest.raises(ValueError):
        with shader_variant([ShaderReplacement("scene.frag", "no such text", "x")]):
            entered = True

    assert not entered


def test_other_shaders_are_untouched_by_a_variant() -> None:
    water = shaders.source("water.frag")
    marked = GRID_LINE + "    // candidate marker\n"

    with shader_variant([ShaderReplacement("scene.frag", GRID_LINE, marked)]):
        assert shaders.source("water.frag") == water


def test_a_replacement_for_an_unknown_shader_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown shader"):
        with shader_variant([ShaderReplacement("nonexistent.frag", "a", "b")]):
            pass


def test_a_candidate_file_must_state_its_hypothesis_and_provenance(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "bare",
                "replacements": [{"file": "scene.frag", "old": "a", "new": "b"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hypothesis"):
        load_candidate(path)


def test_the_window_grid_candidate_applies_to_the_shipped_shader() -> None:
    candidate = load_candidate(WINDOW_FADE)

    patched = apply_replacements(
        shaders.source("scene.frag"),
        [item for item in candidate.replacements if item.file == "scene.frag"],
    )

    assert "cell_detail" in patched


def test_the_window_grid_candidate_never_expects_negative_occupancy() -> None:
    # The V2-3b edit faded occupancy to 1 - threshold, which is negative for the
    # National Assembly's 1.1 threshold and would have emitted negative light.
    candidate = load_candidate(WINDOW_FADE)
    patched = apply_replacements(shaders.source("scene.frag"), candidate.replacements)

    assert "clamp(1.0 - occupancy_threshold, 0.0, 1.0)" in patched


@pytest.mark.opengl
def test_the_window_grid_candidate_compiles_against_the_production_vertex_shader() -> None:
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    candidate = load_candidate(WINDOW_FADE)
    try:
        with shader_variant(candidate.replacements):
            program = shaders.program(ctx, "scene.vert", "scene.frag")
        try:
            assert "view_projection" in program
        finally:
            program.release()
    finally:
        ctx.release()


def _patched_scene(candidate_path: Path) -> str:
    candidate = load_candidate(candidate_path)
    return apply_replacements(shaders.source("scene.frag"), candidate.replacements)


def test_the_footprint_average_candidate_applies_to_the_shipped_shader() -> None:
    patched = _patched_scene(FOOTPRINT_AVERAGE)

    assert "window_emission_at(" in patched
    assert patched.count("window_emission_at(") == 2


def test_the_footprint_average_takes_its_derivatives_before_branching() -> None:
    # Derivatives inside a branch that differs between neighbouring pixels are
    # undefined in GLSL, so every screen derivative the average needs has to be
    # taken before the branch that decides whether to average.
    patched = _patched_scene(FOOTPRINT_AVERAGE)
    main = patched[patched.index("void main()"):]
    branch = main.index("if (footprint_average > 0.0)")
    body = main[branch:main.index("emission = mix(", branch)]

    assert main.index("vec2 grid_dx = dFdx(grid);") < branch
    assert main.index("vec2 grid_dy = dFdy(grid);") < branch
    assert "dFdx" not in body and "dFdy" not in body and "fwidth" not in body


def test_the_footprint_average_leaves_the_single_sample_emission_in_place() -> None:
    original = shaders.source("scene.frag")
    patched = _patched_scene(FOOTPRINT_AVERAGE)
    single = (
        "    vec3 emission = window_color * pane * occupied * blinds * curtains\n"
        "                  * window_radiance_w_m2_sr * .72;\n"
    )

    assert original.count(single) == 1 and patched.count(single) == 1


@pytest.mark.opengl
def test_the_footprint_average_candidate_compiles() -> None:
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    candidate = load_candidate(FOOTPRINT_AVERAGE)
    try:
        with shader_variant(candidate.replacements):
            program = shaders.program(ctx, "scene.vert", "scene.frag")
        try:
            assert "window_radiance_w_m2_sr" in program
        finally:
            program.release()
    finally:
        ctx.release()
