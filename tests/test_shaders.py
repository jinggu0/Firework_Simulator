import re

import pytest

from simulator import shaders

EXPECTED = {
    "quad.vert",
    "background.frag",
    "particle.vert",
    "particle.geom",
    "particle.frag",
    "tonemap.frag",
    "bloom_prefilter.frag",
    "bloom_blur.frag",
    "water.vert",
    "water.frag",
    "land.vert",
    "land.frag",
    "scene.vert",
    "scene.frag",
    "smoke.vert",
    "smoke.frag",
    # Human Vision Mode: a second display transform and the local retinal
    # adaptation buffer it reads.
    "human_vision.frag",
    "adaptation.frag",
    # Aerial perspective: the deferred extinction and airlight composite.
    "haze.frag",
}

EXPECTED_INCLUDES = {"air_extinction.glsl"}


def test_every_expected_shader_ships() -> None:
    assert set(shaders.available()) == EXPECTED


def test_includes_are_not_offered_as_compilable_stages() -> None:
    # A .glsl fragment has no #version and no main; handing one to the driver
    # as a stage would fail, so it must not appear in available().
    assert set(shaders.includes()) == EXPECTED_INCLUDES
    assert not EXPECTED_INCLUDES & set(shaders.available())


def test_air_extinction_reaches_every_stage_that_crosses_air() -> None:
    # Five stages attenuate light over a path. Before the include existed the
    # only way to keep them agreeing was to read all five, and a drifted copy
    # would have been invisible in the output.
    for name in (
        "haze.frag",
        "particle.vert",
        "smoke.frag",
        "scene.frag",
        "water.frag",
    ):
        resolved = shaders.source(name)
        assert "vec3 air_transmittance(" in resolved, name
        assert "uniform vec3 aerosol_extinction_per_m;" in resolved, name
        # The include is resolved before the driver sees the text.
        assert "#include" not in resolved, name


def test_the_retired_extinction_uniform_is_gone() -> None:
    # air_extinction_per_m was a single unsourced scalar implying 32.6 km of
    # visibility, applied only between a lamp and a surface. The name survives
    # in a comment recording that; what must not survive is the declaration.
    for name in shaders.available():
        code = re.sub(r"//[^\n]*", "", shaders.source(name))
        assert "air_extinction_per_m" not in code, name


def test_a_circular_include_is_reported_rather_than_recursing() -> None:
    with pytest.raises(shaders.ShaderIncludeError, match="circular"):
        shaders._resolve("a.glsl", ("b.glsl", "a.glsl"))


def test_every_shader_declares_a_version_and_a_main() -> None:
    for name in shaders.available():
        text = shaders.source(name)
        assert text.lstrip().startswith("#version"), name
        assert re.search(r"\bvoid\s+main\s*\(", text), name


def test_sources_are_cached() -> None:
    # The loader is on the program-creation path; re-reading files there would
    # put filesystem latency into GPU resource creation.
    assert shaders.source("quad.vert") is shaders.source("quad.vert")


def test_a_missing_shader_names_what_is_available() -> None:
    with pytest.raises(FileNotFoundError, match="not found") as error:
        shaders.source("no_such.frag")
    assert "quad.vert" in str(error.value)


def test_no_shader_source_remains_embedded_in_python() -> None:
    # The renderer previously carried 904 lines of GLSL as string constants.
    from pathlib import Path

    renderer = (
        Path(shaders.SHADER_DIRECTORY).parent / "renderer.py"
    ).read_text(encoding="utf-8")
    assert "#version" not in renderer
    assert "gl_Position" not in renderer


def test_uniform_names_are_unique_within_each_stage() -> None:
    # A duplicated uniform declaration compiles on some drivers and fails on
    # others, so it is worth catching before a GPU sees it.
    for name in shaders.available():
        declared = re.findall(
            r"^uniform\s+\w+\s+(\w+)", shaders.source(name), re.MULTILINE
        )
        inline = re.findall(
            r"uniform\s+\w+\s+(\w+)\s*;", shaders.source(name)
        )
        names = declared + inline
        assert len(names) == len(set(names)) or len(set(names)) == len(
            set(declared) | set(inline)
        ), name
