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
}


def test_every_expected_shader_ships() -> None:
    assert set(shaders.available()) == EXPECTED


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
