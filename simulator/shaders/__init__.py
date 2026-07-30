"""GLSL sources, loaded from files rather than embedded in Python.

``simulator/renderer.py`` previously carried 904 lines of shader source as
module-level string constants — half the file. Moving them out makes the GLSL
editable with syntax highlighting, keeps the Python file about the pass graph,
and lets a compilation failure name the file it came from.

Sources are read once and cached, so repeated program creation does not touch
the filesystem.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import moderngl

SHADER_DIRECTORY = Path(__file__).resolve().parent


class ShaderCompilationError(RuntimeError):
    """A GLSL program failed to build, with the stage files identified."""


@lru_cache(maxsize=None)
def source(name: str) -> str:
    """Return the GLSL source for ``name``, e.g. ``"water.frag"``."""

    path = SHADER_DIRECTORY / name
    if not path.is_file():
        available = sorted(
            entry.name
            for entry in SHADER_DIRECTORY.iterdir()
            if entry.suffix in (".vert", ".frag", ".geom")
        )
        raise FileNotFoundError(
            f"shader {name!r} not found in {SHADER_DIRECTORY}; have {available}"
        )
    return path.read_text(encoding="utf-8")


def program(
    ctx: moderngl.Context,
    vertex: str,
    fragment: str,
    geometry: str | None = None,
) -> moderngl.Program:
    """Build a program from shader file names.

    Wraps the driver's compile error with the file names involved. A raw
    moderngl error reports a line number against an anonymous string, which is
    not enough to find the fault when sixteen shaders are in play.
    """

    stages = {
        "vertex_shader": source(vertex),
        "fragment_shader": source(fragment),
    }
    if geometry is not None:
        stages["geometry_shader"] = source(geometry)
    try:
        return ctx.program(**stages)
    except moderngl.Error as error:
        involved = ", ".join(
            name for name in (vertex, geometry, fragment) if name
        )
        raise ShaderCompilationError(
            f"failed to build program from [{involved}]: {error}"
        ) from error


def available() -> list[str]:
    """Every shader file this package ships, for diagnostics and tests."""

    return sorted(
        entry.name
        for entry in SHADER_DIRECTORY.iterdir()
        if entry.suffix in (".vert", ".frag", ".geom")
    )
