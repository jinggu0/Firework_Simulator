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
import re

import moderngl

SHADER_DIRECTORY = Path(__file__).resolve().parent

STAGE_SUFFIXES = (".vert", ".frag", ".geom")
"""Suffixes that name a compilable stage. ``.glsl`` files are includes only."""

INCLUDE_PATTERN = re.compile(
    r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.MULTILINE
)
"""``#include "name.glsl"`` on a line of its own.

GLSL has no include directive, so five shaders that need the same atmospheric
extinction would otherwise carry five copies of it — and a copy that drifts is
exactly the failure this project cannot detect by looking at output. Resolution
happens before the driver sees the text.
"""


class ShaderCompilationError(RuntimeError):
    """A GLSL program failed to build, with the stage files identified."""


class ShaderIncludeError(RuntimeError):
    """An ``#include`` could not be resolved, naming the cycle or the file."""


def _read(name: str) -> str:
    path = SHADER_DIRECTORY / name
    if not path.is_file():
        available = sorted(
            entry.name
            for entry in SHADER_DIRECTORY.iterdir()
            if entry.suffix in STAGE_SUFFIXES + (".glsl",)
        )
        raise FileNotFoundError(
            f"shader {name!r} not found in {SHADER_DIRECTORY}; have {available}"
        )
    return path.read_text(encoding="utf-8")


def _resolve(name: str, stack: tuple[str, ...]) -> str:
    if name in stack:
        chain = " -> ".join(stack + (name,))
        raise ShaderIncludeError(f"circular shader include: {chain}")
    return INCLUDE_PATTERN.sub(
        lambda match: _resolve(match.group(1), stack + (name,)), _read(name)
    )


@lru_cache(maxsize=None)
def source(name: str) -> str:
    """Return the GLSL source for ``name``, with includes resolved."""

    return _resolve(name, ())


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
    """Every compilable stage this package ships, for diagnostics and tests."""

    return sorted(
        entry.name
        for entry in SHADER_DIRECTORY.iterdir()
        if entry.suffix in STAGE_SUFFIXES
    )


def includes() -> list[str]:
    """Every ``.glsl`` fragment shared between stages."""

    return sorted(
        entry.name
        for entry in SHADER_DIRECTORY.iterdir()
        if entry.suffix == ".glsl"
    )
