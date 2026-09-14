"""Evaluate a shader change without editing the shipped shader.

A candidate fix for an appearance defect is a list of exact text replacements
in named GLSL stages. `shader_variant` applies them in memory, so programs
compiled inside the context see the candidate while the file on disk — which
several validation reports lock by checksum — stays the shipped one until a
candidate has earned the edit.

A candidate that no longer matches the shipped source fails before anything
renders instead of silently measuring something else. Replacement is exact and
single, the same fail-closed contract as the facade reduction probes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator

from simulator import shaders


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DIRECTORY = (
    REPOSITORY_ROOT / "docs" / "validation" / "shimmer_ab_v2" / "candidates"
)


@dataclass(frozen=True, slots=True)
class ShaderReplacement:
    """Replace the single occurrence of `old` in stage `file` with `new`."""

    file: str
    old: str
    new: str


@dataclass(frozen=True, slots=True)
class ShaderCandidate:
    candidate_id: str
    hypothesis: str
    provenance: dict[str, Any]
    replacements: tuple[ShaderReplacement, ...]


def apply_replacements(
    source: str, replacements: list[ShaderReplacement] | tuple[ShaderReplacement, ...]
) -> str:
    """Apply replacements in order; each must match exactly once when applied."""

    text = source
    for replacement in replacements:
        count = text.count(replacement.old)
        if count != 1:
            raise ValueError(
                f"replacement in {replacement.file} must match exactly once, "
                f"matched {count} times: {replacement.old[:60]!r}"
            )
        text = text.replace(replacement.old, replacement.new, 1)
    return text


def load_candidate(path: Path) -> ShaderCandidate:
    """Read a candidate file, refusing one that does not say why it exists."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("shader candidate schema_version must be 1")
    for field in ("candidate_id", "hypothesis"):
        if not str(payload.get(field, "")).strip():
            raise ValueError(f"shader candidate requires a {field}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("shader candidate requires provenance")
    items = payload.get("replacements")
    if not isinstance(items, list) or not items:
        raise ValueError("shader candidate requires replacements")
    replacements = []
    for item in items:
        if not all(isinstance(item.get(key), str) for key in ("file", "old", "new")):
            raise ValueError("each replacement needs string file, old and new")
        replacements.append(ShaderReplacement(item["file"], item["old"], item["new"]))
    return ShaderCandidate(
        candidate_id=str(payload["candidate_id"]),
        hypothesis=str(payload["hypothesis"]),
        provenance=provenance,
        replacements=tuple(replacements),
    )


@contextmanager
def shader_variant(
    replacements: list[ShaderReplacement] | tuple[ShaderReplacement, ...],
) -> Iterator[dict[str, str]]:
    """Serve patched sources to programs built inside the context.

    Yields the SHA-256 of each patched stage, so a measurement can record
    exactly which source it rendered. Include files are not patchable: stages
    resolve them without going through `shaders.source`.
    """

    known = set(shaders.available())
    by_file: dict[str, list[ShaderReplacement]] = {}
    for replacement in replacements:
        if replacement.file not in known:
            raise ValueError(f"unknown shader stage {replacement.file!r}")
        by_file.setdefault(replacement.file, []).append(replacement)
    patched = {
        name: apply_replacements(shaders.source(name), items)
        for name, items in by_file.items()
    }
    original = shaders.source

    def source(name: str) -> str:
        return patched[name] if name in patched else original(name)

    shaders.source = source
    try:
        yield {
            name: sha256(text.encode("utf-8")).hexdigest()
            for name, text in patched.items()
        }
    finally:
        shaders.source = original
