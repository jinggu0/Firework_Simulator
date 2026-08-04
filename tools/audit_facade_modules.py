"""Audit the facade module dimensions the renderer actually uses.

Every building wall in the scene is lit by a window grid whose bay width and
floor height are metre quantities written directly into `scene.frag`. The
landmark heights those grids sit inside are sourced — ATTRIBUTION.md cites the
architects and owners — but the grid spacings themselves are not recorded
anywhere. They are bounded appearance reconstructions, and nothing in the
repository said so per dimension or let anyone check them.

This reads the three places the facade description lives — the GLSL branches,
the Python style constants, and the built scene asset — and requires them to
agree. Two failures are worth catching. A style declared in Python but not
handled in GLSL silently falls through to the generic 4.2 m x 3.25 m office
grid, which would paint window cells on something like a stone colonnade. And a
dimension edited in the shader without updating its evidence record would
quietly restate an ungraded claim as though it had been checked.

The audit changes no geometry. It reports what is claimed and what is still
unsourced.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHADER = Path("simulator/shaders/scene.frag")
DEFAULT_SCENE_MODULE = Path("simulator/scene.py")
DEFAULT_SCENE_ASSET = Path("assets/yeouido_scene.npz")
DEFAULT_EVIDENCE = Path("assets/yeouido_facade_module_evidence.json")
DEFAULT_ATTRIBUTION = Path("assets/ATTRIBUTION.md")
DEFAULT_OUTPUT = Path(
    "docs/validation/facade_modules_v3/facade_module_report.json"
)

SURFACE_ROOF = 1.0
#: Grades used across this project: A official observation, B reconstructed
#: from published record, C accepted physical or statistical model, D artistic
#: supplement, U unverified.
EVIDENCE_GRADES = {"A", "B", "C", "D", "U"}
#: Grades that may not be described as surveyed facade measurements.
UNSURVEYED_GRADES = {"C", "D", "U"}

_STYLE_CONSTANT = re.compile(r"^FACADE_([A-Z0-9_]+)\s*=\s*([0-9.]+)\s*$", re.M)
_BRANCH = re.compile(
    r"facade_style\s*>\s*([0-9.]+)\s*&&\s*facade_style\s*<\s*([0-9.]+)"
)
_DECLARED = re.compile(r"float\s+(bay_width|floor_height)\s*=\s*([0-9.]+)\s*;")
_ASSIGNED = re.compile(r"\b(bay_width|floor_height)\s*=\s*([0-9.]+)\s*;")
_LANDMARK_HEIGHT = re.compile(r'"([a-z0-9_]+)"\s*:\s*([0-9.]+)\s*,')
_LEVELS_TO_HEIGHT = re.compile(r"levels\s*\*\s*([0-9.]+)")
_MIN_LEVEL_TO_HEIGHT = re.compile(
    r'building:min_level[^)]*\)\s*\*\s*([0-9.]+)'
)
_DEFAULT_HEIGHT = re.compile(r"return\s+([0-9.]+)\s*$", re.M)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _style_of(lower: str, upper: str) -> float:
    # Branches are written as half-open guards around an integer style, so the
    # midpoint recovers the style the branch selects.
    return round((float(lower) + float(upper)) / 2.0, 3)


def declared_styles(scene_module: str) -> dict[float, str]:
    """FACADE_* constants, as value to name."""

    styles: dict[float, str] = {}
    for name, value in _STYLE_CONSTANT.findall(scene_module):
        styles[round(float(value), 3)] = f"FACADE_{name}"
    return styles


def shader_styles(shader: str) -> set[float]:
    """Every style the shader branches on anywhere."""

    return {_style_of(low, high) for low, high in _BRANCH.findall(shader)}


def _branch_body(shader: str, offset: int) -> str:
    """The braced body of the branch whose guard starts at `offset`."""

    opening = shader.find("{", offset)
    if opening < 0:
        return ""
    depth = 0
    for index in range(opening, len(shader)):
        if shader[index] == "{":
            depth += 1
        elif shader[index] == "}":
            depth -= 1
            if depth == 0:
                return shader[opening : index + 1]
    return shader[opening:]


def opaque_styles(shader: str) -> set[float]:
    """Styles the shader finishes before the window grid is ever reached.

    Membership requires an actual early return ahead of the grid, not merely a
    branch somewhere. A style can be mentioned later — to tint its roof, say —
    while still having no grid of its own, and that style silently inherits the
    generic office spacing. Treating any mention as deliberate would hide
    exactly the case this audit exists to catch.
    """

    ladder = shader.find("float bay_width")
    if ladder < 0:
        return set()
    styles: set[float] = set()
    for match in _BRANCH.finditer(shader):
        if match.start() >= ladder:
            continue
        if "return" in _branch_body(shader, match.start()):
            styles.add(_style_of(match.group(1), match.group(2)))
    return styles


def module_dimensions(shader: str) -> dict[float, dict[str, float]]:
    """Bay width and floor height per style, from the window-grid chain.

    The chain declares the generic office grid first and then overrides it in
    an else-if ladder, so the declarations are style 0 and each branch body
    supplies the rest.
    """

    start = shader.find("float bay_width")
    if start < 0:
        return {}
    # The ladder ends where the per-style overrides stop; the FKI comment that
    # follows is the first thing after it that mentions the style again.
    end = shader.find("float fki_panel", start)
    region = shader[start : end if end > start else len(shader)]

    generic = {
        name: float(value) for name, value in _DECLARED.findall(region)
    }
    dimensions: dict[float, dict[str, float]] = {}
    if generic:
        dimensions[0.0] = generic

    # Split the ladder on its guards so each body is attributed to its style.
    positions = [
        (match.start(), _style_of(match.group(1), match.group(2)))
        for match in _BRANCH.finditer(region)
    ]
    for index, (offset, style) in enumerate(positions):
        stop = positions[index + 1][0] if index + 1 < len(positions) else len(region)
        body = region[offset:stop]
        values = {name: float(value) for name, value in _ASSIGNED.findall(body)}
        if values:
            dimensions[style] = values
    return dimensions


def landmark_heights(scene_module: str) -> dict[str, float]:
    start = scene_module.find("def _landmark_height")
    if start < 0:
        return {}
    end = scene_module.find("def ", start + 1)
    region = scene_module[start:end if end > start else len(scene_module)]
    return {
        name: float(value) for name, value in _LANDMARK_HEIGHT.findall(region)
    }


def import_storey_heights(scene_module: str) -> dict[str, float | None]:
    """The storey heights the importer uses to turn tags into geometry.

    A building with no `height` tag gets its mass from `building:levels` times
    an assumed storey height. That assumption sets real vertex positions, so it
    matters more than the window grid painted over them — and it is a different
    number from every one of those grids.
    """

    def region(name: str) -> str:
        start = scene_module.find(f"def {name}")
        if start < 0:
            return ""
        end = scene_module.find("\ndef ", start + 1)
        return scene_module[start : end if end > start else len(scene_module)]

    height = region("_height")
    minimum = region("_minimum_height")
    levels = _LEVELS_TO_HEIGHT.search(height)
    min_levels = _MIN_LEVEL_TO_HEIGHT.search(minimum)
    defaults = _DEFAULT_HEIGHT.findall(height)
    return {
        "levels_to_height_m": float(levels.group(1)) if levels else None,
        "min_level_to_height_m": float(min_levels.group(1)) if min_levels else None,
        "untagged_default_height_m": float(defaults[-1]) if defaults else None,
    }


def scene_style_usage(asset: Path) -> dict[float, dict[str, int]]:
    """Wall and roof vertex counts per style in the built asset."""

    with np.load(asset) as archive:
        vertices = archive["building_vertices"]
    styles = np.round(vertices[:, 9].astype(float), 3)
    surfaces = np.round(vertices[:, 6].astype(float), 3)
    usage: dict[float, dict[str, int]] = {}
    for style in sorted(set(styles.tolist())):
        selected = styles == style
        roof = int(np.count_nonzero(selected & (surfaces >= SURFACE_ROOF - 0.5)))
        usage[style] = {
            "vertex_count": int(np.count_nonzero(selected)),
            "roof_vertex_count": roof,
            "wall_vertex_count": int(np.count_nonzero(selected)) - roof,
        }
    return usage


def build_report(
    shader_path: Path = DEFAULT_SHADER,
    scene_module_path: Path = DEFAULT_SCENE_MODULE,
    scene_asset_path: Path = DEFAULT_SCENE_ASSET,
    evidence_path: Path = DEFAULT_EVIDENCE,
    attribution_path: Path = DEFAULT_ATTRIBUTION,
) -> dict[str, Any]:
    shader = shader_path.read_text(encoding="utf-8")
    scene_module = scene_module_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    attribution = attribution_path.read_text(encoding="utf-8")

    declared = declared_styles(scene_module)
    branched = shader_styles(shader)
    dimensions = module_dimensions(shader)
    usage = scene_style_usage(scene_asset_path)
    heights = landmark_heights(scene_module)
    records = {
        round(float(record["facade_style"]), 3): record
        for record in evidence["facade_styles"]
    }

    used_styles = sorted(usage)
    undeclared = sorted(style for style in used_styles if style not in declared)
    # Style 0 is the generic office grid: it is the ladder's default rather than
    # a branch, so it is handled without appearing as a guard.
    unhandled = sorted(
        style
        for style in used_styles
        if style != 0.0 and style not in branched and style not in dimensions
    )

    # A style with wall vertices is painted by the window grid, so it needs its
    # own spacing or an early return, or it silently inherits the generic
    # office grid.
    finished_early = opaque_styles(shader)
    windowed = [
        style for style in used_styles if usage[style]["wall_vertex_count"] > 0
    ]
    opaque = sorted(style for style in windowed if style in finished_early)
    missing_dimensions = sorted(
        style
        for style in windowed
        if style not in dimensions and style not in finished_early
    )

    unrecorded = sorted(style for style in dimensions if style not in records)
    drifted: list[dict[str, Any]] = []
    for style, values in sorted(dimensions.items()):
        record = records.get(style)
        if record is None:
            continue
        for key, shader_value in values.items():
            recorded = record.get(f"{key}_m")
            if recorded is None or abs(float(recorded) - shader_value) > 1e-9:
                drifted.append(
                    {
                        "facade_style": style,
                        "dimension": f"{key}_m",
                        "shader_value": shader_value,
                        "recorded_value": recorded,
                    }
                )

    bad_grades = sorted(
        {
            str(record.get("evidence_grade"))
            for record in records.values()
            if record.get("evidence_grade") not in EVIDENCE_GRADES
        }
    )
    cited_urls = [
        record["height_source_url"]
        for record in records.values()
        if record.get("height_source_url")
    ]
    uncited = sorted(
        {url for url in cited_urls if url not in attribution}
    )

    # What the shader's grid implies, given heights that *are* sourced. This is
    # arithmetic on held values, not a claim about the real buildings; the
    # published floor counts needed to check it are on the missing-data list.
    implied: list[dict[str, Any]] = []
    for record in evidence["facade_styles"]:
        style = round(float(record["facade_style"]), 3)
        floor_height = dimensions.get(style, {}).get("floor_height")
        for landmark in record.get("landmarks", []):
            height = heights.get(landmark.get("landmark_key", ""))
            if height is None:
                height = landmark.get("height_m")
            if height is None or not floor_height:
                continue
            implied.append(
                {
                    "landmark_key": landmark.get("landmark_key"),
                    "height_m": float(height),
                    "height_source_url": record.get("height_source_url"),
                    "floor_height_m": floor_height,
                    "implied_floor_count": round(float(height) / floor_height, 2),
                    "published_floor_count_confirmed": False,
                }
            )

    unsurveyed = sorted(
        {
            f"{record['name']} ({record.get('evidence_grade')})"
            for record in records.values()
            if record.get("evidence_grade") in UNSURVEYED_GRADES
        }
    )

    # The importer derives untagged building heights from `building:levels`
    # times one assumed storey height, then the shader paints storey bands at a
    # different one. For those buildings the rendered band count contradicts the
    # only floor-count evidence the importer had.
    storey = import_storey_heights(scene_module)
    import_height = storey["levels_to_height_m"]
    storey_rows: list[dict[str, Any]] = []
    if import_height:
        for style, values in sorted(dimensions.items()):
            painted = values.get("floor_height")
            if not painted:
                continue
            storey_rows.append(
                {
                    "facade_style": style,
                    "name": records.get(style, {}).get("name"),
                    "import_storey_height_m": import_height,
                    "painted_floor_height_m": painted,
                    "painted_bands_per_source_storey": round(
                        import_height / painted, 4
                    ),
                    "relative_error": round(
                        (import_height - painted) / painted, 4
                    ),
                }
            )
    worst = (
        max(storey_rows, key=lambda row: abs(row["relative_error"]))
        if storey_rows
        else None
    )
    storey_heights_agree = bool(
        storey_rows and all(row["relative_error"] == 0.0 for row in storey_rows)
    )

    styles_consistent = not undeclared and not unhandled and not missing_dimensions
    evidence_consistent = not unrecorded and not drifted and not bad_grades
    citations_verified = not uncited

    blockers: list[str] = []
    if undeclared:
        blockers.append(
            "the scene uses facade styles with no FACADE_* constant: "
            + ", ".join(str(style) for style in undeclared)
        )
    if unhandled:
        blockers.append(
            "the shader has no branch for scene styles: "
            + ", ".join(str(style) for style in unhandled)
        )
    if missing_dimensions:
        blockers.append(
            "these styles have wall vertices but neither their own grid nor an "
            "opaque branch, so they inherit the generic office grid: "
            + ", ".join(str(style) for style in missing_dimensions)
        )
    if unrecorded:
        blockers.append(
            "the shader defines module dimensions with no evidence record: "
            + ", ".join(str(style) for style in unrecorded)
        )
    if drifted:
        blockers.append(
            f"{len(drifted)} shader dimensions differ from their evidence record"
        )
    if bad_grades:
        blockers.append("unknown evidence grades: " + ", ".join(bad_grades))
    if uncited:
        blockers.append(
            "evidence cites sources that ATTRIBUTION.md does not record: "
            + ", ".join(uncited)
        )
    if unsurveyed:
        blockers.append(
            f"{len(unsurveyed)} facade families carry unsurveyed module "
            "dimensions, so no surveyed-fidelity claim may be made"
        )
    if worst and not storey_heights_agree:
        blockers.append(
            "the importer derives heights at "
            f"{worst['import_storey_height_m']} m per storey while the shader "
            f"paints {worst['name']} at {worst['painted_floor_height_m']} m, a "
            f"{abs(worst['relative_error']) * 100:.1f}% disagreement, so "
            "rendered storey bands contradict the source floor count"
        )

    return {
        "schema_version": 1,
        "stage": "V3-1a",
        "inputs": {
            "shader": _display_path(shader_path),
            "shader_sha256": _digest(shader_path),
            "scene_module": _display_path(scene_module_path),
            "scene_module_sha256": _digest(scene_module_path),
            "scene_asset": _display_path(scene_asset_path),
            "scene_asset_sha256": _digest(scene_asset_path),
            "evidence": _display_path(evidence_path),
            "evidence_sha256": _digest(evidence_path),
            "attribution": _display_path(attribution_path),
            "attribution_sha256": _digest(attribution_path),
        },
        "coverage": {
            "declared_style_count": len(declared),
            "scene_style_count": len(used_styles),
            "scene_styles": used_styles,
            "undeclared_scene_styles": undeclared,
            "styles_without_a_shader_branch": unhandled,
            "styles_inheriting_the_generic_grid": missing_dimensions,
            "opaque_styles": opaque,
            "windowed_style_count": len(dimensions),
            "styles_consistent": styles_consistent,
        },
        "module_dimensions": [
            {
                "facade_style": style,
                "name": records.get(style, {}).get("name"),
                "bay_width_m": values.get("bay_width"),
                "floor_height_m": values.get("floor_height"),
                "evidence_grade": records.get(style, {}).get("evidence_grade"),
                "wall_vertex_count": usage.get(style, {}).get(
                    "wall_vertex_count", 0
                ),
            }
            for style, values in sorted(dimensions.items())
        ],
        "style_usage": {str(style): counts for style, counts in usage.items()},
        "import_storey_heights": storey,
        "storey_height_consistency": {
            "storey_heights_agree": storey_heights_agree,
            "worst_family": worst,
            "rows": storey_rows,
        },
        "implied_floor_counts": implied,
        "checks": {
            "styles_consistent": styles_consistent,
            "evidence_consistent": evidence_consistent,
            "citations_verified": citations_verified,
            "unrecorded_dimension_styles": unrecorded,
            "drifted_dimensions": drifted,
            "unknown_evidence_grades": bad_grades,
            "uncited_sources": uncited,
            "unsurveyed_families": unsurveyed,
            "storey_heights_agree": storey_heights_agree,
        },
        "application_gates": {
            "module_dimensions_traceable": evidence_consistent
            and styles_consistent,
            "surveyed_facade_module_claim_allowed": not unsurveyed
            and evidence_consistent
            and styles_consistent,
            "photo_registration_available": bool(
                evidence["photo_registration"]["registerable_viewpoint_count"]
            ),
            "runtime_geometry_changed_by_this_stage": False,
            "scene_vertices_modified": 0,
        },
        "missing_data": evidence["missing_data"],
        "blocking_reasons": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader", type=Path, default=DEFAULT_SHADER)
    parser.add_argument("--scene-module", type=Path, default=DEFAULT_SCENE_MODULE)
    parser.add_argument("--scene-asset", type=Path, default=DEFAULT_SCENE_ASSET)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--attribution", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(
        arguments.shader,
        arguments.scene_module,
        arguments.scene_asset,
        arguments.evidence,
        arguments.attribution,
    )
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: styles_consistent="
        f"{report['coverage']['styles_consistent']}, "
        f"surveyed_facade_module_claim_allowed="
        f"{report['application_gates']['surveyed_facade_module_claim_allowed']}"
    )


if __name__ == "__main__":
    main()
