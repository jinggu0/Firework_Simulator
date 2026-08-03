"""Audit the scale and evidence status of rendered road-surface details."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from ..materials import MATERIAL_LIBRARY
from ..passes.scene import (
    KERB_MAX_ORIGIN_DISTANCE_M,
    KERB_MAX_ROAD_WIDTH_M,
    KERB_MIN_ROAD_WIDTH_M,
    KERB_REVEAL_HEIGHT_M,
    KERB_TOP_WIDTH_M,
    linear_feature_uv,
    road_edge_detail_vertices,
)
from ..road_semantics import (
    DEFAULT_ROAD_SEMANTICS_PATH,
    filter_occluded_road_segments,
    load_road_structure_semantics,
)
from ..scene import (
    LINEAR_STYLE_STEPS,
    SURFACE_CYCLEWAY,
    SURFACE_FOOTWAY,
    SURFACE_ROAD,
    SURFACE_TRAIL,
    load_scene,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
DEFAULT_DETAIL_OSM_PATH = (
    REPOSITORY_ROOT / "assets" / "yeouido_detail_osm_2024-10-05.json"
)
DEFAULT_SHADER_PATH = REPOSITORY_ROOT / "simulator" / "shaders" / "scene.frag"
DEFAULT_SCENE_PASS_PATH = REPOSITORY_ROOT / "simulator" / "passes" / "scene.py"
SURFACE_LABELS = {
    int(SURFACE_ROAD): "asphalt_road",
    int(SURFACE_FOOTWAY): "concrete_footway",
    int(SURFACE_CYCLEWAY): "cycleway",
    int(SURFACE_TRAIL): "compacted_trail",
}
ROAD_SHADER_CONSTANTS = (
    "ROAD_EDGE_LINE_V_INNER",
    "ROAD_EDGE_LINE_V_OUTER",
    "ROAD_CENTRE_LINE_V_CORE",
    "ROAD_CENTRE_LINE_V_SUPPORT",
    "ROAD_DASH_PERIOD_M",
    "ROAD_DASH_START_PHASE",
    "ROAD_DASH_CORE_START_PHASE",
    "ROAD_DASH_CORE_END_PHASE",
    "ROAD_DASH_END_PHASE",
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _shader_float_constants(path: Path = DEFAULT_SHADER_PATH) -> dict[str, float]:
    source = path.read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for name in ROAD_SHADER_CONSTANTS:
        match = re.search(
            rf"const\s+float\s+{name}\s*=\s*([0-9.]+)\s*;",
            source,
        )
        if match is None:
            raise ValueError(f"scene shader does not declare {name}")
        values[name] = float(match.group(1))
    return values


def road_quad_measurements(
    vertices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Measure centreline length and authored width for six-vertex quads.

    Miter joins lengthen the stored edge vector at bends. Projecting both end
    spans onto the segment normal recovers the authored road width rather than
    incorrectly treating that miter extension as a wider carriageway.
    """

    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (10,) or len(values) % 6:
        raise ValueError("road vertices must have shape (6*n, 10)")
    quads = values.reshape(-1, 6, 10)
    centre_start = 0.5 * (quads[:, 0, [0, 2]] + quads[:, 1, [0, 2]])
    centre_end = 0.5 * (quads[:, 2, [0, 2]] + quads[:, 5, [0, 2]])
    direction = centre_end - centre_start
    length_m = np.linalg.norm(direction, axis=1)
    unit = direction / np.maximum(length_m[:, None], 1e-12)
    normal = np.stack((-unit[:, 1], unit[:, 0]), axis=1)
    start_span = quads[:, 1, [0, 2]] - quads[:, 0, [0, 2]]
    end_span = quads[:, 2, [0, 2]] - quads[:, 5, [0, 2]]
    width_m = 0.5 * (
        np.abs(np.sum(start_span * normal, axis=1))
        + np.abs(np.sum(end_span * normal, axis=1))
    )
    return {
        "quads": quads,
        "centre_start_xz_m": centre_start,
        "centre_end_xz_m": centre_end,
        "centre_midpoint_xz_m": 0.5 * (centre_start + centre_end),
        "length_m": length_m,
        "width_m": width_m,
        "surface_code": quads[:, 0, 6].astype(np.int32),
        "linear_style": quads[:, 0, 9],
    }


def rendered_road_measurements(
    scene_path: Path = DEFAULT_SCENE_PATH,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    scene = load_scene(scene_path)
    visible, filter_stats = filter_occluded_road_segments(
        scene.road_vertices,
        load_road_structure_semantics(),
    )
    mapped = linear_feature_uv(visible)
    return road_quad_measurements(mapped), asdict(filter_stats), scene.snapshot_utc


def _distribution(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"minimum": None, "p50": None, "p95": None, "maximum": None}
    return {
        "minimum": float(np.min(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "maximum": float(np.max(values)),
    }


def _rounded_counts(values: np.ndarray) -> list[dict[str, Any]]:
    rounded = np.round(values, 3)
    unique, counts = np.unique(rounded, return_counts=True)
    return [
        {"width_m": float(width), "segment_count": int(count)}
        for width, count in zip(unique, counts)
    ]


def _detail_snapshot_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    elements = payload.get("elements", [])
    highway = [item for item in elements if item.get("tags", {}).get("highway")]
    manholes = [
        item for item in elements
        if item.get("tags", {}).get("man_made") == "manhole"
        or "manhole" in item.get("tags", {})
    ]
    bicycle_symbols = [
        item for item in elements
        if item.get("tags", {}).get("road_marking") in {
            "bicycle", "cycleway", "bike"
        }
    ]
    return {
        "element_count": len(elements),
        "highway_element_count": len(highway),
        "manhole_element_count": len(manholes),
        "bicycle_marking_element_count": len(bicycle_symbols),
        "scope_note": (
            "This focused detail snapshot contains land-cover and facility "
            "ways, not the broad road-source elements used to build the NPZ."
        ),
    }


def road_detail_report(
    scene_path: Path = DEFAULT_SCENE_PATH,
    shader_path: Path = DEFAULT_SHADER_PATH,
    detail_osm_path: Path = DEFAULT_DETAIL_OSM_PATH,
    semantics_path: Path = DEFAULT_ROAD_SEMANTICS_PATH,
) -> dict[str, Any]:
    measurements, filter_stats, snapshot_utc = rendered_road_measurements(
        scene_path
    )
    constants = _shader_float_constants(shader_path)
    surfaces = []
    for code, label in SURFACE_LABELS.items():
        selected = measurements["surface_code"] == code
        widths = measurements["width_m"][selected]
        lengths = measurements["length_m"][selected]
        surfaces.append(
            {
                "surface_code": code,
                "surface": label,
                "segment_count": int(np.count_nonzero(selected)),
                "centreline_length_m": float(np.sum(lengths)),
                "authored_width_m": _distribution(widths),
                "rounded_authored_width_counts": _rounded_counts(widths),
            }
        )

    asphalt = measurements["surface_code"] == int(SURFACE_ROAD)
    asphalt_widths = measurements["width_m"][asphalt]
    paint_width_factor = 0.5 * (
        constants["ROAD_EDGE_LINE_V_OUTER"]
        - constants["ROAD_EDGE_LINE_V_INNER"]
    )
    centre_support_factor = constants["ROAD_CENTRE_LINE_V_SUPPORT"]
    edge_widths = asphalt_widths * paint_width_factor
    centre_widths = asphalt_widths * centre_support_factor

    mapped = measurements["quads"].reshape(-1, 10)
    ordinary = mapped[
        ~np.isclose(mapped[:, 9], LINEAR_STYLE_STEPS)
    ]
    kerb_vertices = road_edge_detail_vertices(ordinary)
    kerb_segment_count = len(kerb_vertices) // 24

    cycleway = MATERIAL_LIBRARY.get(SURFACE_CYCLEWAY)
    cycle_frequency = cycleway.pattern_scale[0]
    cycle_period_m = 1.0 / cycle_frequency
    cycle_core_width_m = cycle_period_m * 2.0 * 0.025
    cycle_support_width_m = cycle_period_m * 2.0 * 0.055

    return {
        "schema_version": 1,
        "stage": "V2-2a",
        "scene": {
            "asset": scene_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _digest(scene_path),
            "snapshot_utc": snapshot_utc,
            "coordinate_system": "local East-Up-South metres",
        },
        "implementation_sources": {
            "road_shader": {
                "asset": shader_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _digest(shader_path),
            },
            "kerb_mesh": {
                "asset": DEFAULT_SCENE_PASS_PATH.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "sha256": _digest(DEFAULT_SCENE_PASS_PATH),
            },
        },
        "runtime_road_inventory": {
            "visible_segment_count": int(len(measurements["length_m"])),
            "visible_centreline_length_m": float(
                np.sum(measurements["length_m"])
            ),
            "semantic_filter": filter_stats,
            "surfaces": surfaces,
        },
        "kerb_contract": {
            "implementation": "derived mesh on selected asphalt edges",
            "confidence_grade": "D",
            "surveyed_at_yeouido": False,
            "reveal_height_m": KERB_REVEAL_HEIGHT_M,
            "top_width_m": KERB_TOP_WIDTH_M,
            "eligible_road_width_m": [
                KERB_MIN_ROAD_WIDTH_M,
                KERB_MAX_ROAD_WIDTH_M,
            ],
            "maximum_origin_distance_m": KERB_MAX_ORIGIN_DISTANCE_M,
            "generated_source_segment_count": kerb_segment_count,
            "generated_triangle_count": len(kerb_vertices) // 3,
            "site_position_evidence": False,
        },
        "asphalt_marking_contract": {
            "implementation": "procedural shader on every asphalt surface",
            "confidence_grade": "D",
            "road_specific_lane_semantics_available": False,
            "shader_constants": constants,
            "edge_line_support_width_m": _distribution(edge_widths),
            "centre_line_support_width_m": _distribution(centre_widths),
            "segments_over_0_20_m_line_width": int(
                np.count_nonzero(centre_widths > 0.20)
            ),
            "segments_over_0_30_m_line_width": int(
                np.count_nonzero(centre_widths > 0.30)
            ),
            "dash_support_length_m": (
                constants["ROAD_DASH_END_PHASE"]
                - constants["ROAD_DASH_START_PHASE"]
            ) * constants["ROAD_DASH_PERIOD_M"],
            "dash_core_length_m": (
                constants["ROAD_DASH_CORE_END_PHASE"]
                - constants["ROAD_DASH_CORE_START_PHASE"]
            ) * constants["ROAD_DASH_PERIOD_M"],
            "finding": (
                "Paint width scales with normalized road width, so identical "
                "shader constants produce different physical widths."
            ),
        },
        "cycleway_marking_contract": {
            "implementation": "red surface with periodic transverse stripe",
            "confidence_grade": "D",
            "stripe_period_m": cycle_period_m,
            "stripe_core_width_m": cycle_core_width_m,
            "stripe_support_width_m": cycle_support_width_m,
            "bicycle_symbol_geometry_present": False,
            "site_specific_marking_evidence": False,
        },
        "missing_detail_inventory": {
            "manhole_geometry_present": False,
            "crack_geometry_present": False,
            "bicycle_symbol_geometry_present": False,
            "road_source_way_ids_retained_in_scene_asset": False,
            "road_source_tags_retained_in_scene_asset": False,
            "detail_snapshot": {
                "asset": detail_osm_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _digest(detail_osm_path),
                **_detail_snapshot_inventory(detail_osm_path),
            },
            "structure_semantics": {
                "asset": semantics_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _digest(semantics_path),
                "scope": "tunnel visibility only; not lane or marking semantics",
            },
        },
        "gates": {
            "current_scale_audit_complete": True,
            "kerb_dimensions_site_verified": False,
            "marking_width_is_metric": False,
            "road_specific_marking_semantics_available": False,
            "manhole_or_crack_placement_allowed": False,
            "detail_geometry_application_allowed": False,
            "blocking_reasons": [
                "the scene NPZ does not retain source OSM way ids or road tags",
                "kerb dimensions are generic grade-D values, not site measurements",
                "lane-paint width scales with normalized road width",
                "no dated source locates manholes, cracks, or bicycle symbols",
            ],
        },
    }
