"""Dated plan replacement for the multiply-rendered Seogang Bridge deck."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_SEMANTICS_PATH = (
    REPOSITORY_ROOT / "assets" / "seogang_bridge_semantics_2024-10-05.json"
)


class BridgePlanSemanticError(ValueError):
    """Raised when the plan replacement asset is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class BridgePlanSemantics:
    outline_xz_m: np.ndarray
    cross_sections_xz_m: np.ndarray
    matching_distance_m: float
    minimum_matching_source_segments: int
    local_y_m: float
    surface_code: float
    station_registration_passed: bool
    station_length_residual_m: float
    event_inside_paint_contract: bool
    construction_visual_state_known: bool


@dataclass(frozen=True, slots=True)
class BridgePlanReplacementStats:
    input_segments: int
    removed_generic_segments: int
    retained_segments: int
    replacement_segments: int
    input_vertices: int
    output_vertices: int


def parse_bridge_plan_semantics(document: Mapping[str, Any]) -> BridgePlanSemantics:
    if document.get("schema_version") != 1:
        raise BridgePlanSemanticError("unsupported bridge plan schema")
    if document.get("target_event_date") != "2024-10-05":
        raise BridgePlanSemanticError("bridge plan must target 2024-10-05")
    if document.get("bridge_id") != "seogang_bridge":
        raise BridgePlanSemanticError("bridge plan must identify seogang_bridge")
    outline = np.asarray(document.get("deck_outline_xz_m", []), dtype=np.float64)
    if outline.ndim != 2 or outline.shape[1:] != (2,) or len(outline) < 4:
        raise BridgePlanSemanticError("deck outline must be an n x 2 ring")
    if not np.isfinite(outline).all() or not np.allclose(outline[0], outline[-1]):
        raise BridgePlanSemanticError("deck outline must be finite and closed")
    raw_sections = document.get("replacement_cross_sections", [])
    try:
        sections = np.asarray(
            [
                [item["side_a_xz_m"], item["side_b_xz_m"]]
                for item in raw_sections
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BridgePlanSemanticError("invalid replacement cross sections") from error
    if sections.ndim != 3 or sections.shape[1:] != (2, 2) or len(sections) < 2:
        raise BridgePlanSemanticError("at least two finite cross sections are required")
    if not np.isfinite(sections).all():
        raise BridgePlanSemanticError("cross sections must be finite")
    metrics = document.get("plan_metrics", {})
    policy = document.get("render_policy", {})
    sources = document.get("sources", {})
    paint = sources.get("official_2024_paint_contract", {})
    period = paint.get("work_period", [])
    event_inside = len(period) == 2 and period[0] <= "2024-10-05" <= period[1]
    matching_distance = float(policy.get("matching_distance_m", 0.0))
    minimum_matching = int(policy.get("minimum_matching_source_segments", 0))
    local_y = float(policy.get("replacement_local_y_m", 0.0))
    surface_code = float(policy.get("replacement_surface_code", 2.0))
    length_residual = float(metrics.get("length_residual_m", 0.0))
    if not np.isfinite(
        [matching_distance, local_y, surface_code, length_residual]
    ).all():
        raise BridgePlanSemanticError("bridge render policy must be finite")
    if matching_distance < 0.0:
        raise BridgePlanSemanticError("matching distance cannot be negative")
    if minimum_matching < 1:
        raise BridgePlanSemanticError("minimum matching source segments must be positive")
    widths = np.linalg.norm(sections[:, 1] - sections[:, 0], axis=1)
    if np.any(widths <= 1.0):
        raise BridgePlanSemanticError("replacement cross sections must have width")
    centre = sections.mean(axis=1)
    if np.any(np.linalg.norm(np.diff(centre, axis=0), axis=1) <= 0.01):
        raise BridgePlanSemanticError("replacement stations must advance")
    return BridgePlanSemantics(
        outline_xz_m=outline,
        cross_sections_xz_m=sections,
        matching_distance_m=matching_distance,
        minimum_matching_source_segments=minimum_matching,
        local_y_m=local_y,
        surface_code=surface_code,
        station_registration_passed=(
            metrics.get("station_registration_status") == "passed"
        ),
        station_length_residual_m=length_residual,
        event_inside_paint_contract=event_inside,
        construction_visual_state_known=(
            paint.get("event_day_status") == "registered_event_day_progress"
        ),
    )


def load_bridge_plan_semantics(
    path: Path = DEFAULT_BRIDGE_SEMANTICS_PATH,
) -> BridgePlanSemantics:
    return parse_bridge_plan_semantics(json.loads(path.read_text(encoding="utf-8")))


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    z = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for start, end in zip(polygon[:-1], polygon[1:]):
        crosses = (start[1] > z) != (end[1] > z)
        intersection_x = (
            (end[0] - start[0]) * (z - start[1])
            / (end[1] - start[1] + 1e-30)
            + start[0]
        )
        inside ^= crosses & (x < intersection_x)
    return inside


def _distance_to_outline(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    result = np.full(len(points), np.inf, dtype=np.float64)
    for start, end in zip(polygon[:-1], polygon[1:]):
        segment = end - start
        denominator = float(np.dot(segment, segment))
        if denominator <= 1e-12:
            continue
        alpha = np.clip(((points - start) @ segment) / denominator, 0.0, 1.0)
        closest = start + alpha[:, None] * segment
        result = np.minimum(result, np.linalg.norm(points - closest, axis=1))
    return result


def replacement_deck_vertices(semantics: BridgePlanSemantics) -> np.ndarray:
    output: list[list[float]] = []
    y = semantics.local_y_m
    for current, following in zip(
        semantics.cross_sections_xz_m[:-1], semantics.cross_sections_xz_m[1:]
    ):
        left0, right0 = current
        left1, right1 = following
        for point, v in (
            (left0, -1.0), (right0, 1.0), (right1, 1.0),
            (left0, -1.0), (right1, 1.0), (left1, -1.0),
        ):
            output.append(
                [
                    float(point[0]), y, float(point[1]),
                    0.0, 1.0, 0.0,
                    semantics.surface_code, 0.0, v, 0.0,
                ]
            )
    return np.asarray(output, dtype=np.float32).reshape(-1, 10)


def seogang_bridge_segment_mask(
    bridge_vertices: np.ndarray,
    semantics: BridgePlanSemantics,
) -> np.ndarray:
    """Select generic six-vertex deck segments covered by the dated outline."""
    vertices = np.asarray(bridge_vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 10 or len(vertices) % 6:
        raise BridgePlanSemanticError("bridge mesh must contain six-vertex n x 10 quads")
    quads = vertices.reshape(-1, 6, 10)
    centres = 0.25 * (
        quads[:, 0, [0, 2]]
        + quads[:, 1, [0, 2]]
        + quads[:, 2, [0, 2]]
        + quads[:, 5, [0, 2]]
    )
    inside = _inside_polygon(centres, semantics.outline_xz_m)
    near = _distance_to_outline(centres, semantics.outline_xz_m) <= (
        semantics.matching_distance_m
    )
    return inside | near


def replace_seogang_bridge_plan(
    bridge_vertices: np.ndarray,
    semantics: BridgePlanSemantics,
) -> tuple[np.ndarray, BridgePlanReplacementStats]:
    vertices = np.asarray(bridge_vertices, dtype=np.float32)
    removed = seogang_bridge_segment_mask(vertices, semantics)
    quads = vertices.reshape(-1, 6, 10)
    if int(removed.sum()) < semantics.minimum_matching_source_segments:
        stats = BridgePlanReplacementStats(
            input_segments=len(quads),
            removed_generic_segments=0,
            retained_segments=len(quads),
            replacement_segments=0,
            input_vertices=len(vertices),
            output_vertices=len(vertices),
        )
        return vertices.copy(), stats
    retained = quads[~removed].reshape(-1, 10)
    replacement = replacement_deck_vertices(semantics)
    output = np.concatenate((retained, replacement), axis=0)
    stats = BridgePlanReplacementStats(
        input_segments=len(quads),
        removed_generic_segments=int(removed.sum()),
        retained_segments=int((~removed).sum()),
        replacement_segments=len(replacement) // 6,
        input_vertices=len(vertices),
        output_vertices=len(output),
    )
    return output, stats
