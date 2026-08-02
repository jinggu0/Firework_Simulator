"""Evidence-gated meshes for surveyed retaining walls and earthworks.

Normalized NGII features carry plan geometry and optional absolute elevation.
They do not identify whether a line is a wall top, crest, or toe.  This module
therefore requires an independently sourced profile decision before producing
triangles.  It never supplies a default wall height or slope cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import ConfidenceGrade, DataRecord
from .scene import SURFACE_EARTHWORK, SURFACE_RETAINING_WALL
from .terrain import sample_heightmap_array


class StructureEvidenceError(ValueError):
    """Raised when geometry would require evidence the profile does not hold."""


@dataclass(frozen=True, slots=True)
class StructureMeshResult:
    vertices: np.ndarray
    profiles_built: int
    source_segments: int
    rendered_segments: int
    skipped_segments: int


def _as_elevated_points(feature: Mapping[str, Any]) -> np.ndarray:
    raw = feature.get("points_eus_m", [])
    if len(raw) < 2:
        raise StructureEvidenceError("structure feature has fewer than two points")
    if any(len(point) != 3 or point[1] is None for point in raw):
        raise StructureEvidenceError(
            f"feature {feature.get('feature_id', '<unknown>')} has no complete "
            "source elevation"
        )
    points = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(points).all():
        raise StructureEvidenceError("structure feature contains non-finite points")
    return points


def _subdivide(points: np.ndarray, maximum_segment_length_m: float) -> np.ndarray:
    output = [points[0]]
    for start, end in zip(points[:-1], points[1:]):
        plan_length = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        divisions = max(1, int(math.ceil(plan_length / maximum_segment_length_m)))
        output.extend(
            start + (end - start) * (index / divisions)
            for index in range(1, divisions + 1)
        )
    return np.asarray(output, dtype=np.float64)


def _vertex(
    position: np.ndarray,
    normal: np.ndarray,
    surface: float,
    uv: tuple[float, float],
) -> list[float]:
    return [
        float(position[0]),
        float(position[1]),
        float(position[2]),
        float(normal[0]),
        float(normal[1]),
        float(normal[2]),
        surface,
        uv[0],
        uv[1],
        0.0,
    ]


def _append_quad(
    output: list[list[float]],
    points: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    surface: float,
    uv: tuple[tuple[float, float], ...],
    *,
    double_sided: bool = False,
    normal: np.ndarray | None = None,
) -> bool:
    resolved_normal = (
        np.asarray(normal, dtype=np.float64)
        if normal is not None
        else np.cross(points[1] - points[0], points[2] - points[0])
    )
    length = float(np.linalg.norm(resolved_normal))
    if length < 1e-8:
        return False
    resolved_normal /= length
    for indices in ((0, 1, 2, 0, 2, 3),):
        output.extend(
            _vertex(points[index], resolved_normal, surface, uv[index])
            for index in indices
        )
    if double_sided:
        output.extend(
            _vertex(points[index], -resolved_normal, surface, uv[index])
            for index in (3, 2, 0, 2, 1, 0)
        )
    return True


def retaining_wall_face_vertices(
    top_edge_eus_m: np.ndarray,
    terrain_height_m: np.ndarray,
    terrain_bounds: np.ndarray,
    *,
    maximum_segment_length_m: float = 4.0,
    minimum_visible_height_m: float = 0.08,
) -> tuple[np.ndarray, int, int]:
    """Build a zero-thickness wall face from a surveyed top and terrain foot.

    Source Y is absolute relative to the event datum.  Static scene Y is an
    offset to which the vertex shader adds the official terrain, so every top
    sample is converted to that offset and every bottom sample is exactly zero.
    Both windings are emitted because a plan line carries no outward-side
    semantic and back-face culling must not make one riverbank direction vanish.
    """

    points = _subdivide(
        np.asarray(top_edge_eus_m, dtype=np.float64), maximum_segment_length_m
    )
    ground = sample_heightmap_array(
        terrain_height_m, terrain_bounds, points[:, [0, 2]]
    ).astype(np.float64)
    height = points[:, 1] - ground
    output: list[list[float]] = []
    skipped = 0
    distance = 0.0
    for index in range(len(points) - 1):
        start, end = points[index], points[index + 1]
        length = float(np.linalg.norm(end[[0, 2]] - start[[0, 2]]))
        if length < 1e-6 or min(height[index], height[index + 1]) < minimum_visible_height_m:
            skipped += 1
            distance += length
            continue
        bottom_start = np.array([start[0], 0.0, start[2]], dtype=np.float64)
        bottom_end = np.array([end[0], 0.0, end[2]], dtype=np.float64)
        top_start = np.array([start[0], height[index], start[2]], dtype=np.float64)
        top_end = np.array([end[0], height[index + 1], end[2]], dtype=np.float64)
        world_face = (
            np.array([start[0], ground[index], start[2]]),
            np.array([end[0], ground[index + 1], end[2]]),
            end.copy(),
            start.copy(),
        )
        _append_quad(
            output,
            (bottom_start, bottom_end, top_end, top_start),
            SURFACE_RETAINING_WALL,
            (
                (distance, 0.0),
                (distance + length, 0.0),
                (distance + length, float(height[index + 1])),
                (distance, float(height[index])),
            ),
            double_sided=True,
            normal=np.cross(
                world_face[1] - world_face[0], world_face[2] - world_face[0]
            ),
        )
        distance += length
    vertices = np.asarray(output, dtype=np.float32).reshape(-1, 10)
    return vertices, len(points) - 1, skipped


def _normalized_resample(points: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    segment = np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] < 1e-6:
        raise StructureEvidenceError("surveyed edge has zero plan length")
    cumulative /= cumulative[-1]
    return np.column_stack(
        [np.interp(fractions, cumulative, points[:, axis]) for axis in range(3)]
    )


def surveyed_slope_vertices(
    crest_eus_m: np.ndarray,
    toe_eus_m: np.ndarray,
    terrain_height_m: np.ndarray,
    terrain_bounds: np.ndarray,
    *,
    maximum_segment_length_m: float = 4.0,
) -> tuple[np.ndarray, int]:
    """Join independently surveyed crest and toe edges without a made-up slope."""

    crest = np.asarray(crest_eus_m, dtype=np.float64)
    toe = np.asarray(toe_eus_m, dtype=np.float64)
    direct = np.linalg.norm(crest[0, [0, 2]] - toe[0, [0, 2]]) + np.linalg.norm(
        crest[-1, [0, 2]] - toe[-1, [0, 2]]
    )
    reversed_distance = np.linalg.norm(
        crest[0, [0, 2]] - toe[-1, [0, 2]]
    ) + np.linalg.norm(crest[-1, [0, 2]] - toe[0, [0, 2]])
    if reversed_distance < direct:
        toe = toe[::-1]
    crest_length = float(
        np.linalg.norm(np.diff(crest[:, [0, 2]], axis=0), axis=1).sum()
    )
    toe_length = float(
        np.linalg.norm(np.diff(toe[:, [0, 2]], axis=0), axis=1).sum()
    )
    divisions = max(
        1, int(math.ceil(max(crest_length, toe_length) / maximum_segment_length_m))
    )
    fractions = np.linspace(0.0, 1.0, divisions + 1)
    crest = _normalized_resample(crest, fractions)
    toe = _normalized_resample(toe, fractions)
    combined = np.concatenate((crest, toe), axis=0)
    ground = sample_heightmap_array(
        terrain_height_m, terrain_bounds, combined[:, [0, 2]]
    ).astype(np.float64)
    offsets = combined[:, 1] - ground
    crest_offset, toe_offset = offsets[: len(crest)], offsets[len(crest) :]
    output: list[list[float]] = []
    distance = 0.0
    for index in range(divisions):
        crest_start = np.array(
            [crest[index, 0], crest_offset[index], crest[index, 2]]
        )
        crest_end = np.array(
            [crest[index + 1, 0], crest_offset[index + 1], crest[index + 1, 2]]
        )
        toe_start = np.array([toe[index, 0], toe_offset[index], toe[index, 2]])
        toe_end = np.array(
            [toe[index + 1, 0], toe_offset[index + 1], toe[index + 1, 2]]
        )
        length = 0.5 * (
            np.linalg.norm(crest_end - crest_start) + np.linalg.norm(toe_end - toe_start)
        )
        points = (crest_start, toe_start, toe_end, crest_end)
        world_points = (
            crest[index], toe[index], toe[index + 1], crest[index + 1]
        )
        normal = np.cross(
            world_points[1] - world_points[0],
            world_points[2] - world_points[0],
        )
        if normal[1] < 0.0:
            points = (crest_start, crest_end, toe_end, toe_start)
            world_points = (
                crest[index], crest[index + 1], toe[index + 1], toe[index]
            )
            normal = np.cross(
                world_points[1] - world_points[0],
                world_points[2] - world_points[0],
            )
            uv = (
                (distance, 0.0),
                (distance + length, 0.0),
                (distance + length, 1.0),
                (distance, 1.0),
            )
        else:
            uv = (
                (distance, 0.0),
                (distance, 1.0),
                (distance + length, 1.0),
                (distance + length, 0.0),
            )
        _append_quad(
            output,
            points,
            SURFACE_EARTHWORK,
            uv,
            normal=normal,
        )
        distance += float(length)
    return np.asarray(output, dtype=np.float32).reshape(-1, 10), divisions


def _evidence_record(profile: Mapping[str, Any]) -> DataRecord:
    raw = profile.get("evidence")
    if not isinstance(raw, Mapping):
        raise StructureEvidenceError("every structure profile requires evidence")
    try:
        record = DataRecord.from_dict(raw)
    except ValueError as error:
        raise StructureEvidenceError(str(error)) from error
    if record.grade not in (
        ConfidenceGrade.MEASURED,
        ConfidenceGrade.RECONSTRUCTED,
    ):
        raise StructureEvidenceError(
            "structure meshes require confidence grade A or B; modelled or "
            "artistic cross-sections remain non-geometric"
        )
    return record


def build_structure_mesh(
    asset: Mapping[str, Any],
    profile_document: Mapping[str, Any],
    terrain_height_m: np.ndarray,
    terrain_bounds: np.ndarray,
    *,
    allow_post_event_source: bool = False,
) -> StructureMeshResult:
    """Resolve audited profiles into one static structure vertex batch."""

    if asset.get("schema_version") != 2:
        raise StructureEvidenceError("normalized structure asset must use schema_version 2")
    if profile_document.get("schema_version") != 1:
        raise StructureEvidenceError("profile document must use schema_version 1")
    if asset.get("temporal_relation") == "official_post_event" and not allow_post_event_source:
        raise StructureEvidenceError(
            "post-event structure data requires explicit allow_post_event_source"
        )
    feature_list = asset.get("features", [])
    features = {feature.get("feature_id"): feature for feature in feature_list}
    if None in features:
        raise StructureEvidenceError("normalized asset has a feature without feature_id")
    if len(features) != len(feature_list):
        raise StructureEvidenceError("normalized asset contains duplicate feature_id values")
    profiles = profile_document.get("profiles", [])
    if not isinstance(profiles, list):
        raise StructureEvidenceError("profiles must be a list")
    if not profiles:
        raise StructureEvidenceError("profile document contains no audited profiles")
    batches: list[np.ndarray] = []
    source_segments = 0
    rendered_segments = 0
    skipped_segments = 0
    used: set[str] = set()

    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise StructureEvidenceError("profile entries must be objects")
        _evidence_record(profile)
        feature_id = str(profile.get("feature_id", ""))
        if feature_id in used:
            raise StructureEvidenceError(f"feature {feature_id} is profiled more than once")
        try:
            feature = features[feature_id]
        except KeyError as error:
            raise StructureEvidenceError(f"unknown feature_id {feature_id!r}") from error
        mesh_kind = profile.get("mesh_kind")
        if mesh_kind == "retaining_wall_face":
            if feature.get("kind") != "retaining_wall":
                raise StructureEvidenceError("wall profile references a non-wall feature")
            if profile.get("source_edge_role") != "top" or profile.get(
                "lower_edge_source"
            ) != "official_terrain":
                raise StructureEvidenceError(
                    "wall face requires an evidenced top role and official_terrain foot"
                )
            vertices, segments, skipped = retaining_wall_face_vertices(
                _as_elevated_points(feature), terrain_height_m, terrain_bounds
            )
            if not len(vertices):
                raise StructureEvidenceError(
                    f"wall feature {feature_id} has no top above official terrain"
                )
            source_segments += len(feature["points_eus_m"]) - 1
            rendered_segments += segments - skipped
            skipped_segments += skipped
            batches.append(vertices)
        elif mesh_kind == "surveyed_slope":
            if feature.get("kind") not in {"embankment", "cut_or_fill"}:
                raise StructureEvidenceError("slope profile references an incompatible feature")
            paired_id = str(profile.get("paired_feature_id", ""))
            if paired_id == feature_id or paired_id not in features:
                raise StructureEvidenceError("surveyed slope requires a distinct paired feature")
            paired = features[paired_id]
            if paired.get("kind") != feature.get("kind"):
                raise StructureEvidenceError(
                    "surveyed slope edges must share the same feature kind"
                )
            roles = (profile.get("source_edge_role"), profile.get("paired_edge_role"))
            if set(roles) != {"crest", "toe"}:
                raise StructureEvidenceError("surveyed slope requires crest and toe edge roles")
            vertices, segments = surveyed_slope_vertices(
                _as_elevated_points(feature),
                _as_elevated_points(paired),
                terrain_height_m,
                terrain_bounds,
            )
            source_segments += len(feature["points_eus_m"]) + len(
                paired["points_eus_m"]
            ) - 2
            rendered_segments += segments
            batches.append(vertices)
            used.add(paired_id)
        else:
            raise StructureEvidenceError(f"unsupported mesh_kind {mesh_kind!r}")
        used.add(feature_id)

    vertices = (
        np.concatenate(batches, axis=0)
        if batches
        else np.empty((0, 10), dtype=np.float32)
    )
    return StructureMeshResult(
        vertices=vertices,
        profiles_built=len(profiles),
        source_segments=source_segments,
        rendered_segments=rendered_segments,
        skipped_segments=skipped_segments,
    )
