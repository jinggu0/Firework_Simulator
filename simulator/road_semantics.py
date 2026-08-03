"""Evidence-scoped rendering policy for multi-level OSM roads.

The static vertex asset predates retention of per-way OSM identifiers.  This
module therefore matches only centre-lines and headings locked to a dated OSM
snapshot.  It must never infer a bridge, tunnel, or vertical elevation from
terrain shape alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROAD_SEMANTICS_PATH = (
    REPOSITORY_ROOT / "assets" / "road_structure_semantics_2024-10-05.json"
)


@dataclass(frozen=True, slots=True)
class RoadSemanticCorridor:
    osm_way_id: int
    kind: str
    render_policy: str
    polyline_xz_m: np.ndarray
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class RoadStructureSemantics:
    scene_asset: str
    scene_asset_sha256: str
    snapshot_utc: str
    centreline_tolerance_m: float
    heading_cosine_minimum: float
    corridors: tuple[RoadSemanticCorridor, ...]


@dataclass(frozen=True, slots=True)
class RoadSemanticFilterStats:
    input_segments: int
    output_segments: int
    excluded_segments: int
    excluded_osm_way_ids: tuple[int, ...]


def load_road_structure_semantics(
    path: Path = DEFAULT_ROAD_SEMANTICS_PATH,
) -> RoadStructureSemantics:
    """Load the small, auditable runtime contract derived from dated OSM."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("road semantics schema_version must be 1")
    matching = payload.get("matching", {})
    tolerance = float(matching.get("centreline_tolerance_m", 0.0))
    heading = float(matching.get("heading_cosine_minimum", 0.0))
    if not 0.0 < tolerance <= 1.0:
        raise ValueError("road semantic centreline tolerance must be in (0, 1] m")
    if not 0.9 <= heading <= 1.0:
        raise ValueError("road semantic heading cosine must be in [0.9, 1.0]")
    corridors: list[RoadSemanticCorridor] = []
    way_ids: set[int] = set()
    for item in payload.get("corridors", []):
        way_id = int(item["osm_way_id"])
        if way_id in way_ids:
            raise ValueError(f"duplicate OSM road semantic way {way_id}")
        way_ids.add(way_id)
        polyline = np.asarray(item.get("polyline_xz_m", []), dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[1:] != (2,) or len(polyline) < 2:
            raise ValueError(f"OSM way {way_id} requires a 2D polyline")
        if not np.all(np.isfinite(polyline)):
            raise ValueError(f"OSM way {way_id} contains non-finite coordinates")
        kind = str(item.get("kind", ""))
        policy = str(item.get("render_policy", ""))
        if kind != "tunnel" or policy != "occluded_below_terrain":
            raise ValueError(f"unsupported road semantic policy for OSM way {way_id}")
        corridors.append(
            RoadSemanticCorridor(
                osm_way_id=way_id,
                kind=kind,
                render_policy=policy,
                polyline_xz_m=polyline,
                tags={str(key): str(value) for key, value in item.get("tags", {}).items()},
            )
        )
    return RoadStructureSemantics(
        scene_asset=str(payload.get("scene_asset", "")),
        scene_asset_sha256=str(payload.get("scene_asset_sha256", "")),
        snapshot_utc=str(payload.get("snapshot_utc", "")),
        centreline_tolerance_m=tolerance,
        heading_cosine_minimum=heading,
        corridors=tuple(corridors),
    )


def _corridor_segment_matches(
    road_centres: np.ndarray,
    road_directions: np.ndarray,
    corridor: RoadSemanticCorridor,
    tolerance_m: float,
    heading_cosine_minimum: float,
) -> np.ndarray:
    matched = np.zeros(len(road_centres), dtype=bool)
    for start, end in zip(
        corridor.polyline_xz_m[:-1], corridor.polyline_xz_m[1:]
    ):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length < 0.1:
            continue
        direction = edge / length
        relative = road_centres - start
        along = np.clip(relative @ direction, 0.0, length)
        closest = start + along[:, None] * direction
        distance = np.linalg.norm(road_centres - closest, axis=1)
        aligned = np.abs(road_directions @ direction) >= heading_cosine_minimum
        matched |= (distance <= tolerance_m) & aligned
    return matched


def filter_occluded_road_segments(
    vertices: np.ndarray,
    semantics: RoadStructureSemantics,
) -> tuple[np.ndarray, RoadSemanticFilterStats]:
    """Remove only exact dated tunnel centre-line segments from surface draw.

    Direction matching is essential: a surface road crossing over a tunnel may
    share the same plan position, but must remain visible.  The importer emits
    one six-vertex quad per centre-line segment, which makes this policy usable
    without changing the shipped NPZ vertex layout.
    """

    source = np.asarray(vertices, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 10 or len(source) % 6:
        raise ValueError("road semantic filtering requires six-vertex 10-channel quads")
    if not len(source):
        stats = RoadSemanticFilterStats(0, 0, 0, ())
        return source.copy(), stats
    quads = source.reshape(-1, 6, 10)
    centre_start = 0.5 * (quads[:, 0, [0, 2]] + quads[:, 1, [0, 2]])
    centre_end = 0.5 * (quads[:, 2, [0, 2]] + quads[:, 5, [0, 2]])
    centres = 0.5 * (centre_start + centre_end)
    directions = centre_end - centre_start
    lengths = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(lengths[:, None], 1e-9)
    excluded = np.zeros(len(quads), dtype=bool)
    matched_way_ids: list[int] = []
    for corridor in semantics.corridors:
        matched = _corridor_segment_matches(
            centres,
            directions,
            corridor,
            semantics.centreline_tolerance_m,
            semantics.heading_cosine_minimum,
        )
        if np.any(matched):
            excluded |= matched
            matched_way_ids.append(corridor.osm_way_id)
    output = np.ascontiguousarray(quads[~excluded].reshape(-1, 10))
    stats = RoadSemanticFilterStats(
        input_segments=len(quads),
        output_segments=len(output) // 6,
        excluded_segments=int(excluded.sum()),
        excluded_osm_way_ids=tuple(matched_way_ids),
    )
    return output, stats
