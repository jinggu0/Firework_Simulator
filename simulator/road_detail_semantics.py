"""Checksum-locked dated OSM semantics bound to rendered road segments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROAD_DETAIL_SEMANTICS_PATH = (
    REPOSITORY_ROOT / "assets" / "yeouido_road_detail_semantics_2024-10-05.json"
)
ROAD_MARKING_ENCODING_BASE = 1024.0
ROAD_MARKING_LANE_STRIDE = 64.0
ROAD_MARKING_MAX_LANES = 12
RUNTIME_MARKING_POLICY = "explicit oneway=yes, integer lanes>=2, asphalt only"


@dataclass(frozen=True, slots=True)
class RoadDetailWaySemantics:
    osm_way_id: int
    tags: dict[str, str]
    derived_surface_code: int | None
    derived_width_m: float | None
    rendered_segment_indices: tuple[int, ...]
    binding_status: str


@dataclass(frozen=True, slots=True)
class RoadDetailSemantics:
    scene_asset: str
    scene_sha256: str
    source_asset: str
    source_sha256: str
    snapshot_utc: str
    rendered_segment_count: int
    runtime_marking_application_allowed: bool
    runtime_marking_policy: str
    ways: tuple[RoadDetailWaySemantics, ...]

    def segment_way_ids(self) -> tuple[int, ...]:
        """Return one source way id per rendered segment, or zero if unbound."""

        result = [0] * self.rendered_segment_count
        for way in self.ways:
            for index in way.rendered_segment_indices:
                if result[index] != 0:
                    raise ValueError(
                        f"rendered road segment {index} is bound more than once"
                    )
                result[index] = way.osm_way_id
        return tuple(result)


@dataclass(frozen=True, slots=True)
class MetricRoadMarkingStats:
    rendered_segment_count: int
    explicit_oneway_lane_way_count: int
    marked_segment_count: int
    marked_lane_divider_count: int
    marked_centreline_length_m: float
    suppressed_asphalt_segment_count: int


def _strict_lane_count(tags: dict[str, str]) -> int | None:
    raw = tags.get("lanes", "")
    try:
        lanes = int(raw)
    except ValueError:
        return None
    if str(lanes) != raw.strip() or not 2 <= lanes <= ROAD_MARKING_MAX_LANES:
        return None
    return lanes


def apply_metric_road_marking_semantics(
    road_vertices: np.ndarray,
    semantics: RoadDetailSemantics,
) -> tuple[np.ndarray, MetricRoadMarkingStats]:
    """Pack only explicit one-way lane boundaries into the road style channel.

    OSM ``lanes`` establishes lane count, while explicit ``oneway=yes`` makes
    every interior boundary a same-direction lane divider. Bidirectional ways
    are deliberately excluded because neither centre-line colour nor solid /
    dashed policy can be recovered from lane count alone.
    """

    converted = np.asarray(road_vertices, dtype=np.float32).copy()
    if not semantics.runtime_marking_application_allowed:
        raise ValueError("road detail semantics do not allow runtime markings")
    if semantics.runtime_marking_policy != RUNTIME_MARKING_POLICY:
        raise ValueError("road detail runtime marking policy is unsupported")
    if converted.ndim != 2 or converted.shape[1:] != (10,) or len(converted) % 6:
        raise ValueError("road marking semantics require six-vertex 10-channel quads")
    quads = converted.reshape(-1, 6, 10)
    if len(quads) != semantics.rendered_segment_count:
        raise ValueError(
            "road marking semantics and rendered segment inventories differ"
        )

    marked_segments = 0
    marked_dividers = 0
    marked_length_m = 0.0
    eligible_way_ids: set[int] = set()
    for way in semantics.ways:
        lanes = _strict_lane_count(way.tags)
        if (
            lanes is None
            or way.tags.get("oneway") != "yes"
            or way.derived_surface_code != 3
        ):
            continue
        for index in way.rendered_segment_indices:
            quad = quads[index]
            if not np.isclose(quad[0, 6], 3.0):
                continue
            centre_start = 0.5 * (quad[0, [0, 2]] + quad[1, [0, 2]])
            centre_end = 0.5 * (quad[2, [0, 2]] + quad[5, [0, 2]])
            direction = centre_end - centre_start
            length_m = float(np.linalg.norm(direction))
            if length_m < 1e-6:
                continue
            normal = np.array(
                (-direction[1], direction[0]), dtype=np.float32
            ) / length_m
            start_span = quad[1, [0, 2]] - quad[0, [0, 2]]
            end_span = quad[2, [0, 2]] - quad[5, [0, 2]]
            width_m = 0.5 * (
                abs(float(np.dot(start_span, normal)))
                + abs(float(np.dot(end_span, normal)))
            )
            if not 0.0 < width_m < ROAD_MARKING_LANE_STRIDE:
                raise ValueError(f"road segment {index} width cannot be packed")
            encoded = (
                ROAD_MARKING_ENCODING_BASE
                + lanes * ROAD_MARKING_LANE_STRIDE
                + width_m
            )
            quad[:, 9] = encoded
            eligible_way_ids.add(way.osm_way_id)
            marked_segments += 1
            marked_dividers += lanes - 1
            marked_length_m += length_m

    asphalt_segments = int(np.count_nonzero(np.isclose(quads[:, 0, 6], 3.0)))
    return converted, MetricRoadMarkingStats(
        rendered_segment_count=len(quads),
        explicit_oneway_lane_way_count=len(eligible_way_ids),
        marked_segment_count=marked_segments,
        marked_lane_divider_count=marked_dividers,
        marked_centreline_length_m=marked_length_m,
        suppressed_asphalt_segment_count=asphalt_segments - marked_segments,
    )


def _verified_asset(record: dict[str, str], label: str) -> Path:
    path = REPOSITORY_ROOT / record["asset"]
    if not path.is_file():
        raise FileNotFoundError(f"{label} asset is missing: {path}")
    actual = sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(
            f"{label} checksum mismatch: expected {record['sha256']}, got {actual}"
        )
    return path


def load_road_detail_semantics(
    path: Path = DEFAULT_ROAD_DETAIL_SEMANTICS_PATH,
) -> RoadDetailSemantics:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("road detail semantics schema_version must be 1")
    scene = payload["scene"]
    source = payload["source"]
    _verified_asset(scene, "scene")
    source_path = _verified_asset(source, "road source")
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    snapshot_utc = str(payload["snapshot_utc"])
    if source_payload.get("snapshot_utc") != snapshot_utc:
        raise ValueError("road source and binding snapshot timestamps differ")
    rendered_segment_count = int(payload["matching"]["rendered_segment_count"])
    ways = []
    seen_way_ids: set[int] = set()
    seen_segments: set[int] = set()
    for record in payload.get("ways", []):
        way_id = int(record["osm_way_id"])
        if way_id in seen_way_ids:
            raise ValueError(f"duplicate road detail way id {way_id}")
        seen_way_ids.add(way_id)
        indices = tuple(int(index) for index in record["rendered_segment_indices"])
        if any(not 0 <= index < rendered_segment_count for index in indices):
            raise ValueError(f"road detail way {way_id} has an invalid segment index")
        duplicate = seen_segments.intersection(indices)
        if duplicate:
            raise ValueError(
                f"road detail segments are multiply bound: {sorted(duplicate)[:3]}"
            )
        seen_segments.update(indices)
        ways.append(
            RoadDetailWaySemantics(
                osm_way_id=way_id,
                tags={str(key): str(value) for key, value in record["tags"].items()},
                derived_surface_code=(
                    None
                    if record["derived_surface_code"] is None
                    else int(record["derived_surface_code"])
                ),
                derived_width_m=(
                    None
                    if record["derived_width_m"] is None
                    else float(record["derived_width_m"])
                ),
                rendered_segment_indices=indices,
                binding_status=str(record["binding_status"]),
            )
        )
    expected_bound = int(payload["matching"]["uniquely_bound_segment_count"])
    if len(seen_segments) != expected_bound:
        raise ValueError(
            "road detail binding summary does not match the bound index inventory"
        )
    gates = payload["gates"]
    return RoadDetailSemantics(
        scene_asset=str(scene["asset"]),
        scene_sha256=str(scene["sha256"]),
        source_asset=str(source["asset"]),
        source_sha256=str(source["sha256"]),
        snapshot_utc=snapshot_utc,
        rendered_segment_count=rendered_segment_count,
        runtime_marking_application_allowed=bool(
            gates["runtime_marking_application_allowed"]
        ),
        runtime_marking_policy=str(gates["runtime_marking_policy"]),
        ways=tuple(ways),
    )
