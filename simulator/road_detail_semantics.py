"""Checksum-locked dated OSM semantics bound to rendered road segments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROAD_DETAIL_SEMANTICS_PATH = (
    REPOSITORY_ROOT / "assets" / "yeouido_road_detail_semantics_2024-10-05.json"
)


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
    return RoadDetailSemantics(
        scene_asset=str(scene["asset"]),
        scene_sha256=str(scene["sha256"]),
        source_asset=str(source["asset"]),
        source_sha256=str(source["sha256"]),
        snapshot_utc=snapshot_utc,
        rendered_segment_count=rendered_segment_count,
        ways=tuple(ways),
    )
