from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from simulator.road_detail_semantics import load_road_detail_semantics


SOURCE = Path("assets/yeouido_road_osm_2024-10-05.json")
BINDING = Path("assets/yeouido_road_detail_semantics_2024-10-05.json")


def test_dated_road_source_preserves_highway_ways_and_licence() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["snapshot_utc"] == "2024-10-05T10:20:00Z"
    assert payload["licence"] == "ODbL 1.0"
    assert payload["bbox_wgs84"] == [126.91, 37.515, 126.96, 37.545]
    assert len(payload["elements"]) == 2_421
    assert all(element["tags"].get("highway") for element in payload["elements"])


def test_committed_binding_covers_every_nonambiguous_rendered_segment() -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    semantics = load_road_detail_semantics()
    matching = payload["matching"]

    assert payload["stage"] == "V2-2b"
    assert payload["source"]["sha256"] == sha256(SOURCE.read_bytes()).hexdigest()
    assert semantics.snapshot_utc == "2024-10-05T10:20:00Z"
    assert len(semantics.ways) == 2_421
    assert semantics.rendered_segment_count == 35_215
    assert matching["uniquely_bound_segment_count"] == 35_203
    assert matching["ambiguous_segment_count"] == 12
    assert matching["unmatched_segment_count"] == 0
    assert matching["unique_binding_coverage_fraction"] > 0.999
    segment_way_ids = semantics.segment_way_ids()
    assert len(segment_way_ids) == 35_215
    assert sum(way_id != 0 for way_id in segment_way_ids) == 35_203


def test_binding_retains_only_evidenced_semantics_and_blocks_runtime_paint() -> None:
    payload = json.loads(BINDING.read_text(encoding="utf-8"))
    coverage = payload["tag_coverage_way_counts"]

    assert coverage["width"] == 52
    assert coverage["lanes"] == 189
    assert coverage["surface"] == 437
    assert coverage["cycleway:right"] == 52
    assert coverage["bicycle"] == 162
    assert coverage["turn:lanes"] == 0
    assert payload["matching"]["binding_status_counts"]["runtime_occluded"] == 8
    gates = payload["gates"]
    assert gates["historical_source_preserved"]
    assert gates["scene_checksum_unchanged"]
    assert gates["unique_binding_coverage_sufficient"]
    assert not gates["runtime_marking_application_allowed"]
