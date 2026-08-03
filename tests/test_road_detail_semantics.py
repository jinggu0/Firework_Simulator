from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from simulator.road_detail_semantics import (
    ROAD_MARKING_ENCODING_BASE,
    ROAD_MARKING_LANE_STRIDE,
    apply_metric_road_marking_semantics,
    load_road_detail_semantics,
)
from simulator.validation.road_details import rendered_road_measurements


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


def test_binding_retains_only_evidenced_semantics_and_limits_runtime_paint() -> None:
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
    assert gates["runtime_marking_application_allowed"]
    assert "oneway=yes" in gates["runtime_marking_policy"]


def test_metric_markings_apply_only_to_explicit_oneway_multilane_segments() -> None:
    measurements, _, _ = rendered_road_measurements()
    semantics = load_road_detail_semantics()
    converted, stats = apply_metric_road_marking_semantics(
        measurements["quads"].reshape(-1, 10), semantics
    )
    quads = converted.reshape(-1, 6, 10)
    encoded = quads[:, 0, 9] >= ROAD_MARKING_ENCODING_BASE

    assert stats.rendered_segment_count == 35_215
    assert stats.explicit_oneway_lane_way_count == 66
    assert stats.marked_segment_count == 4_387
    assert stats.suppressed_asphalt_segment_count == 21_229
    assert int(encoded.sum()) == stats.marked_segment_count

    packed = quads[encoded, 0, 9] - ROAD_MARKING_ENCODING_BASE
    lane_counts = np.floor(packed / ROAD_MARKING_LANE_STRIDE).astype(int)
    assert lane_counts.min() == 2
    assert lane_counts.max() == 9


def test_metric_markings_refuse_a_disabled_binding_gate() -> None:
    measurements, _, _ = rendered_road_measurements()
    semantics = replace(
        load_road_detail_semantics(),
        runtime_marking_application_allowed=False,
    )

    with pytest.raises(ValueError, match="do not allow"):
        apply_metric_road_marking_semantics(
            measurements["quads"].reshape(-1, 10), semantics
        )
