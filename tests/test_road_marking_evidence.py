from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest


REPORT = Path(
    "docs/validation/road_marking_evidence_v2/road_marking_evidence_report.json"
)


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_committed_evidence_audit_locks_event_dated_inputs() -> None:
    report = _report()

    assert report["stage"] == "V2-2d"
    assert report["event_date"] == "2024-10-05"
    assert report["snapshot_utc"] == "2024-10-05T10:20:00Z"
    for source in report["inputs"].values():
        path = Path(source["asset"])
        assert sha256(path.read_bytes()).hexdigest() == source["sha256"]
    diagnostic = report["diagnostic"]
    assert sha256(Path(diagnostic["asset"]).read_bytes()).hexdigest() == diagnostic[
        "sha256"
    ]


def test_candidate_inventory_is_bound_but_not_misread_as_placement_evidence() -> None:
    groups = _report()["candidate_groups"]

    expected = {
        "already_rendered_oneway_lane_dividers": (66, 4_387, 48_567.027438),
        "centre_line_candidates_non_oneway_yes_multilane": (37, 502, 5_137.440903),
        "directional_lane_count_candidates": (5, 103, 1_118.209351),
        "cycle_lane_tag_candidates": (31, 694, 7_409.043998),
        "crossing_marking_presence_candidates": (3, 6, 43.591528),
    }
    for name, (ways, segments, length_m) in expected.items():
        group = groups[name]
        assert group["way_count"] == ways
        assert group["rendered_segment_count"] == segments
        assert group["rendered_centreline_length_m"] == pytest.approx(length_m)
        assert sum(record["rendered_segment_count"] for record in group["records"]) == segments


def test_missing_way_level_marking_tags_keep_all_new_types_blocked() -> None:
    report = _report()
    inventory = report["historical_osm_tag_inventory"]
    gates = report["application_gates"]

    for key in (
        "divider",
        "shoulder",
        "turn:lanes",
        "turn:lanes:forward",
        "turn:lanes:backward",
        "road_marking",
        "lane_markings",
        "centre_line",
        "center_line",
    ):
        assert inventory[key]["way_count"] == 0
    assert gates["existing_same_direction_lane_dividers_remain_allowed"]
    assert not gates["bidirectional_centre_line_application_allowed"]
    assert not gates["centre_line_colour_selection_allowed"]
    assert not gates["centre_line_solid_dash_selection_allowed"]
    assert not gates["edge_line_application_allowed"]
    assert not gates["turn_arrow_application_allowed"]
    assert not gates["bicycle_symbol_application_allowed"]
    assert not gates["crossing_marking_shape_application_allowed"]
    assert not gates["runtime_geometry_changed_by_this_stage"]


def test_next_imagery_registration_contract_is_spatial_and_dated() -> None:
    contract = _report()["next_registration_contract"]

    assert contract["registration"] == {
        "minimum_control_points": 4,
        "minimum_independent_check_points": 2,
        "project_max_planimetric_residual_m": 1.0,
        "control_points_must_span_each_registered_crop": True,
    }
    acceptance = " ".join(contract["marking_acceptance"])
    assert "event" in acceptance
    assert "OSM way IDs" in acceptance
    assert "solid/dashed" in acceptance
