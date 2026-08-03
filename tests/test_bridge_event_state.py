from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from simulator.bridge_event_state import (
    BridgeEventStateError,
    load_bridge_event_state,
    parse_bridge_event_state,
)
from tools.audit_bridge_event_state import build_report


ASSET = Path("assets/seogang_bridge_event_state_2024-10-05.json")


def _document() -> dict:
    return json.loads(ASSET.read_text(encoding="utf-8"))


def test_shipped_event_state_blocks_unobserved_construction_and_station_claims() -> None:
    state = load_bridge_event_state()

    assert state.independent_event_views == 3
    assert state.qualifying_event_views == 0
    assert state.identified_station_controls == 2
    assert state.longitudinal_plan_m_per_pixel == pytest.approx(0.5)
    assert not state.event_day_visual_state_allowed
    assert not state.station_registration_allowed
    assert not state.vertical_profile_allowed
    assert "event_day_scaffolding_state_unresolved" in state.reasons
    assert "structural_history_not_verified_through_event_date" in state.reasons


def test_two_registered_views_do_not_unlock_a_coarse_two_control_drawing() -> None:
    document = _document()
    for record in document["event_photo_review"][:2]:
        record["seogang_bridge_visible"] = True
        record["resolved_bridge_span_pixels"] = 500
    document["event_day_assessment"]["scaffolding_state"] = "observed_absent"
    document["event_day_assessment"]["fresh_paint_boundary_state"] = "observed_absent"
    document["application"]["event_day_visual_state_allowed"] = True

    state = parse_bridge_event_state(document)

    assert state.event_day_visual_state_allowed
    assert not state.station_registration_allowed
    assert not state.vertical_profile_allowed


def test_station_gate_requires_resolution_grade_controls_rmse_and_history() -> None:
    document = _document()
    drawing = document["official_drawing_review"]
    drawing["identified_control_landmarks"].append(
        {"control_id": "main_pier_p10", "drawing_role": "psc_control"}
    )
    drawing["longitudinal_raster_estimated_plan_m_per_pixel_lower_bound"] = 0.1
    drawing["plan_target_source_grade"] = "B"
    drawing["station_plan_rmse_m"] = 0.2
    document["structural_history"][
        "verified_no_major_design_change_through"
    ] = "2024-10-05"
    document["structural_history"]["verified_through_event_date"] = True
    document["application"]["station_registration_allowed"] = True
    document["application"]["vertical_profile_allowed"] = True

    state = parse_bridge_event_state(document)

    assert state.station_registration_allowed
    assert state.vertical_profile_allowed


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["sources"][4].update({"checksum": "unknown"}),
        lambda document: document["event_photo_review"][0].update({"width_px": 0}),
        lambda document: document["official_drawing_review"]["extracted_artifacts"][0].update({"sha256": "bad"}),
    ],
)
def test_event_state_rejects_unlocked_or_invalid_review_inputs(mutation) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(BridgeEventStateError):
        parse_bridge_event_state(document)


def test_event_state_rejects_an_application_overclaim() -> None:
    document = deepcopy(_document())
    document["application"]["vertical_profile_allowed"] = True
    with pytest.raises(BridgeEventStateError, match="claims"):
        parse_bridge_event_state(document)


def test_event_state_rejects_rendering_an_unresolved_scaffold() -> None:
    document = _document()
    document["event_day_assessment"]["render_scaffolding"] = True
    with pytest.raises(BridgeEventStateError, match="render flags overclaim"):
        parse_bridge_event_state(document)


def test_event_state_audit_locks_the_linked_manifest_and_keeps_runtime_unchanged() -> None:
    report = build_report()

    assert report["vertical_evidence_link"]["checksum_matches"]
    assert not report["application"]["event_day_visual_state_allowed"]
    assert not report["application"]["station_registration_allowed"]
    assert not report["application"]["vertical_profile_allowed"]
    assert report["application"]["scene_vertices_modified"] == 0
    assert not report["application"]["runtime_frame_path_changed"]
    assert report["passed"]
