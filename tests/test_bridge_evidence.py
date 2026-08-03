from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator.bridge_evidence import (
    BridgeEvidenceError,
    apply_bridge_vertical_offsets,
    parse_bridge_evidence,
)
from tools.audit_bridge_vertical_evidence import build_report


EVIDENCE_PATH = Path("assets/seogang_bridge_vertical_evidence.json")


def _document() -> dict:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _registered_document() -> dict:
    document = _document()
    application = document["vertical_profile_application"]
    application["status"] = "registered"
    application["blockers"] = []
    application["station_registration"] = {
        "plan_rmse_m": 0.05,
        "control_points": [
            {"station_m": 0.0, "world_xz_m": [-600.0, -100.0]},
            {"station_m": 60.0, "world_xz_m": [-600.0, -160.0]},
        ],
    }
    application["profiles"] = [
        {
            "profile_id": "synthetic-gate-fixture",
            "component": "main_navigation_span",
            "source_id": "seoul_council_hangang_bridge_clearance_2012",
            "samples": [
                {
                    "station_m": 0.0,
                    "deck_underside_el_m": 23.3,
                    "structural_depth_m": 2.0,
                    "vertical_uncertainty_m": 0.05,
                },
                {
                    "station_m": 60.0,
                    "deck_top_el_m": 25.4,
                    "vertical_uncertainty_m": 0.05,
                },
            ],
        }
    ]
    return document


def test_official_underside_anchor_converts_to_runtime_datum() -> None:
    evidence = parse_bridge_evidence(_document())

    assert len(evidence.anchors) == 1
    assert evidence.anchors[0].elevation_el_m == pytest.approx(23.3)
    assert evidence.anchors[0].runtime_y_m == pytest.approx(20.51)
    assert evidence.elevation_datum_m == pytest.approx(2.79)


def test_current_manifest_blocks_clearance_as_deck_top_substitution() -> None:
    evidence = parse_bridge_evidence(_document())
    vertices = np.zeros((6, 10), dtype=np.float32)
    original = vertices.copy()

    with pytest.raises(BridgeEvidenceError, match="application is blocked"):
        apply_bridge_vertical_offsets(vertices, np.ones(6), evidence)

    assert np.array_equal(vertices, original)
    assert "no_registered_vertical_profile" in evidence.application.reasons
    assert "station_registration_missing" in evidence.application.reasons


def test_registered_station_and_vertical_profile_unlock_only_explicit_offsets() -> None:
    evidence = parse_bridge_evidence(_registered_document())
    vertices = np.zeros((6, 10), dtype=np.float32)
    offsets = np.linspace(0.0, 0.5, 6)

    corrected = apply_bridge_vertical_offsets(vertices, offsets, evidence)

    assert evidence.application.allowed
    assert np.allclose(corrected[:, 1], offsets)
    assert np.array_equal(vertices, np.zeros((6, 10), dtype=np.float32))
    assert np.array_equal(corrected[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]], vertices[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda document: document["vertical_profile_application"][
                "station_registration"
            ].update({"plan_rmse_m": 0.5}),
            "station_plan_rmse_exceeds_threshold",
        ),
        (
            lambda document: document["vertical_profile_application"]["profiles"][0][
                "samples"
            ][0].update({"vertical_uncertainty_m": 0.2}),
            "uncertainty_exceeds_threshold",
        ),
        (
            lambda document: document["vertical_profile_application"]["profiles"][0][
                "samples"
            ][0].update(
                {
                    "deck_underside_el_m": None,
                    "structural_depth_m": None,
                }
            ),
            "cannot_resolve_deck_top",
        ),
    ],
)
def test_profile_quality_failures_keep_application_closed(mutation, reason: str) -> None:
    document = _registered_document()
    mutation(document)

    evidence = parse_bridge_evidence(document)

    assert not evidence.application.allowed
    assert any(reason in item for item in evidence.application.reasons)


def test_manifest_rejects_unlocked_or_non_evidence_source() -> None:
    document = _document()
    document["sources"][0]["checksum"] = "unknown"
    with pytest.raises(BridgeEvidenceError, match="sha256"):
        parse_bridge_evidence(document)

    document = _document()
    document["sources"][0]["confidence_grade"] = "D"
    with pytest.raises(BridgeEvidenceError, match="grade A or B"):
        parse_bridge_evidence(document)

    document = _document()
    document["vertical_anchors"][0]["runtime_y_m"] = 23.3
    with pytest.raises(BridgeEvidenceError, match="does not match the scene datum"):
        parse_bridge_evidence(document)

    document = _document()
    document["published_dimensions"][0]["source_id"] = "unknown"
    with pytest.raises(BridgeEvidenceError, match="published dimension 0"):
        parse_bridge_evidence(document)


def test_audit_is_checksum_locked_and_reports_no_scene_mutation(tmp_path: Path) -> None:
    scene = Path("assets/yeouido_scene.npz").resolve()
    report = build_report(EVIDENCE_PATH.resolve(), scene)

    assert report["passed"] is True
    assert report["scene"]["checksum_matches"] is True
    assert report["scene"]["datum_matches"] is True
    assert report["geometry_application"]["allowed"] is False
    assert report["geometry_application"]["vertices_modified"] == 0
    assert report["regression"]["scene_checksum_unchanged_during_audit"] is True
    assert report["regression"]["runtime_frame_path_changed"] is False


def test_schema_locks_event_and_requires_station_controls() -> None:
    schema = json.loads(
        Path("assets/bridge_vertical_evidence.schema.json").read_text(encoding="utf-8")
    )

    assert schema["properties"]["target_event_date"]["const"] == "2024-10-05"
    assert schema["$defs"]["station_registration"]["properties"]["control_points"][
        "minItems"
    ] == 2
    assert schema["$defs"]["profile"]["properties"]["samples"]["minItems"] == 2
