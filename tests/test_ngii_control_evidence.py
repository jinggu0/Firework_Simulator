from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from simulator.ngii_control_evidence import (
    NgiiControlEvidenceError,
    load_ngii_control_evidence,
    parse_ngii_control_evidence,
)
from tools.audit_ngii_public_controls import build_report


ASSET = Path("assets/yeouido_ngii_public_controls_2017.json")


def _document() -> dict:
    return json.loads(ASSET.read_text(encoding="utf-8"))


def test_shipped_controls_are_crs_references_but_not_geometry_controls() -> None:
    evidence = load_ngii_control_evidence()

    assert len(evidence.controls) == 4
    assert evidence.destroyed_control_count == 4
    assert evidence.active_field_control_count == 0
    assert evidence.active_bridge_control_count == 0
    assert evidence.catalogue_crs_reference_allowed
    assert not evidence.digital_map_crs_verified
    assert not evidence.bridge_station_registration_allowed
    assert not evidence.vertical_profile_allowed
    assert "all_public_controls_are_destroyed" in evidence.reasons
    assert "available_sheet_is_post_event_2025" in evidence.reasons


def test_portal_axis_mapping_is_explicit_and_not_assigned_to_delivery() -> None:
    document = _document()

    assert document["coordinate_reference"]["code"] == 5186
    assert document["coordinate_reference"]["portal_field_mapping"] == {
        "minx": "easting_m",
        "miny": "northing_m",
    }
    assert not document["catalogue_product"]["projected_crs_verified_from_delivery"]


def test_destroyed_status_requires_the_portal_source_value() -> None:
    document = _document()
    document["public_controls"][0]["status_source_value"] = "unknown"

    with pytest.raises(NgiiControlEvidenceError, match="source-locked"):
        parse_ngii_control_evidence(document)


def test_axis_swap_or_epsg_guess_is_rejected() -> None:
    document = _document()
    document["coordinate_reference"]["portal_field_mapping"] = {
        "minx": "northing_m",
        "miny": "easting_m",
    }

    with pytest.raises(NgiiControlEvidenceError, match="axis mapping"):
        parse_ngii_control_evidence(document)


def test_application_overclaim_is_rejected() -> None:
    document = _document()
    document["application"]["field_control_use_allowed"] = True

    with pytest.raises(NgiiControlEvidenceError, match="claims"):
        parse_ngii_control_evidence(document)


def test_three_current_bridge_controls_and_verified_delivery_unlock_gate() -> None:
    document = deepcopy(_document())
    product = document["catalogue_product"]
    product["production_year"] = 2024
    product["projected_crs_verified_from_delivery"] = True
    product["download_event_observed"] = True
    product["downloaded_file_count"] = 1
    product["delivery_sha256"] = "a" * 64
    product["delivery_projected_crs"] = "EPSG:5186"
    product["license"] = "verified-test-licence"
    product["license_verified"] = True
    product["temporal_relation"] = "official_same_year_date_unverified"
    product["history_result"] = "authenticated_pre_event_delivery"
    for index, control in enumerate(document["public_controls"][:3], start=8):
        control["status"] = "active"
        control["status_source_value"] = "현존"
        control["installed_on"] = "2024-09-01"
        control["observed_on"] = "2024-09-01"
        control["bridge_landmark"] = f"P{index}"
    application = document["application"]
    application["digital_map_crs_verified"] = True
    application["field_control_use_allowed"] = True
    application["bridge_station_registration_allowed"] = True
    application["vertical_profile_allowed"] = True
    application["blocking_reasons"] = []

    evidence = parse_ngii_control_evidence(document)

    assert evidence.active_field_control_count == 3
    assert evidence.active_bridge_control_count == 3
    assert evidence.bridge_station_registration_allowed
    assert evidence.vertical_profile_allowed


def test_audit_proves_correct_axis_mapping_and_zero_runtime_impact() -> None:
    pytest.importorskip("pyproj")

    report = build_report()

    assert report["crs_audit"]["correct_mapping_inside_scene_count"] == 4
    assert report["crs_audit"]["swapped_mapping_inside_scene_count"] == 0
    assert report["source_manifest_link"]["matches"]
    assert report["qualification"]["destroyed_control_count"] == 4
    assert not report["qualification"]["bridge_station_registration_allowed"]
    assert report["runtime_impact"]["scene_vertices_modified"] == 0
    assert not report["runtime_impact"]["frame_path_changed"]
    assert report["passed"]


def test_schema_locks_event_crs_and_runtime_non_application() -> None:
    schema = json.loads(
        Path("assets/ngii_public_control_evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["target_event_date"]["const"] == "2024-10-05"
    assert schema["$defs"]["coordinate_reference"]["properties"]["code"][
        "const"
    ] == 5186
    assert schema["$defs"]["thresholds"]["properties"][
        "minimum_active_bridge_controls"
    ]["minimum"] == 3
    assert schema["$defs"]["application"]["properties"][
        "scene_vertices_modified"
    ]["const"] == 0


def test_schema_accepts_the_shipped_evidence() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        Path("assets/ngii_public_control_evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(_document())
