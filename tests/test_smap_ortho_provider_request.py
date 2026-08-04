"""V2-2f keeps the S-Map enquiry honest and the event-day gate shut.

Two things are worth guarding here. The first is that the enquiry stays tied to
the report it cites: if V2-2e grows a new gate, or its checksum moves, the
package must not keep claiming it covers the open questions. The second is the
floor guarantee — a favourable acquisition date must not, on its own, unlock
event-day marking classification, because the independent check point is still
missing and no reply can conjure one.
"""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from tools.audit_smap_ortho_provider_request import (
    DOWNSTREAM_GATES,
    PROVIDER_ANSWERABLE_GATES,
    REQUIRED_DELIVERABLES,
    STATUS_FLAGS,
    build_report,
    evaluate_reply,
)


REQUEST = Path("assets/yeouido_smap_ortho_provider_request.json")
REPORT = Path(
    "docs/validation/road_marking_registration_v2/"
    "smap_2024_provider_request_report.json"
)
REGISTRATION = Path(
    "docs/validation/road_marking_registration_v2/smap_2024_registration_report.json"
)
SCHEMA = Path("assets/smap_ortho_provider_reply.schema.json")
ARCHIVE_REQUEST = Path(
    "docs/validation/road_marking_registration_v2/ARCHIVE_REQUEST.md"
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _favourable_reply() -> dict:
    """A reply that answers everything the provider could possibly answer."""

    return {
        "schema_version": 1,
        "target_event_date": "2024-10-05",
        "received_at": "2026-08-20",
        "channel": "smap_qna_board",
        "reply_reference": "307",
        "responding_department": "공간정보과",
        "coverage": {
            "bbox_epsg5186_m": [194044.0, 547224.0, 194428.0, 547608.0],
            "covers_requested_bbox": True,
            "composition": "single_sortie",
        },
        "acquisition": {
            "date_earliest": "2024-09-28",
            "date_latest": "2024-09-28",
            "basis": "attached_document",
            "source_agency": "서울특별시",
            "notes": None,
        },
        "positional_accuracy": {
            "document_provided": True,
            "independent_check_point_residual_m": 0.4,
            "metric": "RMSE",
            "orthorectification_reference_surface": "항공 LiDAR DSM",
        },
        "use_scope": {
            "tile_service_local_research_use_allowed": True,
            "redistribution_allowed": False,
            "attribution_required": "출처 : 서울시 S-Map(https://smap.seoul.go.kr)",
            "notes": None,
        },
        "applicant_personal_information_recorded": False,
    }


def _write(tmp_path: Path, name: str, document: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_shipped_request_is_ready_but_unsubmitted() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert request["status"] == "ready_for_manual_submission_not_submitted"
    assert request["submission"]["authorized"] is False
    assert request["submission"]["performed"] is False
    assert request["submission"]["submission_is_user_gated"] is True
    assert request["submission"]["credentials_entered_by_agent"] is False
    assert request["submission"]["provider_attachments_downloaded_by_agent"] is False
    assert request["submission"]["applicant_personal_information_stored"] is False
    assert set(request["required_deliverables"]) == REQUIRED_DELIVERABLES
    assert request["acceptance_gate"]["event_date_marking_classification_allowed"] is False
    assert request["acceptance_gate"]["scene_vertices_modified"] == 0
    assert request["evidence"]["registration_report_sha256"] == _digest(REGISTRATION)


def test_the_committed_report_never_unlocks_the_event_day_reading() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["ready_for_manual_submission"]
    assert report["submission_payload_ready"]
    assert not report["external_submission_authorized"]
    assert not report["external_submission_performed"]
    assert not report["event_date_marking_classification_allowed"]
    assert not report["reply_outcome"]["reply_received"]
    assert report["scene_vertices_modified"] == 0
    assert report["runtime_geometry_changed_by_this_stage"] is False
    assert report["checks"]["personal_information_keys_found"] == []
    assert report["checks"]["gate_classification_complete"]
    assert report["request_sha256"] == _digest(REQUEST)
    assert report["registration_report_sha256"] == _digest(REGISTRATION)


def test_every_open_gate_has_a_question_and_no_question_is_stale() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["checks"]["open_provider_gates"] == sorted(
        PROVIDER_ANSWERABLE_GATES
    )
    assert report["checks"]["open_gates_without_a_question"] == []
    assert report["checks"]["stale_questions"] == []
    assert report["checks"]["questions_well_formed"]
    assert report["checks"]["question_count"] >= len(PROVIDER_ANSWERABLE_GATES)


def test_gate_classifications_are_disjoint() -> None:
    # Overlapping sets would let a provider-answerable gate hide in the status
    # flags, where nothing requires a question for it.
    assert not PROVIDER_ANSWERABLE_GATES & DOWNSTREAM_GATES
    assert not PROVIDER_ANSWERABLE_GATES & STATUS_FLAGS
    assert not DOWNSTREAM_GATES & STATUS_FLAGS


def test_a_new_unclassified_gate_blocks_the_package(tmp_path: Path) -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    registration["application_gates"]["radiometric_calibration_confirmed"] = False
    tampered = _write(tmp_path, "registration.json", registration)

    report = build_report(REQUEST, tampered)

    assert report["checks"]["unclassified_gates"] == [
        "radiometric_calibration_confirmed"
    ]
    assert not report["checks"]["gate_classification_complete"]
    assert not report["submission_payload_ready"]


def test_a_changed_registration_report_breaks_the_checksum_link(
    tmp_path: Path,
) -> None:
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    registration["local_acquisition"]["tile_count"] = 35
    tampered = _write(tmp_path, "registration.json", registration)

    report = build_report(REQUEST, tampered)

    assert not report["checks"]["registration_link_matches"]
    assert not report["submission_payload_ready"]


def test_a_personal_information_key_blocks_the_package(tmp_path: Path) -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["submission"]["email"] = "redacted"
    tampered = _write(tmp_path, "request.json", request)

    report = build_report(tampered, REGISTRATION)

    assert report["checks"]["personal_information_keys_found"] == ["email"]
    assert not report["checks"]["privacy_safe"]
    assert not report["submission_payload_ready"]


def test_a_favourable_reply_opens_the_marking_gate(tmp_path: Path) -> None:
    reply = _write(tmp_path, "reply.json", _favourable_reply())

    report = build_report(REQUEST, REGISTRATION, reply)

    assert report["reply_outcome"]["acquisition_date_confirmed"]
    assert report["reply_outcome"]["event_date_applicability_confirmed"]
    assert report["reply_outcome"]["independent_spatial_check_passes"]
    assert report["event_date_marking_classification_allowed"]
    assert report["reply_outcome"]["reasons"] == []


def test_a_date_after_the_event_keeps_the_gate_shut() -> None:
    reply = _favourable_reply()
    reply["acquisition"]["date_latest"] = "2024-11-20"

    outcome = evaluate_reply(reply)

    assert outcome["acquisition_date_confirmed"]
    assert not outcome["event_date_applicability_confirmed"]
    assert any("after the 2024-10-05 event" in reason for reason in outcome["reasons"])


def test_a_mosaic_is_judged_on_its_latest_frame() -> None:
    # An early first frame must not carry a mosaic that was completed after the
    # event, which is the whole reason the gate reads date_latest.
    reply = _favourable_reply()
    reply["coverage"]["composition"] = "multi_date_mosaic"
    reply["acquisition"]["date_earliest"] = "2024-09-28"
    reply["acquisition"]["date_latest"] = "2024-10-30"

    outcome = evaluate_reply(reply)

    assert not outcome["event_date_applicability_confirmed"]


def test_a_confirmed_date_alone_does_not_unlock_marking(tmp_path: Path) -> None:
    # The floor guarantee. This is the realistic outcome: the operator can state
    # the capture date but has no published check point for the crop.
    reply = _favourable_reply()
    reply["positional_accuracy"] = {
        "document_provided": False,
        "independent_check_point_residual_m": None,
        "metric": None,
        "orthorectification_reference_surface": None,
    }
    path = _write(tmp_path, "reply.json", reply)

    report = build_report(REQUEST, REGISTRATION, path)

    assert report["reply_outcome"]["event_date_applicability_confirmed"]
    assert not report["reply_outcome"]["independent_spatial_check_passes"]
    assert not report["event_date_marking_classification_allowed"]


def test_a_residual_above_one_metre_keeps_the_gate_shut() -> None:
    reply = _favourable_reply()
    reply["positional_accuracy"]["independent_check_point_residual_m"] = 1.4

    outcome = evaluate_reply(reply)

    assert not outcome["independent_spatial_check_passes"]


def test_a_reply_that_misses_the_crop_confirms_nothing() -> None:
    reply = _favourable_reply()
    reply["coverage"]["covers_requested_bbox"] = False

    outcome = evaluate_reply(reply)

    assert not outcome["acquisition_date_confirmed"]
    assert not outcome["event_date_applicability_confirmed"]


def test_a_date_without_a_basis_confirms_nothing() -> None:
    reply = _favourable_reply()
    reply["acquisition"]["basis"] = "not_provided"

    outcome = evaluate_reply(reply)

    assert not outcome["acquisition_date_confirmed"]
    assert not outcome["event_date_applicability_confirmed"]


def test_the_schema_accepts_a_favourable_reply_and_rejects_stored_identity() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    validator.validate(_favourable_reply())

    with_identity = copy.deepcopy(_favourable_reply())
    with_identity["applicant_name"] = "redacted"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(with_identity)


def test_no_provider_reply_is_committed() -> None:
    # The audit must stay blocked because nothing has been received, not because
    # a placeholder was left lying around pretending to be a reply.
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert request["reply"]["received"] is False
    assert request["reply"]["asset"] is None
    assert request["reply"]["schema"] == SCHEMA.as_posix()


def test_the_enquiry_draft_records_the_verified_route_and_its_limits() -> None:
    text = ARCHIVE_REQUEST.read_text(encoding="utf-8")

    assert "https://smap.seoul.go.kr/guide/qna.html" in text
    assert "ortho_drone_25cm_2024" in text
    assert "EPSG:5186" in text
    assert "2024-10-05" in text
    # The two honest caveats that stop the 2025-12-03 notice being read as a
    # blanket permission for what V2-2e actually did.
    assert "지도 화면 캡처" in text
    assert "공공누리 제4유형" in text
