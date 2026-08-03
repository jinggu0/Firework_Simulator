from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from tools.audit_ngii_dem_provider_request import (
    EXPECTED_FALLBACK_SHEETS,
    REQUIRED_DELIVERABLES,
    build_report,
)


REQUEST = Path("assets/yeouido_ngii_dem_provider_request.json")
REPORT = Path(
    "docs/validation/ngii_dem_v1/ngii_dem_provider_request_report.json"
)
DEM_REQUEST = Path("assets/yeouido_ngii_dem_request.json")
SOURCE_MANIFEST = Path("assets/yeouido_ngii_1000_source_manifest.json")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_shipped_request_is_complete_private_and_not_submitted() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    fallback = {
        sheet["sheet_id"]: sheet["sheet_name"]
        for sheet in request["request_scope"]["fallback_legacy_sheets"]
    }

    assert request["request_scope"]["preferred_product"]["selection_label"] == (
        "2024 서울 37608"
    )
    assert fallback == EXPECTED_FALLBACK_SHEETS
    assert set(request["required_deliverables"]) == REQUIRED_DELIVERABLES
    assert request["submission"]["applicant_personal_information_stored"] is False
    assert request["submission"]["authorized"] is False
    assert request["submission"]["performed"] is False
    assert request["acceptance_gate"]["scene_merge_allowed"] is False


def test_committed_report_is_ready_but_never_unlocks_scene() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["ready_for_manual_submission"]
    assert report["checks"]["fallback_sheet_count"] == 6
    assert report["checks"]["personal_information_keys_found"] == []
    assert not report["external_submission_authorized"]
    assert not report["external_submission_performed"]
    assert not report["scene_merge_allowed"]
    assert report["scene_vertices_modified"] == 0


def test_request_and_audit_are_checksum_linked_from_existing_manifests() -> None:
    dem_request = json.loads(DEM_REQUEST.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    dem_link = dem_request["acquisition"]["provider_request"]
    manifest_link = manifest["dem_provider_request"]

    assert dem_link["request_sha256"] == _digest(REQUEST)
    assert dem_link["audit_report_sha256"] == _digest(REPORT)
    assert manifest_link["sha256"] == _digest(REQUEST)
    assert manifest_link["audit_report_sha256"] == _digest(REPORT)
    assert manifest["dem_acquisition_request"]["sha256"] == _digest(DEM_REQUEST)


def test_missing_fallback_sheet_blocks_readiness(tmp_path: Path) -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["request_scope"]["fallback_legacy_sheets"].pop()
    tampered = tmp_path / "request.json"
    tampered.write_text(
        json.dumps(request, ensure_ascii=False), encoding="utf-8"
    )

    report = build_report(tampered)

    assert not report["ready_for_manual_submission"]
    assert not report["checks"]["fallback_sheets_match_metadata"]
    assert not report["scene_merge_allowed"]


def test_personal_information_key_blocks_readiness(tmp_path: Path) -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["submission"]["email"] = "redacted"
    tampered = tmp_path / "request.json"
    tampered.write_text(
        json.dumps(request, ensure_ascii=False), encoding="utf-8"
    )

    report = build_report(tampered)

    assert not report["ready_for_manual_submission"]
    assert report["checks"]["personal_information_keys_found"] == ["email"]
    assert not report["scene_merge_allowed"]
