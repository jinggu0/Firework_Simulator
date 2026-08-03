from __future__ import annotations

import json
from pathlib import Path


REQUEST = Path("assets/yeouido_ngii_dem_request.json")


def test_dem_request_targets_the_event_year_and_covering_sheet() -> None:
    document = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert document["target_event_date"] == "2024-10-05"
    assert document["request"] == {
        "sheet_id": "37608",
        "sheet_name": "서울",
        "production_year": 2024,
        "catalogue_candidate_count": 1,
        "selection_label": "2024 서울 37608",
    }
    assert document["source"]["login_required_for_download"]
    assert document["acquisition"]["authenticated_session_confirmed"]
    assert document["acquisition"]["application_prepared"]
    assert document["acquisition"]["terms_accepted"]
    acquisition = document["acquisition"]
    assert acquisition["date_of_birth_handling"] == (
        "user_authorized_browser_session_reuse_not_stored_in_repository"
    )
    assert acquisition["status"] == "blocked_portal_endpoint_not_found_metadata_only"
    assert acquisition["observed_mime_type"] == "text/html"
    assert acquisition["portal_registered_dem_count"] == 0
    assert len(acquisition["metadata_evidence"]["legacy_candidate_sheet_ids"]) == 6
    assert acquisition["metadata_evidence"]["legacy_candidate_accuracy"] is None
    assert len(acquisition["metadata_evidence"]["audit_report_sha256"]) == 64


def test_pending_dem_request_cannot_modify_the_scene() -> None:
    application = json.loads(REQUEST.read_text(encoding="utf-8"))["validation_gate"]

    assert not application["scene_merge_allowed"]
    assert application["scene_vertices_modified"] == 0
    assert application["blocking_reasons"]
