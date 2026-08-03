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


def test_pending_dem_request_cannot_modify_the_scene() -> None:
    application = json.loads(REQUEST.read_text(encoding="utf-8"))["validation_gate"]

    assert not application["scene_merge_allowed"]
    assert application["scene_vertices_modified"] == 0
    assert application["blocking_reasons"]
