"""V3-1e verifies the grade-A route without inventing the data it would supply.

The register is the only source that publishes both a building's height and its
above-ground floor count, so it is the only way the nine assumed facade floor
heights and the 12.0 m untagged default stop being assumptions. What can be
checked today is that the route is real and correctly recorded; what cannot is
anything derived from records nobody has retrieved.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.audit_building_register_readiness import (
    MINIMUM_DERIVATION_SAMPLE,
    PLAUSIBLE_FLOOR_HEIGHT_M,
    build_report,
    derive_from_records,
    find_credentials,
)


REQUEST = Path("assets/yeouido_building_register_request.json")
REPORT = Path("docs/validation/facade_modules_v3/building_register_report.json")
BUILDING_TAGS = Path("assets/yeouido_building_osm_2024-10-05.json")


def _records(count: int, height: float = 40.0, floors: float = 12.0) -> list[dict]:
    return [
        {
            "heit": str(height),
            "grndFlrCnt": str(floors),
            "mainPurpsCdNm": "업무시설",
        }
        for _ in range(count)
    ]


def _write(tmp_path: Path, name: str, document: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_route_is_recorded_as_verified_not_assumed() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    route = request["verified_route"]

    assert route["base_url"] == "https://apis.data.go.kr/1613000/BldRgstHubService"
    assert route["endpoint"] == "/getBrTitleInfo"
    assert route["browser_verified_utc"]
    # The terms are what make this different from the earlier provider requests:
    # nothing needs to be negotiated, only a key issued.
    assert route["terms"]["cost"] == "무료"
    assert route["terms"]["use_scope"] == "이용허락범위 제한 없음"
    assert "자동승인" in route["terms"]["approval"]


def test_the_codes_reconstruct_the_verified_legal_dong_code() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    parameters = request["request_parameters"]

    assert parameters["sigunguCd"] == "11560"
    assert parameters["bjdongCd"] == "11000"
    assert (
        parameters["sigunguCd"] + parameters["bjdongCd"]
        == parameters["code_basis"]["bjdong_code_10_digit"]
        == "1156011000"
    )
    assert "code.go.kr" in parameters["code_basis"]["source"]
    # The split is derived, so the package must say it is unconfirmed rather
    # than presenting it as verified alongside the code itself.
    assert "confirmed against the first non-empty response" in (
        parameters["code_basis"]["split_rule"]
    )


def test_the_join_assessment_matches_the_dated_snapshot() -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    assessment = request["join_assessment"]
    snapshot = json.loads(BUILDING_TAGS.read_text(encoding="utf-8"))
    untagged = [
        element
        for element in snapshot["elements"]
        if not element["tags"].get("height")
        and not element["tags"].get("building:levels")
    ]

    assert assessment["untagged_height_ways"] == len(untagged) == 456
    assert assessment["untagged_ways_with_addr_housenumber"] == sum(
        1 for element in untagged if element["tags"].get("addr:housenumber")
    )
    assert assessment["untagged_ways_with_a_name"] == sum(
        1 for element in untagged if element["tags"].get("name")
    )


def test_the_committed_report_is_ready_but_unlocks_nothing() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["application_gates"]["route_ready_for_key_issue"]
    assert not report["application_gates"]["register_data_recorded"]
    assert not report["application_gates"]["grade_a_upgrade_available"]
    assert report["application_gates"]["scene_vertices_modified"] == 0
    assert report["checks"]["no_credential_committed"]
    assert report["derivation"] is None
    assert any(
        "no register response is recorded" in reason
        for reason in report["blocking_reasons"]
    )


def test_a_committed_service_key_blocks_the_package(tmp_path: Path) -> None:
    request = json.loads(REQUEST.read_text(encoding="utf-8"))
    request["credential_handling"]["serviceKey"] = (
        "Ab3dEf6hIj9lMn2pQr5tUv8xYz1cDe4gHi7kLm0oPq3sTu6wXy9zBc2eFg5h"
    )
    tampered = _write(tmp_path, "request.json", request)

    report = build_report(tampered, None)

    assert report["checks"]["credential_like_values_found"]
    assert not report["checks"]["no_credential_committed"]
    assert not report["application_gates"]["route_ready_for_key_issue"]


def test_the_credential_scan_ignores_ordinary_prose() -> None:
    # A guard that fires on normal text would be turned off, so it must stay
    # quiet on the package it actually ships with.
    request = json.loads(REQUEST.read_text(encoding="utf-8"))

    assert find_credentials(request) == []
    assert find_credentials({"note": "heit divided by grndFlrCnt gives a floor height"}) == []
    assert find_credentials({"serviceKey": ""}) == []


def test_the_derivation_turns_official_pairs_into_floor_heights() -> None:
    derivation = derive_from_records(_records(40, height=48.0, floors=16.0))

    assert derivation["records_usable_for_derivation"] == 40
    assert derivation["floor_height_median_m"] == 3.0
    assert derivation["height_median_m"] == 48.0
    assert derivation["derivation_sample_sufficient"]
    assert derivation["floor_height_by_main_purpose"]["업무시설"]["sample"] == 40


def test_implausible_pairs_are_excluded_rather_than_averaged_in() -> None:
    # A 200 m building recorded as one storey is a bad pairing, not a 200 m
    # floor. Averaging it in would move the median it is supposed to inform.
    low, high = PLAUSIBLE_FLOOR_HEIGHT_M
    records = _records(30, height=45.0, floors=15.0) + [
        {"heit": "200.0", "grndFlrCnt": "1", "mainPurpsCdNm": "업무시설"},
        {"heit": "0", "grndFlrCnt": "10", "mainPurpsCdNm": "업무시설"},
        {"heit": "30.0", "grndFlrCnt": "", "mainPurpsCdNm": "업무시설"},
    ]

    derivation = derive_from_records(records)

    assert derivation["record_count"] == 33
    assert derivation["records_usable_for_derivation"] == 30
    assert derivation["rejected"]["implausible_floor_height"] == 1
    assert derivation["rejected"]["no_height"] == 1
    assert derivation["rejected"]["no_floor_count"] == 1
    assert derivation["floor_height_median_m"] == 3.0
    assert not low <= 200.0 <= high


def test_a_small_or_unconfirmed_response_does_not_unlock_the_upgrade(
    tmp_path: Path,
) -> None:
    thin = _write(
        tmp_path,
        "thin.json",
        {"parameter_split_confirmed": True, "records": _records(5)},
    )
    unconfirmed = _write(
        tmp_path,
        "unconfirmed.json",
        {
            "parameter_split_confirmed": False,
            "records": _records(MINIMUM_DERIVATION_SAMPLE + 10),
        },
    )

    assert not build_report(REQUEST, thin)["application_gates"][
        "grade_a_upgrade_available"
    ]
    assert not build_report(REQUEST, unconfirmed)["application_gates"][
        "grade_a_upgrade_available"
    ]


def test_a_sufficient_confirmed_response_does_unlock_the_upgrade(
    tmp_path: Path,
) -> None:
    # Proves the gate is not simply wired shut.
    response = _write(
        tmp_path,
        "response.json",
        {
            "parameter_split_confirmed": True,
            "records": _records(MINIMUM_DERIVATION_SAMPLE + 10, 45.0, 15.0),
        },
    )

    report = build_report(REQUEST, response)

    assert report["application_gates"]["grade_a_upgrade_available"]
    assert report["derivation"]["floor_height_median_m"] == 3.0
    assert report["application_gates"]["scene_vertices_modified"] == 0
