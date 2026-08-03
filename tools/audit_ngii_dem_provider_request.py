"""Audit the manual NGII DEM provider-request package without submitting it."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST = Path("assets/yeouido_ngii_dem_provider_request.json")
DEFAULT_METADATA_REPORT = Path(
    "docs/validation/ngii_dem_v1/ngii_dem_metadata_report.json"
)
DEFAULT_OUTPUT = Path(
    "docs/validation/ngii_dem_v1/ngii_dem_provider_request_report.json"
)
EXPECTED_FALLBACK_SHEETS = {
    "37608087": "서울087",
    "37608088": "서울088",
    "37608089": "서울089",
    "37608097": "서울097",
    "37608098": "서울098",
    "37608099": "서울099",
}
REQUIRED_DELIVERABLES = {
    "provider raster package",
    "explicit projected CRS or EPSG evidence",
    "vertical datum and height type",
    "grid spacing",
    "NoData value",
    "accuracy RMSE LE90 or equivalent quality document",
    "production and acquisition dates",
    "provider package checksum or official receipt",
    "local derived use and redistribution terms",
}
FORBIDDEN_PERSONAL_KEYS = {
    "name",
    "contact",
    "email",
    "phone",
    "dob",
    "birth",
    "applicant_name",
    "이름",
    "연락처",
    "이메일",
    "전화번호",
    "생년월일",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _collect_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key).lower() for key in value}
        for nested in value.values():
            keys.update(_collect_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_collect_keys(nested))
        return keys
    return set()


def build_report(
    request_path: Path = DEFAULT_REQUEST,
    metadata_report_path: Path = DEFAULT_METADATA_REPORT,
) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_report_path.read_text(encoding="utf-8"))
    submission = request.get("submission", {})
    preferred = request.get("request_scope", {}).get("preferred_product", {})
    fallback = request.get("request_scope", {}).get("fallback_legacy_sheets", [])
    fallback_sheets = {
        sheet.get("sheet_id"): sheet.get("sheet_name") for sheet in fallback
    }
    metadata_sheets = {
        sheet["sheet_id"]: sheet["sheet_name"]
        for sheet in metadata["legacy_lidar_candidates"]["sheets"]
    }
    deliverables = set(request.get("required_deliverables", []))
    personal_keys = sorted(_collect_keys(request) & FORBIDDEN_PERSONAL_KEYS)
    metadata_link = request.get("evidence", {})
    metadata_link_matches = bool(
        metadata_link.get("metadata_audit") == _display_path(metadata_report_path)
        and metadata_link.get("metadata_audit_sha256")
        == _digest(metadata_report_path)
    )
    official_route_valid = bool(
        submission.get("primary_route")
        == "https://www.data.go.kr/tcs/dor/insertDataOfferReqstProcssView.do"
        and submission.get("provider") == "국토교통부 국토지리정보원"
    )
    preferred_valid = bool(
        preferred.get("sheet_id") == "37608"
        and preferred.get("production_year") == 2024
        and preferred.get("product") == "공개DEM"
    )
    fallback_valid = bool(
        fallback_sheets == EXPECTED_FALLBACK_SHEETS
        and metadata_sheets == EXPECTED_FALLBACK_SHEETS
    )
    deliverables_complete = deliverables == REQUIRED_DELIVERABLES
    privacy_safe = bool(
        not personal_keys
        and submission.get("applicant_personal_information_stored") is False
    )
    external_submission_authorized = submission.get("authorized") is True
    external_submission_performed = submission.get("performed") is True
    portal_authentication_pending = bool(
        submission.get("authentication_required") is True
        and submission.get("authentication_completed") is False
    )
    payload_ready = bool(
        request.get("status")
        in {
            "ready_for_manual_submission_not_submitted",
            "submission_authorized_waiting_for_portal_authentication",
        }
        and official_route_valid
        and preferred_valid
        and fallback_valid
        and deliverables_complete
        and metadata_link_matches
        and privacy_safe
        and not external_submission_performed
        and request.get("acceptance_gate", {}).get("scene_merge_allowed") is False
        and request.get("acceptance_gate", {}).get("scene_vertices_modified") == 0
    )
    blockers: list[str] = []
    if not official_route_valid:
        blockers.append("official public-data request route or provider is invalid")
    if not preferred_valid:
        blockers.append("preferred 2024 Seoul 37608 DEM request is incomplete")
    if not fallback_valid:
        blockers.append("fallback sheet set does not match the metadata audit")
    if not deliverables_complete:
        blockers.append("required provider evidence list is incomplete")
    if not metadata_link_matches:
        blockers.append("metadata audit link or checksum does not match")
    if not privacy_safe:
        blockers.append("request package contains a personal-information field")
    if not external_submission_authorized:
        blockers.append("external submission has not been authorized")
    if external_submission_performed:
        blockers.append("repository package unexpectedly claims external submission")
    blockers.extend(request.get("acceptance_gate", {}).get("blocking_reasons", []))

    return {
        "schema_version": 1,
        "stage": "V1-11e",
        "request_asset": _display_path(request_path),
        "request_sha256": _digest(request_path),
        "metadata_audit": _display_path(metadata_report_path),
        "metadata_audit_sha256": _digest(metadata_report_path),
        "checks": {
            "official_route_valid": official_route_valid,
            "preferred_2024_product_valid": preferred_valid,
            "fallback_sheets_match_metadata": fallback_valid,
            "fallback_sheet_count": len(fallback_sheets),
            "required_deliverables_complete": deliverables_complete,
            "metadata_link_matches": metadata_link_matches,
            "personal_information_keys_found": personal_keys,
            "privacy_safe": privacy_safe,
        },
        "submission_payload_ready": payload_ready,
        "ready_for_manual_submission": payload_ready,
        "external_submission_authorized": external_submission_authorized,
        "external_submission_performed": external_submission_performed,
        "portal_authentication_pending": portal_authentication_pending,
        "scene_merge_allowed": False,
        "scene_vertices_modified": 0,
        "blocking_reasons": blockers,
        "next_evidence_gate": (
            "A user-reviewed external submission and provider delivery are still required. "
            "Verify the raster header, checksum, license, plan RMSE <= 0.25 m, and "
            "vertical uncertainty <= 0.10 m before any scene merge."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument(
        "--metadata-report", type=Path, default=DEFAULT_METADATA_REPORT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(arguments.request, arguments.metadata_report)
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: ready_for_manual_submission="
        f"{report['ready_for_manual_submission']}, "
        f"scene_merge_allowed={report['scene_merge_allowed']}"
    )


if __name__ == "__main__":
    main()
