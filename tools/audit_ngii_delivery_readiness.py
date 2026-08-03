"""Audit the authenticated NGII delivery and post-event adoption gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from simulator.ngii_delivery import (
    DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
    EXPECTED_SHEETS,
    load_ngii_delivery_receipt,
)


DEFAULT_SOURCE_MANIFEST = Path("assets/yeouido_ngii_1000_source_manifest.json")
DEFAULT_OUTPUT = Path(
    "docs/validation/ngii_delivery_v1/ngii_delivery_readiness_report.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_report(
    receipt_path: Path = DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
) -> dict[str, Any]:
    receipt = load_ngii_delivery_receipt(receipt_path)
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    request = receipt.document["request"]
    delivery = receipt.document["delivery"]
    requested_sheets = {
        sheet["sheet_id"]: sheet["sheet_name"] for sheet in request["sheets"]
    }
    catalogue_sheets = {
        sheet["sheet_id"]: sheet["sheet_name"]
        for sheet in manifest["event_area_sheets"]
    }
    manifest_link = manifest.get("authenticated_delivery_receipt", {})
    receipt_digest = _digest(receipt_path)
    receipt_link_matches = bool(
        manifest_link.get("asset") == _display_path(receipt_path)
        and manifest_link.get("sha256") == receipt_digest
        and manifest_link.get("required_sheet_id") == "376082447"
        and manifest_link.get("maximum_source_year") == 2024
        and manifest_link.get("package_count") == len(delivery["packages"])
        and manifest_link.get("import_member_count") == len(delivery["import_members"])
    )
    blockers = list(receipt.reasons)
    safety_gate_passed = bool(
        requested_sheets == EXPECTED_SHEETS
        and catalogue_sheets == EXPECTED_SHEETS
        and receipt_link_matches
        and request["maximum_source_year"] == 2024
        and receipt.verified
        and not blockers
        and receipt.post_event_authorized
        and not receipt.historical_identity_verified
        and receipt.document["application"]["scene_vertices_modified"] == 0
        and receipt.document["application"]["runtime_frame_path_changed"] is False
    )
    return {
        "schema_version": 2,
        "stage": "V1-10d",
        "target_event_date": receipt.document["target_event_date"],
        "receipt_asset": _display_path(receipt_path),
        "receipt_sha256": receipt_digest,
        "source_manifest": _display_path(source_manifest_path),
        "source_manifest_sha256": _digest(source_manifest_path),
        "source_manifest_link": {
            "asset": manifest_link.get("asset"),
            "expected_sha256": manifest_link.get("sha256"),
            "matches": receipt_link_matches,
        },
        "official_route": {
            "provider": receipt.document["source"]["provider"],
            "portal_url": receipt.document["source"]["portal_url"],
            "login_required": receipt.document["source"]["login_required"],
            "current_repository_has_authenticated_delivery": delivery[
                "authenticated_download_confirmed"
            ],
        },
        "request": {
            "scale_denominator": request["scale_denominator"],
            "maximum_source_year": request["maximum_source_year"],
            "accepted_formats": request["accepted_formats"],
            "requested_sheets": requested_sheets,
            "required_current_stage_sheet": "376082447",
            "required_current_stage_sheets": sorted(EXPECTED_SHEETS),
            "matches_catalogue_sheet_set": requested_sheets == catalogue_sheets,
        },
        "delivery": {
            "status": delivery["status"],
            "package_count": len(delivery["packages"]),
            "import_member_count": len(delivery["import_members"]),
            "production_year": receipt.production_year,
            "projected_crs": receipt.projected_crs,
            "verified": receipt.verified,
            "post_event_authorized": receipt.post_event_authorized,
            "historical_identity_verified": receipt.historical_identity_verified,
            "blocking_reasons": blockers,
        },
        "enforcement": {
            "catalogue_crs_inference_allowed": False,
            "public_control_crs_inference_allowed": False,
            "package_sha256_required": True,
            "exact_package_file_set_required": True,
            "import_member_sha256_and_byte_count_required": True,
            "exact_geometry_member_set_required": True,
            "explicit_delivery_epsg_required": True,
            "local_derived_use_license_required": True,
            "requested_maximum_production_year": 2024,
            "authorized_post_event_maximum_year": 2025,
            "historical_identity_claim_allowed": False,
        },
        "runtime_impact": {
            "scene_vertices_modified": 0,
            "frame_path_changed": False,
            "expected_frame_time_delta_ms": 0.0,
        },
        "safety_gate_passed": safety_gate_passed,
        "stage_complete": receipt.verified,
        "historical_fidelity_complete": receipt.historical_identity_verified,
        "next_evidence_gate": (
            "Use the checksum-locked SHP plan geometry for normalization only. Retain "
            "unknown heights as null, and require separately cited elevation profiles "
            "plus a 2024-10-05 change audit before modifying runtime scene geometry."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_NGII_DELIVERY_RECEIPT_PATH)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(arguments.receipt, arguments.source_manifest)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {arguments.output}: safety_gate_passed="
        f"{report['safety_gate_passed']}, stage_complete={report['stage_complete']}"
    )


if __name__ == "__main__":
    main()
