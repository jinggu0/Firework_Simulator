from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from simulator.ngii_delivery import (
    NgiiDeliveryError,
    load_ngii_delivery_receipt,
    parse_ngii_delivery_receipt,
    validate_delivery_packages,
    validate_import_sources,
)
from tools.audit_ngii_delivery_readiness import build_report


ASSET = Path("assets/yeouido_ngii_delivery_receipt.json")


def _document() -> dict:
    return json.loads(ASSET.read_text(encoding="utf-8"))


def _verified_document(raw: bytes = b"verified dxf bytes") -> dict:
    document = deepcopy(_document())
    digest = sha256(raw).hexdigest()
    delivery = document["delivery"]
    delivery.update(
        {
            "status": "acquired",
            "authenticated_download_confirmed": True,
            "acquired_at": "2026-08-03T21:00:00+09:00",
            "production_year": 2024,
            "projected_crs": "EPSG:5186",
            "projected_crs_evidence": {
                "method": "provider_sidecar",
                "artifact_sha256": "b" * 64,
                "catalogue_or_control_inference": False,
                "detail": "Provider sidecar explicitly states EPSG:5186.",
            },
            "license": {
                "name": "verified local-use terms",
                "url": "https://map.ngii.go.kr/terms",
                "verified": True,
                "allows_local_derived_use": True,
                "redistribution_allowed": False,
            },
            "packages": [
                {"file_name": "delivery.zip", "sha256": "a" * 64, "bytes": 2048}
            ],
            "import_members": [
                {
                    "member_path": "376082447.dxf",
                    "sheet_id": "376082447",
                    "format": "DXF",
                    "sha256": digest,
                    "bytes": len(raw),
                }
            ],
        }
    )
    document["application"]["import_allowed"] = True
    document["application"]["blocking_reasons"] = []
    return document


def test_shipped_receipt_is_safe_but_not_delivery_verified() -> None:
    receipt = load_ngii_delivery_receipt()

    assert not receipt.verified
    assert not receipt.import_members
    assert receipt.production_year is None
    assert receipt.projected_crs is None
    assert "authenticated_download_not_confirmed" in receipt.reasons
    assert "required_sheet_coverage_incomplete" in receipt.reasons


def test_event_sheet_request_is_exact_and_seoul2447_is_required() -> None:
    document = _document()
    document["request"]["sheets"][0]["required_for_current_stage"] = False

    with pytest.raises(NgiiDeliveryError, match="sheet request"):
        parse_ngii_delivery_receipt(document)


def test_catalogue_or_control_crs_inference_is_rejected() -> None:
    document = _verified_document()
    document["delivery"]["projected_crs_evidence"][
        "catalogue_or_control_inference"
    ] = True
    document["application"]["import_allowed"] = False
    document["application"]["blocking_reasons"] = [
        "projected_crs_not_verified_from_delivery"
    ]

    receipt = parse_ngii_delivery_receipt(document)

    assert not receipt.verified
    assert receipt.reasons == ("projected_crs_not_verified_from_delivery",)


def test_application_overclaim_is_rejected() -> None:
    document = _document()
    document["application"]["import_allowed"] = True

    with pytest.raises(NgiiDeliveryError, match="import claim"):
        parse_ngii_delivery_receipt(document)


def test_verified_receipt_accepts_only_checksum_locked_dxf() -> None:
    raw = b"verified dxf bytes"
    receipt = parse_ngii_delivery_receipt(_verified_document(raw))

    validate_import_sources(receipt, [("renamed.dxf", raw)])

    with pytest.raises(NgiiDeliveryError, match="missing=1, unexpected=1"):
        validate_import_sources(receipt, [("376082447.dxf", raw + b"tampered")])


def test_download_package_is_bound_by_name_hash_and_size(tmp_path: Path) -> None:
    raw = b"download archive bytes"
    document = _verified_document()
    document["delivery"]["packages"] = [
        {
            "file_name": "delivery.zip",
            "sha256": sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    ]
    receipt = parse_ngii_delivery_receipt(document)
    package = tmp_path / "delivery.zip"
    package.write_bytes(raw)

    validate_delivery_packages(receipt, [package])

    package.write_bytes(raw + b"tampered")
    with pytest.raises(NgiiDeliveryError, match="receipt checksum"):
        validate_delivery_packages(receipt, [package])

    package.write_bytes(raw)
    extra = tmp_path / "unlisted.zip"
    extra.write_bytes(b"empty archive placeholder")
    with pytest.raises(NgiiDeliveryError, match="absent from the receipt"):
        validate_delivery_packages(receipt, [package, extra])


def test_duplicate_import_member_hash_is_rejected() -> None:
    document = _verified_document()
    duplicate = deepcopy(document["delivery"]["import_members"][0])
    duplicate["member_path"] = "duplicate.dxf"
    document["delivery"]["import_members"].append(duplicate)

    with pytest.raises(NgiiDeliveryError, match="hashes must be unique"):
        parse_ngii_delivery_receipt(document)


def test_readiness_report_separates_safety_from_completion() -> None:
    report = build_report()

    assert report["safety_gate_passed"]
    assert not report["stage_complete"]
    assert not report["delivery"]["verified"]
    assert report["runtime_impact"]["scene_vertices_modified"] == 0
    assert not report["runtime_impact"]["frame_path_changed"]
    assert report["enforcement"]["exact_dxf_member_set_required"]
    assert report["enforcement"]["exact_package_file_set_required"]


def test_schema_accepts_the_shipped_pending_receipt() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        Path("assets/ngii_delivery_receipt.schema.json").read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(_document())
