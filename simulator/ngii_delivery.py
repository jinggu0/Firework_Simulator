"""Authenticated NGII delivery receipt and imported-file integrity gate.

Catalogue metadata, survey controls, and an authenticated file delivery are
different evidence scopes.  This module accepts only the last of those as
permission to import geometry, and binds every imported DXF byte stream to a
checksum recorded in the delivery receipt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NGII_DELIVERY_RECEIPT_PATH = (
    REPOSITORY_ROOT / "assets" / "yeouido_ngii_delivery_receipt.json"
)
EVENT_DATE = date(2024, 10, 5)
EXPECTED_SHEETS = {
    "376082447": "서울2447",
    "376082448": "서울2448",
    "376082457": "서울2457",
    "376082458": "서울2458",
}
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
EPSG_PATTERN = re.compile(r"EPSG:[0-9]+")


class NgiiDeliveryError(ValueError):
    """Raised when a delivery receipt is malformed or overclaims readiness."""


@dataclass(frozen=True, slots=True)
class DeliveryMember:
    member_path: str
    sheet_id: str
    format: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class NgiiDeliveryReceipt:
    document: Mapping[str, Any]
    import_members: tuple[DeliveryMember, ...]
    production_year: int | None
    projected_crs: str | None
    verified: bool
    reasons: tuple[str, ...]


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise NgiiDeliveryError(f"{field} must be an array")
    return value


def _sha256(value: Any, field: str) -> str:
    result = str(value)
    if not HASH_PATTERN.fullmatch(result):
        raise NgiiDeliveryError(f"{field} must be a lowercase SHA-256")
    return result


def _positive_bytes(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise NgiiDeliveryError(f"{field} must be an integer") from error
    if result <= 0:
        raise NgiiDeliveryError(f"{field} must be positive")
    return result


def _parse_datetime(value: Any, field: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise NgiiDeliveryError(f"{field} must be an ISO-8601 datetime") from error


def parse_ngii_delivery_receipt(document: Mapping[str, Any]) -> NgiiDeliveryReceipt:
    if document.get("schema_version") != 1:
        raise NgiiDeliveryError("unsupported NGII delivery receipt schema")
    if document.get("target_event_date") != EVENT_DATE.isoformat():
        raise NgiiDeliveryError("delivery receipt must target 2024-10-05")
    date.fromisoformat(str(document.get("observed_at")))

    source = document.get("source")
    if not isinstance(source, Mapping):
        raise NgiiDeliveryError("source must be an object")
    if (
        urlparse(str(source.get("portal_url", ""))).hostname != "map.ngii.go.kr"
        or source.get("provider") != "국토지리정보원 국토정보플랫폼"
        or source.get("login_required") is not True
        or source.get("source_files_redistributed") is not False
    ):
        raise NgiiDeliveryError("delivery source must be the official login-gated portal")

    request = document.get("request")
    if not isinstance(request, Mapping):
        raise NgiiDeliveryError("request must be an object")
    if (
        request.get("scale_denominator") != 1000
        or request.get("maximum_source_year") != EVENT_DATE.year
        or request.get("accepted_formats") != ["DXF", "NGI"]
    ):
        raise NgiiDeliveryError("request scale, year, or formats are inconsistent")
    sheets = _sequence(request.get("sheets"), "request.sheets")
    declared_sheets: dict[str, str] = {}
    required_sheet_ids: set[str] = set()
    for raw in sheets:
        if not isinstance(raw, Mapping):
            raise NgiiDeliveryError("every requested sheet must be an object")
        sheet_id = str(raw.get("sheet_id", ""))
        name = str(raw.get("sheet_name", ""))
        if sheet_id in declared_sheets:
            raise NgiiDeliveryError("requested sheet ids must be unique")
        declared_sheets[sheet_id] = name
        if raw.get("required_for_current_stage") is True:
            required_sheet_ids.add(sheet_id)
    if declared_sheets != EXPECTED_SHEETS or required_sheet_ids != {"376082447"}:
        raise NgiiDeliveryError("event-area sheet request is incomplete")

    delivery = document.get("delivery")
    if not isinstance(delivery, Mapping):
        raise NgiiDeliveryError("delivery must be an object")
    status = delivery.get("status")
    if status not in {"not_acquired", "acquired"}:
        raise NgiiDeliveryError("delivery status is invalid")
    authenticated = delivery.get("authenticated_download_confirmed") is True
    acquired_at = delivery.get("acquired_at")
    if acquired_at is not None:
        _parse_datetime(acquired_at, "delivery.acquired_at")

    production_year_raw = delivery.get("production_year")
    production_year = None
    if production_year_raw is not None:
        production_year = int(production_year_raw)
        if production_year < 1900:
            raise NgiiDeliveryError("delivery production year is implausible")

    projected_crs_raw = delivery.get("projected_crs")
    projected_crs = None if projected_crs_raw is None else str(projected_crs_raw).upper()
    if projected_crs is not None and not EPSG_PATTERN.fullmatch(projected_crs):
        raise NgiiDeliveryError("delivery projected CRS must be an explicit EPSG code")

    packages: list[tuple[str, str, int]] = []
    package_names: set[str] = set()
    package_keys: set[tuple[str, int]] = set()
    for index, raw in enumerate(_sequence(delivery.get("packages"), "delivery.packages")):
        if not isinstance(raw, Mapping):
            raise NgiiDeliveryError("every package must be an object")
        file_name = str(raw.get("file_name", "")).strip()
        if (
            not file_name
            or Path(file_name).name != file_name
            or file_name in package_names
        ):
            raise NgiiDeliveryError("delivery package names must be unique basenames")
        package_names.add(file_name)
        digest = _sha256(raw.get("sha256"), f"package {index} sha256")
        byte_count = _positive_bytes(raw.get("bytes"), f"package {index} bytes")
        key = (digest, byte_count)
        if key in package_keys:
            raise NgiiDeliveryError("delivery package hashes must be unique")
        package_keys.add(key)
        packages.append((file_name, digest, byte_count))

    members: list[DeliveryMember] = []
    member_keys: set[tuple[str, int]] = set()
    for index, raw in enumerate(
        _sequence(delivery.get("import_members"), "delivery.import_members")
    ):
        if not isinstance(raw, Mapping):
            raise NgiiDeliveryError("every import member must be an object")
        member_path = str(raw.get("member_path", "")).replace("\\", "/")
        sheet_id = str(raw.get("sheet_id", ""))
        file_format = str(raw.get("format", "")).upper()
        digest = _sha256(raw.get("sha256"), f"import member {index} sha256")
        byte_count = _positive_bytes(raw.get("bytes"), f"import member {index} bytes")
        if (
            not member_path
            or member_path.startswith("/")
            or ".." in Path(member_path).parts
            or sheet_id not in EXPECTED_SHEETS
            or file_format not in {"DXF", "NGI"}
        ):
            raise NgiiDeliveryError(f"import member {index} metadata is invalid")
        key = (digest, byte_count)
        if key in member_keys:
            raise NgiiDeliveryError("import member hashes must be unique")
        member_keys.add(key)
        members.append(
            DeliveryMember(
                member_path=member_path,
                sheet_id=sheet_id,
                format=file_format,
                sha256=digest,
                bytes=byte_count,
            )
        )

    crs_evidence = delivery.get("projected_crs_evidence")
    crs_verified = False
    if isinstance(crs_evidence, Mapping):
        method = crs_evidence.get("method")
        artifact_hash = crs_evidence.get("artifact_sha256")
        crs_verified = bool(
            method in {"embedded_metadata", "provider_sidecar", "provider_receipt"}
            and isinstance(artifact_hash, str)
            and HASH_PATTERN.fullmatch(artifact_hash)
            and crs_evidence.get("catalogue_or_control_inference") is False
            and projected_crs is not None
        )

    license_record = delivery.get("license")
    license_verified = bool(
        isinstance(license_record, Mapping)
        and license_record.get("verified") is True
        and license_record.get("allows_local_derived_use") is True
        and str(license_record.get("name", "")).strip()
        and urlparse(str(license_record.get("url", ""))).scheme in {"http", "https"}
    )
    delivered_sheet_ids = {member.sheet_id for member in members}

    reasons: list[str] = []
    if not authenticated:
        reasons.append("authenticated_download_not_confirmed")
    if status != "acquired" or acquired_at is None:
        reasons.append("delivery_not_acquired")
    if not packages:
        reasons.append("delivery_packages_not_checksum_locked")
    if not members:
        reasons.append("import_members_not_checksum_locked")
    if production_year is None or production_year > EVENT_DATE.year:
        reasons.append("event_compatible_source_year_not_verified")
    if not crs_verified:
        reasons.append("projected_crs_not_verified_from_delivery")
    if not license_verified:
        reasons.append("license_not_verified_for_local_derived_use")
    if not required_sheet_ids.issubset(delivered_sheet_ids):
        reasons.append("required_sheet_coverage_incomplete")

    verified = not reasons
    application = document.get("application")
    if not isinstance(application, Mapping):
        raise NgiiDeliveryError("application must be an object")
    if application.get("import_allowed") is not verified:
        raise NgiiDeliveryError("application import claim does not match delivery")
    if tuple(application.get("blocking_reasons", ())) != tuple(reasons):
        raise NgiiDeliveryError("declared delivery blockers do not match evidence")
    if (
        application.get("scene_vertices_modified") != 0
        or application.get("runtime_frame_path_changed") is not False
    ):
        raise NgiiDeliveryError("pending delivery evidence cannot modify runtime")

    return NgiiDeliveryReceipt(
        document=document,
        import_members=tuple(members),
        production_year=production_year,
        projected_crs=projected_crs,
        verified=verified,
        reasons=tuple(reasons),
    )


def load_ngii_delivery_receipt(
    path: Path = DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
) -> NgiiDeliveryReceipt:
    return parse_ngii_delivery_receipt(json.loads(path.read_text(encoding="utf-8")))


def validate_import_sources(
    receipt: NgiiDeliveryReceipt,
    sources: Iterable[tuple[str, bytes]],
) -> None:
    """Verify that imported DXFs are exactly the receipt-locked byte streams."""

    if not receipt.verified:
        raise NgiiDeliveryError(
            "NGII delivery receipt is not verified: " + ", ".join(receipt.reasons)
        )
    expected = Counter(
        (member.sha256, member.bytes)
        for member in receipt.import_members
        if member.format == "DXF"
    )
    actual = Counter((sha256(raw).hexdigest(), len(raw)) for _, raw in sources)
    if not expected:
        raise NgiiDeliveryError("verified receipt has no DXF import members")
    if actual != expected:
        missing = sum((expected - actual).values())
        unexpected = sum((actual - expected).values())
        raise NgiiDeliveryError(
            "DXF inputs do not match the authenticated delivery receipt "
            f"(missing={missing}, unexpected={unexpected})"
        )


def validate_delivery_packages(
    receipt: NgiiDeliveryReceipt,
    input_paths: Iterable[Path],
) -> None:
    """Bind the raw files supplied to the importer to downloaded packages."""

    if not receipt.verified:
        raise NgiiDeliveryError(
            "NGII delivery receipt is not verified: " + ", ".join(receipt.reasons)
        )
    expected_records = receipt.document["delivery"]["packages"]
    expected = {
        str(record["file_name"]): (str(record["sha256"]), int(record["bytes"]))
        for record in expected_records
    }
    candidates: dict[str, list[Path]] = {name: [] for name in expected}
    supplied_packages: list[Path] = []
    for supplied in input_paths:
        if supplied.is_dir():
            supplied_packages.extend(
                path
                for path in supplied.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".zip", ".dxf", ".ngi"}
            )
        elif supplied.is_file():
            supplied_packages.append(supplied)
    unexpected_names = sorted(
        path.name for path in supplied_packages if path.name not in candidates
    )
    if unexpected_names:
        raise NgiiDeliveryError(
            "input contains packages absent from the receipt: "
            + ", ".join(unexpected_names)
        )
    for path in supplied_packages:
        candidates[path.name].append(path)
    for name, paths in candidates.items():
        if len(paths) != 1:
            raise NgiiDeliveryError(
                f"delivery package {name!r} must resolve exactly once; found {len(paths)}"
            )
        raw = paths[0].read_bytes()
        actual = (sha256(raw).hexdigest(), len(raw))
        if actual != expected[name]:
            raise NgiiDeliveryError(
                f"delivery package {name!r} does not match its receipt checksum"
            )
