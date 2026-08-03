"""Evidence gate for NGII public controls and authenticated map delivery.

The official portal exposes historic public-control coordinates around the
event area, but a destroyed marker is not a usable field control and an EPSG
code attached to that marker is not proof of an undelivered map file's CRS.
This module keeps those evidence scopes separate before bridge geometry can be
registered or elevated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NGII_CONTROL_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "assets" / "yeouido_ngii_public_controls_2017.json"
)
EVENT_DATE = date(2024, 10, 5)


class NgiiControlEvidenceError(ValueError):
    """Raised when survey-control evidence is malformed or overclaimed."""


@dataclass(frozen=True, slots=True)
class PublicSurveyControl:
    control_id: str
    point_name: str
    status: str
    easting_m: float
    northing_m: float
    orthometric_height_m: float
    observed_on: date
    bridge_landmark: str | None


@dataclass(frozen=True, slots=True)
class NgiiControlEvidence:
    document: Mapping[str, Any]
    controls: tuple[PublicSurveyControl, ...]
    destroyed_control_count: int
    active_field_control_count: int
    active_bridge_control_count: int
    catalogue_crs_reference_allowed: bool
    digital_map_crs_verified: bool
    bridge_station_registration_allowed: bool
    vertical_profile_allowed: bool
    reasons: tuple[str, ...]


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise NgiiControlEvidenceError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise NgiiControlEvidenceError(f"{field} must be finite")
    return result


def _iso_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise NgiiControlEvidenceError(f"{field} must be an ISO-8601 date") from error


def parse_ngii_control_evidence(
    document: Mapping[str, Any],
) -> NgiiControlEvidence:
    if document.get("schema_version") != 1:
        raise NgiiControlEvidenceError("unsupported NGII control schema")
    if document.get("target_event_date") != EVENT_DATE.isoformat():
        raise NgiiControlEvidenceError("NGII controls must target 2024-10-05")
    _iso_date(document.get("observed_at"), "observed_at")

    source = document.get("source")
    if not isinstance(source, Mapping):
        raise NgiiControlEvidenceError("source must be an object")
    source_host = urlparse(str(source.get("source_url", ""))).hostname
    if source_host != "map.ngii.go.kr" or source.get("confidence_grade") not in {
        "A",
        "B",
    }:
        raise NgiiControlEvidenceError("controls require an official NGII A/B source")
    if source.get("source_files_redistributed") is not False:
        raise NgiiControlEvidenceError("source redistribution state must be explicit")

    thresholds = document.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise NgiiControlEvidenceError("thresholds must be an object")
    minimum_bridge_controls = int(thresholds.get("minimum_active_bridge_controls", 0))
    maximum_rmse = _finite(
        thresholds.get("maximum_station_plan_rmse_m", 0.0),
        "maximum station plan rmse",
    )
    maximum_age_days = int(
        thresholds.get("maximum_control_age_days_without_reobservation", 0)
    )
    maximum_source_year = int(thresholds.get("maximum_source_year", 0))
    if (
        minimum_bridge_controls < 3
        or maximum_rmse <= 0.0
        or maximum_age_days <= 0
        or maximum_source_year != EVENT_DATE.year
        or thresholds.get("require_delivery_crs_metadata") is not True
    ):
        raise NgiiControlEvidenceError("control evidence thresholds are incomplete")

    coordinate_reference = document.get("coordinate_reference")
    if not isinstance(coordinate_reference, Mapping):
        raise NgiiControlEvidenceError("coordinate_reference must be an object")
    field_mapping = coordinate_reference.get("portal_field_mapping")
    if (
        coordinate_reference.get("authority") != "EPSG"
        or coordinate_reference.get("code") != 5186
        or coordinate_reference.get("portal_dom_reference_code") != "5186"
        or coordinate_reference.get("epsg_native_axis_order")
        != ["northing_m", "easting_m"]
        or not isinstance(field_mapping, Mapping)
        or field_mapping.get("minx") != "easting_m"
        or field_mapping.get("miny") != "northing_m"
    ):
        raise NgiiControlEvidenceError("EPSG:5186 portal axis mapping is inconsistent")

    raw_controls = document.get("public_controls")
    if not isinstance(raw_controls, Sequence) or isinstance(
        raw_controls, (str, bytes)
    ) or not raw_controls:
        raise NgiiControlEvidenceError("public_controls must be a non-empty array")
    controls: list[PublicSurveyControl] = []
    control_ids: set[str] = set()
    for index, raw in enumerate(raw_controls):
        if not isinstance(raw, Mapping):
            raise NgiiControlEvidenceError(f"control {index} must be an object")
        control_id = str(raw.get("control_id", ""))
        if not control_id or control_id in control_ids:
            raise NgiiControlEvidenceError("control ids must be present and unique")
        control_ids.add(control_id)
        status = str(raw.get("status", ""))
        if status not in {"active", "destroyed", "unknown"}:
            raise NgiiControlEvidenceError(f"control {control_id} has invalid status")
        if status == "destroyed" and raw.get("status_source_value") != "망실":
            raise NgiiControlEvidenceError(
                f"control {control_id} destroyed status is not source-locked"
            )
        observed_on = _iso_date(raw.get("observed_on"), f"control {control_id} observed_on")
        installed_on = _iso_date(
            raw.get("installed_on"), f"control {control_id} installed_on"
        )
        if installed_on > observed_on or observed_on > EVENT_DATE:
            raise NgiiControlEvidenceError(
                f"control {control_id} has inconsistent survey dates"
            )
        easting = _finite(raw.get("easting_m"), f"control {control_id} easting")
        northing = _finite(raw.get("northing_m"), f"control {control_id} northing")
        height = _finite(
            raw.get("orthometric_height_m"),
            f"control {control_id} orthometric height",
        )
        if not (100_000.0 <= easting <= 400_000.0 and 300_000.0 <= northing <= 700_000.0):
            raise NgiiControlEvidenceError(
                f"control {control_id} is outside the declared projected domain"
            )
        landmark_raw = raw.get("bridge_landmark")
        landmark = None if landmark_raw is None else str(landmark_raw).strip()
        if landmark_raw is not None and not landmark:
            raise NgiiControlEvidenceError("bridge_landmark cannot be blank")
        controls.append(
            PublicSurveyControl(
                control_id=control_id,
                point_name=str(raw.get("point_name", "")),
                status=status,
                easting_m=easting,
                northing_m=northing,
                orthometric_height_m=height,
                observed_on=observed_on,
                bridge_landmark=landmark,
            )
        )
    if any(not control.point_name for control in controls):
        raise NgiiControlEvidenceError("every control requires a point name")

    field_controls = tuple(
        control
        for control in controls
        if control.status == "active"
        and (EVENT_DATE - control.observed_on).days <= maximum_age_days
    )
    bridge_controls = tuple(
        control for control in field_controls if control.bridge_landmark is not None
    )

    product = document.get("catalogue_product")
    if not isinstance(product, Mapping):
        raise NgiiControlEvidenceError("catalogue_product must be an object")
    production_year = int(product.get("production_year", 0))
    downloaded_count = int(product.get("downloaded_file_count", 0))
    if production_year < 1900 or downloaded_count < 0:
        raise NgiiControlEvidenceError("catalogue product values are invalid")
    delivery_hash = product.get("delivery_sha256")
    delivery_crs = product.get("delivery_projected_crs")
    delivery_checksum_locked = bool(
        isinstance(delivery_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", delivery_hash)
    )
    delivery_crs_explicit = bool(
        isinstance(delivery_crs, str)
        and re.fullmatch(r"EPSG:[0-9]+", delivery_crs)
    )
    licence_verified = bool(
        product.get("license_verified") and product.get("license")
    )
    catalogue_crs_allowed = bool(controls)
    digital_map_crs_verified = bool(
        product.get("projected_crs_verified_from_delivery")
        and product.get("download_event_observed")
        and downloaded_count > 0
        and delivery_checksum_locked
        and delivery_crs_explicit
        and licence_verified
        and production_year <= maximum_source_year
    )
    station_allowed = bool(
        digital_map_crs_verified
        and len(bridge_controls) >= minimum_bridge_controls
    )
    vertical_allowed = bool(
        station_allowed
        and all(math.isfinite(control.orthometric_height_m) for control in bridge_controls)
    )

    reasons: list[str] = []
    if all(control.status == "destroyed" for control in controls):
        reasons.append("all_public_controls_are_destroyed")
    if not bridge_controls:
        reasons.append("no_control_is_a_seogang_bridge_landmark")
    if downloaded_count == 0 or not product.get("download_event_observed"):
        reasons.append("digital_map_delivery_not_acquired")
    if not delivery_checksum_locked:
        reasons.append("digital_map_delivery_not_checksum_locked")
    if not product.get("projected_crs_verified_from_delivery"):
        reasons.append("digital_map_projected_crs_not_verified_from_delivery")
    if not licence_verified:
        reasons.append("digital_map_license_not_verified")
    if production_year > maximum_source_year:
        reasons.append("available_sheet_is_post_event_2025")
    if product.get("history_result") == "no_history_available":
        reasons.append("no_pre_event_sheet_history_available")

    application = document.get("application")
    if not isinstance(application, Mapping):
        raise NgiiControlEvidenceError("application must be an object")
    expected_claims = {
        "catalogue_crs_reference_allowed": catalogue_crs_allowed,
        "digital_map_crs_verified": digital_map_crs_verified,
        "field_control_use_allowed": bool(field_controls),
        "bridge_station_registration_allowed": station_allowed,
        "vertical_profile_allowed": vertical_allowed,
    }
    if any(application.get(key) is not expected for key, expected in expected_claims.items()):
        raise NgiiControlEvidenceError("application claims do not match control evidence")
    if tuple(application.get("blocking_reasons", ())) != tuple(reasons):
        raise NgiiControlEvidenceError("declared blocking reasons do not match evidence")
    if (
        application.get("scene_vertices_modified") != 0
        or application.get("runtime_frame_path_changed") is not False
    ):
        raise NgiiControlEvidenceError("offline control evidence cannot modify runtime")

    return NgiiControlEvidence(
        document=document,
        controls=tuple(controls),
        destroyed_control_count=sum(
            control.status == "destroyed" for control in controls
        ),
        active_field_control_count=len(field_controls),
        active_bridge_control_count=len(bridge_controls),
        catalogue_crs_reference_allowed=catalogue_crs_allowed,
        digital_map_crs_verified=digital_map_crs_verified,
        bridge_station_registration_allowed=station_allowed,
        vertical_profile_allowed=vertical_allowed,
        reasons=tuple(reasons),
    )


def load_ngii_control_evidence(
    path: Path = DEFAULT_NGII_CONTROL_EVIDENCE_PATH,
) -> NgiiControlEvidence:
    return parse_ngii_control_evidence(json.loads(path.read_text(encoding="utf-8")))
