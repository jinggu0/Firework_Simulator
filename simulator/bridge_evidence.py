"""Evidence gate for bridge and ramp vertical geometry.

Bridge plan geometry is not a vertical survey.  A published clearance is also
not the road-surface elevation unless the matching station and structural
depth are known.  This module keeps those distinctions machine-readable and
prevents an attractive but unsupported height correction from reaching the
runtime scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .provenance import DataRecord


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "assets" / "seogang_bridge_vertical_evidence.json"
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class BridgeEvidenceError(ValueError):
    """Raised when a bridge correction is not supported by registered evidence."""


@dataclass(frozen=True, slots=True)
class VerticalAnchor:
    anchor_id: str
    component: str
    quantity: str
    elevation_el_m: float
    runtime_y_m: float
    source_id: str
    uncertainty_m: float | None


@dataclass(frozen=True, slots=True)
class BridgeApplicationState:
    allowed: bool
    status: str
    registered_profiles: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BridgeEvidence:
    document: Mapping[str, Any]
    scene_asset: str
    scene_sha256: str
    elevation_datum_m: float
    sources: Mapping[str, DataRecord]
    anchors: tuple[VerticalAnchor, ...]
    application: BridgeApplicationState


def _finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BridgeEvidenceError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise BridgeEvidenceError(f"{field} must be finite")
    return result


def _source_records(raw: Any) -> dict[str, DataRecord]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise BridgeEvidenceError("sources must contain at least one record")
    records: dict[str, DataRecord] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise BridgeEvidenceError(f"sources[{index}] must be an object")
        source_id = str(item.get("source_id", ""))
        if not source_id or source_id in records:
            raise BridgeEvidenceError("source ids must be present and unique")
        record = DataRecord.from_dict(item)
        if not record.grade.is_evidence:
            raise BridgeEvidenceError(
                f"source {source_id} must be confidence grade A or B"
            )
        if not _SHA256.fullmatch(record.checksum):
            raise BridgeEvidenceError(
                f"source {source_id} must carry a lowercase sha256 checksum"
            )
        records[source_id] = record
    return records


def _application_state(
    application: Any,
    sources: Mapping[str, DataRecord],
) -> BridgeApplicationState:
    if not isinstance(application, Mapping):
        raise BridgeEvidenceError("vertical_profile_application must be an object")
    status = str(application.get("status", ""))
    profiles = application.get("profiles", [])
    registration = application.get("station_registration")
    thresholds = application.get("thresholds", {})
    reasons: list[str] = []
    if status != "registered":
        reasons.append("application_status_is_not_registered")
    if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)):
        raise BridgeEvidenceError("profiles must be an array")
    if not profiles:
        reasons.append("no_registered_vertical_profile")
    blockers = application.get("blockers", [])
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        raise BridgeEvidenceError("blockers must be an array")
    if blockers:
        reasons.append("declared_blockers_remain")

    maximum_plan_rmse = _finite_number(
        thresholds.get("maximum_station_plan_rmse_m", 0.0),
        "maximum_station_plan_rmse_m",
    )
    maximum_vertical_uncertainty = _finite_number(
        thresholds.get("maximum_vertical_uncertainty_m", 0.0),
        "maximum_vertical_uncertainty_m",
    )
    if maximum_plan_rmse <= 0.0 or maximum_vertical_uncertainty <= 0.0:
        raise BridgeEvidenceError("application thresholds must be positive")
    minimum_controls = int(thresholds.get("minimum_station_control_points", 0))
    maximum_drawing_mpp = _finite_number(
        thresholds.get("maximum_drawing_plan_m_per_pixel", 0.0),
        "maximum_drawing_plan_m_per_pixel",
    )
    accepted_plan_grades = set(thresholds.get("accepted_plan_source_grades", []))
    if minimum_controls < 3 or maximum_drawing_mpp <= 0.0:
        raise BridgeEvidenceError("station evidence thresholds are incomplete")
    if not accepted_plan_grades or not accepted_plan_grades <= {"A", "B"}:
        raise BridgeEvidenceError("accepted plan source grades must be A and/or B")

    event_validation = application.get("event_date_validation")
    if not isinstance(event_validation, Mapping):
        raise BridgeEvidenceError("event_date_validation must be an object")
    manifest_hash = str(event_validation.get("event_state_manifest_sha256", ""))
    if not event_validation.get("event_state_manifest") or not re.fullmatch(
        r"[0-9a-f]{64}", manifest_hash
    ):
        raise BridgeEvidenceError("event state manifest and sha256 are required")
    try:
        event_verified_through = date.fromisoformat(
            str(event_validation.get("verified_through", ""))
        )
    except ValueError as error:
        raise BridgeEvidenceError(
            "event verified-through date must be ISO-8601"
        ) from error
    if (
        event_validation.get("status") != "verified"
        or event_verified_through < date(2024, 10, 5)
    ):
        reasons.append("event_date_structural_history_not_verified")

    controls: Sequence[Any] = []
    if not isinstance(registration, Mapping):
        reasons.append("station_registration_missing")
    else:
        controls = registration.get("control_points", [])
        if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)):
            raise BridgeEvidenceError("station control_points must be an array")
        if len(controls) < minimum_controls:
            reasons.append("fewer_than_three_station_controls")
        plan_grade = str(registration.get("plan_source_confidence_grade", ""))
        if plan_grade not in accepted_plan_grades:
            reasons.append("station_plan_source_grade_not_accepted")
        drawing_mpp = _finite_number(
            registration.get("drawing_plan_m_per_pixel", math.inf),
            "drawing plan metres per pixel",
        )
        if drawing_mpp > maximum_drawing_mpp:
            reasons.append("drawing_plan_resolution_exceeds_threshold")
        rmse = _finite_number(
            registration.get("plan_rmse_m", math.inf), "station plan_rmse_m"
        )
        if rmse > maximum_plan_rmse:
            reasons.append("station_plan_rmse_exceeds_threshold")
        control_stations: list[float] = []
        for index, control in enumerate(controls):
            if not isinstance(control, Mapping):
                raise BridgeEvidenceError(f"station control {index} must be an object")
            control_stations.append(
                _finite_number(control.get("station_m"), f"control {index} station_m")
            )
            world_xz = control.get("world_xz_m", [])
            if (
                not isinstance(world_xz, Sequence)
                or isinstance(world_xz, (str, bytes))
                or len(world_xz) != 2
            ):
                raise BridgeEvidenceError(
                    f"control {index} world_xz_m must contain two values"
                )
            _finite_number(world_xz[0], f"control {index} world x")
            _finite_number(world_xz[1], f"control {index} world z")
        if len(set(control_stations)) != len(control_stations):
            reasons.append("duplicate_station_controls")

    for profile_index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            raise BridgeEvidenceError(f"profile {profile_index} must be an object")
        if not profile.get("profile_id") or not profile.get("component"):
            raise BridgeEvidenceError(
                f"profile {profile_index} needs profile_id and component"
            )
        source_id = str(profile.get("source_id", ""))
        if source_id not in sources:
            reasons.append(f"profile_{profile_index}_source_not_registered")
        samples = profile.get("samples", [])
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise BridgeEvidenceError(f"profile {profile_index} samples must be an array")
        if len(samples) < 2:
            reasons.append(f"profile_{profile_index}_has_fewer_than_two_samples")
        previous_station = -math.inf
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise BridgeEvidenceError(
                    f"profile {profile_index} sample {sample_index} must be an object"
                )
            station = _finite_number(
                sample.get("station_m"),
                f"profile {profile_index} sample {sample_index} station_m",
            )
            if station <= previous_station:
                reasons.append(f"profile_{profile_index}_stations_not_strictly_increasing")
            previous_station = station
            top = sample.get("deck_top_el_m")
            underside = sample.get("deck_underside_el_m")
            depth = sample.get("structural_depth_m")
            if top is None and (underside is None or depth is None):
                reasons.append(
                    f"profile_{profile_index}_sample_{sample_index}_cannot_resolve_deck_top"
                )
            if top is not None:
                _finite_number(top, "deck_top_el_m")
            if underside is not None:
                _finite_number(underside, "deck_underside_el_m")
            if depth is not None and _finite_number(depth, "structural_depth_m") <= 0.0:
                raise BridgeEvidenceError("structural_depth_m must be positive")
            uncertainty = _finite_number(
                sample.get("vertical_uncertainty_m", math.inf),
                "vertical_uncertainty_m",
            )
            if uncertainty < 0.0:
                raise BridgeEvidenceError("vertical uncertainty cannot be negative")
            if uncertainty > maximum_vertical_uncertainty:
                reasons.append(
                    f"profile_{profile_index}_sample_{sample_index}_uncertainty_exceeds_threshold"
                )
            if top is not None and underside is not None and depth is not None:
                if abs(float(top) - (float(underside) + float(depth))) > uncertainty:
                    reasons.append(
                        f"profile_{profile_index}_sample_{sample_index}_vertical_terms_disagree"
                    )

    unique_reasons = tuple(dict.fromkeys(reasons))
    return BridgeApplicationState(
        allowed=not unique_reasons,
        status=status,
        registered_profiles=len(profiles),
        reasons=unique_reasons,
    )


def parse_bridge_evidence(document: Mapping[str, Any]) -> BridgeEvidence:
    """Validate a bridge evidence manifest without accepting implicit defaults."""

    if document.get("schema_version") != 1:
        raise BridgeEvidenceError("unsupported bridge evidence schema version")
    if document.get("target_event_date") != "2024-10-05":
        raise BridgeEvidenceError("bridge evidence must target 2024-10-05")
    if document.get("bridge_id") != "seogang_bridge":
        raise BridgeEvidenceError("this gate accepts only seogang_bridge")

    scene = document.get("scene")
    if not isinstance(scene, Mapping):
        raise BridgeEvidenceError("scene must be an object")
    scene_asset = str(scene.get("asset", ""))
    scene_sha256 = str(scene.get("sha256", ""))
    if not scene_asset or not re.fullmatch(r"[0-9a-f]{64}", scene_sha256):
        raise BridgeEvidenceError("scene asset and lowercase sha256 are required")
    datum = _finite_number(scene.get("elevation_datum_el_m"), "scene elevation datum")

    sources = _source_records(document.get("sources"))
    dimensions = document.get("published_dimensions", [])
    if not isinstance(dimensions, Sequence) or isinstance(dimensions, (str, bytes)):
        raise BridgeEvidenceError("published_dimensions must be an array")
    for index, item in enumerate(dimensions):
        if not isinstance(item, Mapping) or not item.get("component"):
            raise BridgeEvidenceError(f"published dimension {index} needs a component")
        if str(item.get("source_id", "")) not in sources:
            raise BridgeEvidenceError(f"published dimension {index} has unknown source")
        for field, value in item.items():
            if field.endswith("_m") and field != "width_range_m":
                if _finite_number(value, f"published dimension {index} {field}") <= 0.0:
                    raise BridgeEvidenceError(f"{field} must be positive")
        width_range = item.get("width_range_m")
        if width_range is not None:
            if (
                not isinstance(width_range, Sequence)
                or isinstance(width_range, (str, bytes))
                or len(width_range) != 2
            ):
                raise BridgeEvidenceError("width_range_m must contain two values")
            lower = _finite_number(width_range[0], "width range lower")
            upper = _finite_number(width_range[1], "width range upper")
            if lower <= 0.0 or upper < lower:
                raise BridgeEvidenceError("width_range_m must be positive and ordered")
    anchors_raw = document.get("vertical_anchors", [])
    if not isinstance(anchors_raw, Sequence) or isinstance(anchors_raw, (str, bytes)):
        raise BridgeEvidenceError("vertical_anchors must be an array")
    anchors: list[VerticalAnchor] = []
    for index, item in enumerate(anchors_raw):
        if not isinstance(item, Mapping):
            raise BridgeEvidenceError(f"vertical anchor {index} must be an object")
        source_id = str(item.get("source_id", ""))
        if source_id not in sources:
            raise BridgeEvidenceError(f"vertical anchor {index} has unknown source")
        elevation = _finite_number(item.get("elevation_el_m"), "anchor elevation_el_m")
        declared_runtime_y = _finite_number(
            item.get("runtime_y_m"), "anchor runtime_y_m"
        )
        calculated_runtime_y = elevation - datum
        if abs(declared_runtime_y - calculated_runtime_y) > 1e-6:
            raise BridgeEvidenceError(
                f"vertical anchor {index} runtime_y_m does not match the scene datum"
            )
        raw_uncertainty = item.get("uncertainty_m")
        uncertainty = (
            None
            if raw_uncertainty is None
            else _finite_number(raw_uncertainty, "anchor uncertainty_m")
        )
        if uncertainty is not None and uncertainty < 0.0:
            raise BridgeEvidenceError("anchor uncertainty cannot be negative")
        anchors.append(
            VerticalAnchor(
                anchor_id=str(item.get("anchor_id", "")),
                component=str(item.get("component", "")),
                quantity=str(item.get("quantity", "")),
                elevation_el_m=elevation,
                runtime_y_m=calculated_runtime_y,
                source_id=source_id,
                uncertainty_m=uncertainty,
            )
        )
    if not anchors or any(
        not anchor.anchor_id or not anchor.component or not anchor.quantity
        for anchor in anchors
    ):
        raise BridgeEvidenceError("at least one fully identified vertical anchor is required")

    application = _application_state(
        document.get("vertical_profile_application"), sources
    )
    return BridgeEvidence(
        document=document,
        scene_asset=scene_asset,
        scene_sha256=scene_sha256,
        elevation_datum_m=datum,
        sources=sources,
        anchors=tuple(anchors),
        application=application,
    )


def load_bridge_evidence(
    path: Path = DEFAULT_BRIDGE_EVIDENCE_PATH,
) -> BridgeEvidence:
    import json

    return parse_bridge_evidence(json.loads(path.read_text(encoding="utf-8")))


def apply_bridge_vertical_offsets(
    bridge_vertices: np.ndarray,
    vertical_offsets_m: np.ndarray,
    evidence: BridgeEvidence,
) -> np.ndarray:
    """Apply precomputed offsets only after the evidence document is registered.

    Station interpolation deliberately lives outside this function: callers
    must first turn a registered profile into one offset per vertex.  The gate
    cannot silently substitute the clearance anchor for a deck-top profile.
    """

    vertices = np.asarray(bridge_vertices, dtype=np.float32)
    offsets = np.asarray(vertical_offsets_m, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 10:
        raise BridgeEvidenceError("bridge vertices must use the n x 10 scene layout")
    if offsets.shape != (len(vertices),) or not np.isfinite(offsets).all():
        raise BridgeEvidenceError("vertical offsets must be one finite value per vertex")
    if not evidence.application.allowed:
        reasons = ", ".join(evidence.application.reasons)
        raise BridgeEvidenceError(f"bridge vertical application is blocked: {reasons}")
    output = vertices.copy()
    output[:, 1] += offsets.astype(np.float32)
    return output
