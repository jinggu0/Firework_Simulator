"""Evidence gate for the Seogang Bridge state on the event date."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRIDGE_EVENT_STATE_PATH = (
    REPOSITORY_ROOT / "assets" / "seogang_bridge_event_state_2024-10-05.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BridgeEventStateError(ValueError):
    """Raised when event-day bridge evidence is malformed or overclaimed."""


@dataclass(frozen=True, slots=True)
class BridgeEventState:
    document: Mapping[str, Any]
    qualifying_event_views: int
    independent_event_views: int
    identified_station_controls: int
    longitudinal_plan_m_per_pixel: float
    event_day_visual_state_allowed: bool
    station_registration_allowed: bool
    vertical_profile_allowed: bool
    reasons: tuple[str, ...]


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BridgeEventStateError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise BridgeEventStateError(f"{field} must be finite")
    return result


def parse_bridge_event_state(document: Mapping[str, Any]) -> BridgeEventState:
    if document.get("schema_version") != 1:
        raise BridgeEventStateError("unsupported bridge event-state schema")
    if document.get("target_event_date") != "2024-10-05":
        raise BridgeEventStateError("bridge event state must target 2024-10-05")
    if document.get("bridge_id") != "seogang_bridge":
        raise BridgeEventStateError("bridge event state must identify seogang_bridge")

    thresholds = document.get("thresholds", {})
    minimum_views = int(thresholds.get("minimum_independent_event_views", 0))
    minimum_span_px = int(thresholds.get("minimum_bridge_span_pixels", 0))
    minimum_controls = int(thresholds.get("minimum_station_control_points", 0))
    maximum_mpp = _finite(
        thresholds.get("maximum_drawing_plan_m_per_pixel", math.inf),
        "maximum drawing plan metres per pixel",
    )
    maximum_rmse = _finite(
        thresholds.get("maximum_station_plan_rmse_m", math.inf),
        "maximum station plan rmse",
    )
    accepted_grades = set(thresholds.get("accepted_plan_source_grades", []))
    if min(minimum_views, minimum_span_px, minimum_controls) <= 0:
        raise BridgeEventStateError("all count thresholds must be positive")
    if maximum_mpp <= 0.0 or maximum_rmse <= 0.0:
        raise BridgeEventStateError("all metric thresholds must be positive")
    if not accepted_grades or not accepted_grades <= {"A", "B"}:
        raise BridgeEventStateError("accepted plan grades must be A and/or B")

    sources_raw = document.get("sources", [])
    if not isinstance(sources_raw, Sequence) or isinstance(sources_raw, (str, bytes)):
        raise BridgeEventStateError("sources must be an array")
    sources: dict[str, Mapping[str, Any]] = {}
    for item in sources_raw:
        if not isinstance(item, Mapping):
            raise BridgeEventStateError("each source must be an object")
        source_id = str(item.get("source_id", ""))
        if not source_id or source_id in sources:
            raise BridgeEventStateError("source ids must be present and unique")
        checksum = item.get("checksum")
        if checksum is not None and not _SHA256.fullmatch(str(checksum)):
            raise BridgeEventStateError(f"source {source_id} has an invalid checksum")
        sources[source_id] = item

    photo_records = document.get("event_photo_review", [])
    if not isinstance(photo_records, Sequence) or isinstance(photo_records, (str, bytes)):
        raise BridgeEventStateError("event_photo_review must be an array")
    qualifying_view_ids: set[str] = set()
    all_view_ids: set[str] = set()
    for index, record in enumerate(photo_records):
        if not isinstance(record, Mapping):
            raise BridgeEventStateError(f"photo review {index} must be an object")
        source_id = str(record.get("source_id", ""))
        source = sources.get(source_id)
        if source is None or source.get("role") not in {
            "dated_event_photo", "dated_event_photo_composite",
            "dated_event_photo_duplicate",
        }:
            raise BridgeEventStateError(f"photo review {index} has no dated source")
        if not _SHA256.fullmatch(str(source.get("checksum", ""))):
            raise BridgeEventStateError(f"photo source {source_id} is not checksum locked")
        width = int(record.get("width_px", 0))
        height = int(record.get("height_px", 0))
        if width <= 0 or height <= 0:
            raise BridgeEventStateError("reviewed photo dimensions must be positive")
        view_id = str(record.get("independent_view_id", ""))
        if not view_id:
            raise BridgeEventStateError("independent_view_id is required")
        all_view_ids.add(view_id)
        span = int(record.get("resolved_bridge_span_pixels", 0))
        if span < 0:
            raise BridgeEventStateError("resolved bridge span cannot be negative")
        if bool(record.get("seogang_bridge_visible")) and span >= minimum_span_px:
            qualifying_view_ids.add(view_id)

    drawing = document.get("official_drawing_review", {})
    if not isinstance(drawing, Mapping):
        raise BridgeEventStateError("official_drawing_review must be an object")
    if str(drawing.get("source_id", "")) not in sources:
        raise BridgeEventStateError("drawing review source is unknown")
    artifacts = drawing.get("extracted_artifacts", [])
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise BridgeEventStateError("drawing artifacts must be an array")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or not _SHA256.fullmatch(
            str(artifact.get("sha256", ""))
        ):
            raise BridgeEventStateError("every drawing artifact needs a sha256")
    controls = drawing.get("identified_control_landmarks", [])
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)):
        raise BridgeEventStateError("identified controls must be an array")
    control_ids = [str(item.get("control_id", "")) for item in controls]
    if any(not value for value in control_ids) or len(set(control_ids)) != len(control_ids):
        raise BridgeEventStateError("control ids must be present and unique")
    mpp = _finite(
        drawing.get("longitudinal_raster_estimated_plan_m_per_pixel_lower_bound"),
        "longitudinal raster metres per pixel",
    )
    if mpp <= 0.0:
        raise BridgeEventStateError("longitudinal raster metres per pixel must be positive")
    plan_grade = str(drawing.get("plan_target_source_grade", ""))
    rmse_raw = drawing.get("station_plan_rmse_m")
    rmse = None if rmse_raw is None else _finite(rmse_raw, "station plan rmse")

    assessment = document.get("event_day_assessment", {})
    if not isinstance(assessment, Mapping):
        raise BridgeEventStateError("event_day_assessment must be an object")
    for field in ("scaffolding_state", "fresh_paint_boundary_state"):
        if assessment.get(field) not in {
            "observed_present",
            "observed_absent",
            "unresolved",
        }:
            raise BridgeEventStateError(f"{field} has an unsupported state")
    expected_render_flags = {
        "render_scaffolding": assessment["scaffolding_state"] == "observed_present",
        "render_fresh_paint_boundaries": (
            assessment["fresh_paint_boundary_state"] == "observed_present"
        ),
    }
    if any(
        assessment.get(key) is not expected
        for key, expected in expected_render_flags.items()
    ):
        raise BridgeEventStateError("event-day render flags overclaim the observed state")
    visual_allowed = (
        len(qualifying_view_ids) >= minimum_views
        and assessment.get("scaffolding_state") != "unresolved"
        and assessment.get("fresh_paint_boundary_state") != "unresolved"
    )
    station_allowed = (
        len(control_ids) >= minimum_controls
        and mpp <= maximum_mpp
        and plan_grade in accepted_grades
        and rmse is not None
        and rmse <= maximum_rmse
    )
    history = document.get("structural_history", {})
    if not isinstance(history, Mapping):
        raise BridgeEventStateError("structural_history must be an object")
    try:
        history_through = date.fromisoformat(
            str(history.get("verified_no_major_design_change_through", ""))
        )
    except ValueError as error:
        raise BridgeEventStateError(
            "structural history verified-through date must be ISO-8601"
        ) from error
    history_allowed = bool(history.get("verified_through_event_date"))
    if history_allowed != (history_through >= date(2024, 10, 5)):
        raise BridgeEventStateError(
            "structural history claim does not match its verified-through date"
        )
    vertical_allowed = station_allowed and history_allowed

    reasons: list[str] = []
    if len(qualifying_view_ids) < minimum_views:
        reasons.append("fewer_than_two_qualifying_event_views")
    if assessment.get("scaffolding_state") == "unresolved":
        reasons.append("event_day_scaffolding_state_unresolved")
    if assessment.get("fresh_paint_boundary_state") == "unresolved":
        reasons.append("event_day_paint_boundary_unresolved")
    if len(control_ids) < minimum_controls:
        reasons.append("fewer_than_three_station_controls")
    if mpp > maximum_mpp:
        reasons.append("drawing_plan_resolution_exceeds_threshold")
    if plan_grade not in accepted_grades:
        reasons.append("plan_target_source_grade_not_accepted")
    if rmse is None:
        reasons.append("station_plan_rmse_not_measured")
    elif rmse > maximum_rmse:
        reasons.append("station_plan_rmse_exceeds_threshold")
    if not history_allowed:
        reasons.append("structural_history_not_verified_through_event_date")

    claimed = document.get("application", {})
    expected = {
        "event_day_visual_state_allowed": visual_allowed,
        "station_registration_allowed": station_allowed,
        "vertical_profile_allowed": vertical_allowed,
    }
    if not isinstance(claimed, Mapping) or any(
        claimed.get(key) is not value for key, value in expected.items()
    ):
        raise BridgeEventStateError("application claims do not match evidence gates")
    return BridgeEventState(
        document=document,
        qualifying_event_views=len(qualifying_view_ids),
        independent_event_views=len(all_view_ids),
        identified_station_controls=len(control_ids),
        longitudinal_plan_m_per_pixel=mpp,
        event_day_visual_state_allowed=visual_allowed,
        station_registration_allowed=station_allowed,
        vertical_profile_allowed=vertical_allowed,
        reasons=tuple(reasons),
    )


def load_bridge_event_state(
    path: Path = DEFAULT_BRIDGE_EVENT_STATE_PATH,
) -> BridgeEventState:
    return parse_bridge_event_state(json.loads(path.read_text(encoding="utf-8")))
