"""Audit V1-8 event-day Seogang Bridge appearance and station evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from simulator.bridge_event_state import (
    DEFAULT_BRIDGE_EVENT_STATE_PATH,
    load_bridge_event_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERTICAL_EVIDENCE_PATH = (
    REPOSITORY_ROOT / "assets" / "seogang_bridge_vertical_evidence.json"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "bridge_event_state_v1"
    / "bridge_event_state_report.json"
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_report(
    event_path: Path = DEFAULT_BRIDGE_EVENT_STATE_PATH,
    vertical_path: Path = DEFAULT_VERTICAL_EVIDENCE_PATH,
) -> dict[str, object]:
    state = load_bridge_event_state(event_path)
    event_document = state.document
    vertical_document = json.loads(vertical_path.read_text(encoding="utf-8"))
    link = vertical_document["vertical_profile_application"]["event_date_validation"]
    observed_hash = _hash(event_path)
    expected_hash = str(link["event_state_manifest_sha256"])
    photos = event_document["event_photo_review"]
    drawing = event_document["official_drawing_review"]
    assessment = event_document["event_day_assessment"]
    return {
        "schema_version": 1,
        "stage": "V1-8",
        "target_event_date": "2024-10-05",
        "bridge_id": "seogang_bridge",
        "event_state_manifest": event_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "event_state_manifest_sha256": observed_hash,
        "vertical_evidence_link": {
            "manifest": vertical_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "expected_event_state_sha256": expected_hash,
            "checksum_matches": observed_hash == expected_hash,
        },
        "event_photo_review": {
            "reviewed_file_count": len(photos),
            "independent_view_count": state.independent_event_views,
            "qualifying_seogang_view_count": state.qualifying_event_views,
            "minimum_required_independent_views": event_document["thresholds"][
                "minimum_independent_event_views"
            ],
            "all_downloads_checksum_locked": all(
                bool(event_document["sources"])
                and next(
                    item for item in event_document["sources"]
                    if item["source_id"] == photo["source_id"]
                )["checksum"]
                for photo in photos
            ),
            "seogang_bridge_visible_in_any_reviewed_photo": any(
                photo["seogang_bridge_visible"] for photo in photos
            ),
            "conclusion": (
                "The reviewed event photos face the launch corridor and Yeouido "
                "skyline; none resolves Seogang Bridge. They cannot establish "
                "scaffold or fresh-paint state."
            ),
        },
        "event_day_work_state": {
            "paint_contract_active": True,
            "seogang_work_in_official_traffic_control_list": False,
            "topside_traffic_control": assessment["topside_traffic_control"],
            "scaffolding_state": assessment["scaffolding_state"],
            "fresh_paint_boundary_state": assessment[
                "fresh_paint_boundary_state"
            ],
            "render_scaffolding": assessment["render_scaffolding"],
            "render_fresh_paint_boundaries": assessment[
                "render_fresh_paint_boundaries"
            ],
            "scope_warning": assessment["topside_inference_scope"],
        },
        "station_registration": {
            "identified_controls": [
                item["control_id"]
                for item in drawing["identified_control_landmarks"]
            ],
            "control_count": state.identified_station_controls,
            "minimum_required_control_count": event_document["thresholds"][
                "minimum_station_control_points"
            ],
            "longitudinal_raster_plan_m_per_pixel_lower_bound": (
                state.longitudinal_plan_m_per_pixel
            ),
            "maximum_allowed_plan_m_per_pixel": event_document["thresholds"][
                "maximum_drawing_plan_m_per_pixel"
            ],
            "plan_target_source_grade": drawing["plan_target_source_grade"],
            "station_plan_rmse_m": drawing["station_plan_rmse_m"],
            "official_drawing_artifacts_checksum_locked": all(
                item["sha256"] for item in drawing["extracted_artifacts"]
            ),
            "registration_allowed": state.station_registration_allowed,
        },
        "structural_history": event_document["structural_history"],
        "application": {
            "event_day_visual_state_allowed": (
                state.event_day_visual_state_allowed
            ),
            "station_registration_allowed": state.station_registration_allowed,
            "vertical_profile_allowed": state.vertical_profile_allowed,
            "blocking_reasons": list(state.reasons),
            "scene_vertices_modified": 0,
            "runtime_frame_path_changed": False,
            "expected_frame_time_delta_ms": 0.0,
        },
        "passed": bool(
            observed_hash == expected_hash
            and not state.event_day_visual_state_allowed
            and not state.station_registration_allowed
            and not state.vertical_profile_allowed
            and not assessment["render_scaffolding"]
            and not assessment["render_fresh_paint_boundaries"]
        ),
        "next_evidence_gate": (
            "Obtain at least two independent dated views resolving Seogang Bridge "
            "and a third grade-A/B plan control with <=0.125 m/px source scale. "
            "Then measure plan RMSE <=0.25 m and verify structural history through "
            "2024-10-05 before applying a vertical profile."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-state", type=Path, default=DEFAULT_BRIDGE_EVENT_STATE_PATH)
    parser.add_argument(
        "--vertical-evidence", type=Path, default=DEFAULT_VERTICAL_EVIDENCE_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    report = build_report(arguments.event_state.resolve(), arguments.vertical_evidence.resolve())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
