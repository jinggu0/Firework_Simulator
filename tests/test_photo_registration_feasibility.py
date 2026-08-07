"""V0-4 determines whether any held photograph can be registered.

The interesting part of the judgement is that it is not about the photographs.
An unpublished pose can be solved; what cannot be conjured is something to
solve against. So the audit has to keep the camera-side and scene-side reasons
separate, and it has to quote the solver's real requirement rather than a
number that once matched it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator.photogrammetry import (
    MINIMUM_CONTROL_POINTS,
    CameraIntrinsics,
    CameraPose,
    RegistrationError,
    fit_camera_pose,
)
from tools.audit_photo_registration_feasibility import (
    REQUIRED_VIEWPOINTS,
    TARGET_CONTROL_UNCERTAINTY_M,
    assess_control_points,
    assess_sources,
    build_report,
)


REPORT = Path(
    "docs/validation/photo_registration_v0/photo_registration_feasibility.json"
)
APPEARANCE = Path("assets/yeouido_2024-10-05_appearance_reference.json")
CONTROLS = Path("assets/yeouido_ngii_public_controls_2017.json")


def test_the_solver_really_rejects_fewer_points_than_the_constant() -> None:
    # Guards the constant against becoming decorative: it must describe what
    # fit_camera_pose actually does, since the audit quotes it as the
    # requirement a photograph has to meet.
    intrinsics = CameraIntrinsics.from_physical_camera(
        width_px=1280,
        height_px=720,
        focal_length_mm=4.65,
        sensor_width_mm=5.95,
        sensor_height_mm=3.35,
    )
    pose = CameraPose(
        position_eus_m=(0.0, 10.0, 0.0), yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0
    )
    world = np.array(
        [[float(i), 5.0, 100.0 + i] for i in range(MINIMUM_CONTROL_POINTS - 1)]
    )
    pixels = np.tile([640.0, 360.0], (len(world), 1))

    with pytest.raises(RegistrationError, match="control points are required"):
        fit_camera_pose(world, pixels, intrinsics, pose)

    # And it accepts the constant's own count, so the requirement is exactly
    # where the audit says it is rather than somewhere above it.
    enough = np.array(
        [[float(i) * 3.0, 5.0, 100.0 + i * 7.0] for i in range(MINIMUM_CONTROL_POINTS)]
    )
    fit_camera_pose(
        enough, np.tile([640.0, 360.0], (len(enough), 1)), intrinsics, pose
    )


def test_the_sources_split_into_camera_ready_and_not() -> None:
    appearance = json.loads(APPEARANCE.read_text(encoding="utf-8"))
    sources = {row["source_id"]: row for row in assess_sources(appearance)}

    assert len(sources) == 5
    # The two CC0 phone captures publish focal length and sensor equivalent and
    # may be held, so nothing on the camera side stops them.
    for source_id in ("commons_mobile_203051", "commons_mobile_203729"):
        assert sources[source_id]["intrinsics_available"]
        assert sources[source_id]["pixels_available_to_the_project"]
        assert sources[source_id]["blockers"] == []
    # The press images publish neither, and the project holds no pixels for them.
    for source_id in ("news1_night_wide", "news1_day_site", "newsis_night_river"):
        assert not sources[source_id]["intrinsics_available"]
        assert not sources[source_id]["pixels_available_to_the_project"]
        assert len(sources[source_id]["blockers"]) == 2


def test_every_published_control_point_is_destroyed() -> None:
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    published = controls["public_controls"]

    assert published
    assert all(point["status"] == "destroyed" for point in published)
    assessment = assess_control_points(controls)
    ngii = next(
        row
        for row in assessment["candidates"]
        if row["family"] == "NGII public control points"
    )
    assert ngii["usable_count"] == 0
    assert not assessment["sufficient"]


def test_the_committed_determination_is_zero_and_blames_control() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["counts"] == {
        "source_count": 5,
        "with_intrinsics": 2,
        "with_usable_pixels": 2,
        "camera_ready": 2,
        "registerable": 0,
    }
    assert report["determination"]["registerable_viewpoint_count"] == 0
    assert not report["determination"]["meets_v0_requirement"]
    # The distinction the audit exists to draw: two sources are camera-ready and
    # still cannot be registered, so the limit is the scene, not the cameras.
    assert report["determination"]["limiting_factor"] == "control points"
    assert not report["application_gates"]["photo_registration_available"]
    assert report["application_gates"]["scene_vertices_modified"] == 0
    assert report["requirement"]["minimum_control_points_per_view"] == (
        MINIMUM_CONTROL_POINTS
    )
    assert report["requirement"]["viewpoints"] == REQUIRED_VIEWPOINTS


def test_the_best_available_control_is_worse_than_the_target() -> None:
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    assessment = assess_control_points(controls)

    assert assessment["points_meeting_target_uncertainty"] == 0
    assert assessment["best_available_uncertainty_m"] == 1.0
    assert assessment["target_uncertainty_m"] == TARGET_CONTROL_UNCERTAINTY_M
    # OSM corners start at the target and reach 3 m, so a registration built on
    # them could not be quoted at 1 m even at their best.
    osm = next(
        row
        for row in assessment["candidates"]
        if row["family"] == "OSM building corners"
    )
    assert osm["uncertainty_m"] == [1.0, 3.0]


def test_sufficient_control_would_make_the_camera_ready_sources_registerable(
    monkeypatch,
) -> None:
    # Proves the determination is a consequence of the evidence rather than a
    # constant zero: grant control and the two CC0 captures become registerable.
    import tools.audit_photo_registration_feasibility as audit

    original = audit.assess_control_points

    def sufficient(controls):
        result = original(controls)
        result["sufficient"] = True
        result["points_meeting_target_uncertainty"] = MINIMUM_CONTROL_POINTS
        return result

    monkeypatch.setattr(audit, "assess_control_points", sufficient)
    report = audit.build_report(APPEARANCE, CONTROLS)

    assert report["determination"]["registerable_viewpoint_count"] == 2
    assert report["determination"]["limiting_factor"] == (
        "camera metadata and pixel availability"
    )
    # Still short of three viewpoints, so V0 would remain unmet for a second
    # and independent reason.
    assert not report["determination"]["meets_v0_requirement"]


def test_the_report_names_what_would_unblock_it() -> None:
    report = build_report(APPEARANCE, CONTROLS)

    assert report["missing_data"]
    assert any("1 m" in item for item in report["missing_data"])
    assert any("Seogang" in item for item in report["missing_data"])
