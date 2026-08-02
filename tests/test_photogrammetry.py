from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from simulator.camera_optics import LensDistortion
from simulator.photogrammetry import (
    CameraIntrinsics,
    CameraPose,
    RegistrationError,
    calibrate_registration_document,
    camera_basis,
    fit_camera_pose,
    project_world_points,
)
from tools.calibrate_structure_reference import main as calibrate_main


def _intrinsics(distortion: LensDistortion | None = None) -> CameraIntrinsics:
    return CameraIntrinsics.from_physical_camera(
        1280, 720, 24.0, 36.0, 20.25, distortion
    )


def _world_points() -> np.ndarray:
    return np.array(
        [
            [-8.0, 0.0, -15.0],
            [12.0, 0.0, -15.0],
            [-10.0, 9.0, -18.0],
            [10.0, 12.0, -18.0],
            [-18.0, 3.0, -35.0],
            [18.0, 7.0, -35.0],
            [-7.0, 16.0, -48.0],
            [15.0, 20.0, -52.0],
            [0.0, 5.0, -65.0],
        ],
        dtype=np.float64,
    )


def _evidence() -> dict[str, str]:
    return {
        "confidence_grade": "B",
        "source_id": "synthetic-registration-fixture",
        "source_url": "https://example.test/reference",
        "license": "test fixture",
        "coordinate_reference_system": "local EUS metres / image pixels",
        "units": "m and px",
        "notes": "Synthetic control used only for deterministic tests.",
    }


def _registration_document() -> dict:
    truth = CameraPose((3.0, 2.4, 8.0), 7.0, 4.0, -1.5)
    world = _world_points()
    pixels, _ = project_world_points(world, truth, _intrinsics())
    controls = [
        {
            "control_id": f"point-{index}",
            "world_eus_m": point.tolist(),
            "pixel_xy": pixel.tolist(),
            "evidence": _evidence(),
        }
        for index, (point, pixel) in enumerate(zip(world, pixels))
    ]
    return {
        "schema_version": 1,
        "registration_id": "synthetic-event-view",
        "target_event_date": "2024-10-05",
        "image": {"width_px": 1280, "height_px": 720, "evidence": _evidence()},
        "intrinsics": {
            "focal_length_mm": 24.0,
            "sensor_width_mm": 36.0,
            "sensor_height_mm": 20.25,
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "initial_pose": {
            "position_eus_m": [4.0, 2.0, 9.0],
            "yaw_deg": 6.0,
            "pitch_deg": 3.0,
            "roll_deg": -1.0,
        },
        "control_points": controls,
    }


def test_camera_basis_matches_runtime_forward_convention() -> None:
    right, up, forward = camera_basis(CameraPose((0.0, 0.0, 0.0), 0.0, 0.0))

    assert np.allclose(right, [1.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, 1.0, 0.0])
    assert np.allclose(forward, [0.0, 0.0, -1.0])


def test_projection_uses_top_left_pixels_and_brown_conrady_distortion() -> None:
    pose = CameraPose((0.0, 0.0, 0.0), 0.0, 0.0)
    ideal, depth = project_world_points(
        np.array([[0.0, 0.0, -10.0], [2.0, 1.0, -10.0]]),
        pose,
        _intrinsics(),
    )
    distorted, _ = project_world_points(
        np.array([[0.0, 0.0, -10.0], [2.0, 1.0, -10.0]]),
        pose,
        _intrinsics(LensDistortion(k1=-0.2)),
    )

    assert np.all(depth > 0.0)
    assert np.allclose(ideal[0], [640.0, 360.0])
    assert ideal[1, 0] > 640.0
    assert ideal[1, 1] < 360.0
    assert abs(distorted[1, 0] - 640.0) < abs(ideal[1, 0] - 640.0)


def test_pose_fit_recovers_synthetic_six_dof_camera() -> None:
    truth = CameraPose((3.0, 2.4, 8.0), 7.0, 4.0, -1.5)
    intrinsics = _intrinsics(LensDistortion(k1=-0.04, k2=0.008))
    world = _world_points()
    observed, _ = project_world_points(world, truth, intrinsics)
    initial = CameraPose((4.0, 1.8, 9.5), 5.5, 3.0, -0.5)

    result = fit_camera_pose(world, observed, intrinsics, initial)

    assert result.passed
    assert result.jacobian_rank == 6
    assert np.allclose(result.pose.position_eus_m, truth.position_eus_m, atol=1e-5)
    assert result.pose.yaw_deg == pytest.approx(truth.yaw_deg, abs=1e-5)
    assert result.pose.pitch_deg == pytest.approx(truth.pitch_deg, abs=1e-5)
    assert result.pose.roll_deg == pytest.approx(truth.roll_deg, abs=1e-5)
    assert result.reprojection_rmse_px < 1e-5
    assert result.control_bbox_fraction >= 0.02


def test_pose_fit_rejects_too_few_or_duplicate_controls() -> None:
    intrinsics = _intrinsics()
    pose = CameraPose((0.0, 1.7, 5.0), 0.0, 0.0)
    world = _world_points()
    pixels, _ = project_world_points(world, pose, intrinsics)

    with pytest.raises(RegistrationError, match="at least six"):
        fit_camera_pose(world[:5], pixels[:5], intrinsics, pose)
    duplicate = world.copy()
    duplicate[-1] = duplicate[0]
    with pytest.raises(RegistrationError, match="duplicate"):
        fit_camera_pose(duplicate, pixels, intrinsics, pose)


def test_registration_document_validates_provenance_and_reports_control_ids() -> None:
    world = _world_points()
    document = _registration_document()

    result, ids = calibrate_registration_document(document)

    assert result.passed
    assert ids == [f"point-{index}" for index in range(len(world))]

    document["image"]["evidence"]["confidence_grade"] = "D"
    with pytest.raises(RegistrationError, match="grade A or B"):
        calibrate_registration_document(document)


def test_calibration_cli_writes_a_checksum_locked_passing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "registration.json"
    output = tmp_path / "report.json"
    source.write_text(
        json.dumps(_registration_document()), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys, "argv", ["calibrate_structure_reference", str(source), "--output", str(output)]
    )

    calibrate_main()
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["metrics"]["reprojection_rmse_px"] < 1e-5
    assert report["metrics"]["control_bbox_fraction"] >= 0.02
    assert len(report["source_document_sha256"]) == 64
    assert len(report["control_point_residuals"]) == len(_world_points())


def test_registration_schema_requires_six_controls_and_event_date() -> None:
    schema = json.loads(
        Path("assets/structure_reference_registration.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["target_event_date"]["const"] == "2024-10-05"
    assert schema["properties"]["control_points"]["minItems"] == 6
    assert schema["properties"]["intrinsics"]["properties"][
        "distortion_coefficients"
    ]["description"] == "OpenCV order k1, k2, p1, p2, k3"
