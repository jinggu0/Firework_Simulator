"""Fit and audit a camera pose for a georegistered structure photograph."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from simulator.photogrammetry import (
    RegistrationError,
    calibrate_registration_document,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        raw = arguments.input.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        result, control_ids = calibrate_registration_document(document)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, RegistrationError) as error:
        parser.error(str(error))

    residuals = []
    for index, control_id in enumerate(control_ids):
        residuals.append(
            {
                "control_id": control_id,
                "projected_pixel_xy": result.projected_pixels[index].tolist(),
                "residual_xy_px": result.residuals_px[index].tolist(),
                "residual_radius_px": float(
                    (result.residuals_px[index] ** 2).sum() ** 0.5
                ),
                "camera_depth_m": float(result.depths_m[index]),
            }
        )
    report = {
        "schema_version": 1,
        "registration_id": document["registration_id"],
        "source_document": str(arguments.input),
        "source_document_sha256": sha256(raw).hexdigest(),
        "target_event_date": document.get("target_event_date"),
        "passed": result.passed,
        "thresholds": {
            "minimum_control_points": 6,
            "required_jacobian_rank": 6,
            "maximum_rmse_px": 2.0,
            "maximum_p95_px": 3.0,
            "maximum_single_residual_px": 5.0,
            "minimum_control_bbox_fraction": 0.02,
            "all_points_in_front_of_camera": True,
        },
        "pose": {
            "position_eus_m": list(result.pose.position_eus_m),
            "yaw_deg": result.pose.yaw_deg,
            "pitch_deg": result.pose.pitch_deg,
            "roll_deg": result.pose.roll_deg,
        },
        "metrics": {
            "converged": result.converged,
            "iterations": result.iterations,
            "control_points": result.control_points,
            "jacobian_rank": result.jacobian_rank,
            "reprojection_rmse_px": result.reprojection_rmse_px,
            "reprojection_p95_px": result.reprojection_p95_px,
            "reprojection_max_px": result.reprojection_max_px,
            "control_bbox_fraction": result.control_bbox_fraction,
            "minimum_camera_depth_m": float(result.depths_m.min()),
        },
        "control_point_residuals": residuals,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{'PASS' if result.passed else 'FAIL'} {document['registration_id']}: "
        f"RMSE {result.reprojection_rmse_px:.3f} px, "
        f"p95 {result.reprojection_p95_px:.3f} px, "
        f"max {result.reprojection_max_px:.3f} px; wrote {arguments.output}"
    )
    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
