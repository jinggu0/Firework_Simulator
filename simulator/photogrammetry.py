"""Offline camera-pose calibration for georegistered reference photographs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .camera_optics import LensDistortion
from .provenance import ConfidenceGrade, DataRecord


class RegistrationError(ValueError):
    """Raised when a photograph cannot support an auditable registration."""


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    width_px: int
    height_px: int
    focal_x_px: float
    focal_y_px: float
    principal_x_px: float
    principal_y_px: float
    distortion: LensDistortion = LensDistortion()

    @classmethod
    def from_physical_camera(
        cls,
        width_px: int,
        height_px: int,
        focal_length_mm: float,
        sensor_width_mm: float,
        sensor_height_mm: float,
        distortion: LensDistortion | None = None,
    ) -> "CameraIntrinsics":
        if min(
            width_px,
            height_px,
            focal_length_mm,
            sensor_width_mm,
            sensor_height_mm,
        ) <= 0:
            raise RegistrationError("camera dimensions and focal length must be positive")
        return cls(
            width_px=width_px,
            height_px=height_px,
            focal_x_px=focal_length_mm / sensor_width_mm * width_px,
            focal_y_px=focal_length_mm / sensor_height_mm * height_px,
            principal_x_px=width_px * 0.5,
            principal_y_px=height_px * 0.5,
            distortion=distortion or LensDistortion(),
        )

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise RegistrationError("image dimensions must be positive")
        if self.focal_x_px <= 0.0 or self.focal_y_px <= 0.0:
            raise RegistrationError("pixel focal lengths must be positive")
        if not 0.0 <= self.principal_x_px <= self.width_px:
            raise RegistrationError("principal x lies outside the image")
        if not 0.0 <= self.principal_y_px <= self.height_px:
            raise RegistrationError("principal y lies outside the image")
        values = (
            self.focal_x_px,
            self.focal_y_px,
            self.principal_x_px,
            self.principal_y_px,
            self.distortion.k1,
            self.distortion.k2,
            self.distortion.k3,
            self.distortion.p1,
            self.distortion.p2,
        )
        if not all(math.isfinite(value) for value in values):
            raise RegistrationError("camera intrinsics must be finite")


@dataclass(frozen=True, slots=True)
class CameraPose:
    position_eus_m: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        if len(self.position_eus_m) != 3 or not all(
            math.isfinite(float(value)) for value in self.position_eus_m
        ):
            raise RegistrationError("camera position must contain three finite values")
        if not all(
            math.isfinite(value)
            for value in (self.yaw_deg, self.pitch_deg, self.roll_deg)
        ):
            raise RegistrationError("camera angles must be finite")
        if not -89.0 < self.pitch_deg < 89.0:
            raise RegistrationError("camera pitch must remain away from the poles")

    def vector(self) -> np.ndarray:
        return np.array(
            [*self.position_eus_m, self.yaw_deg, self.pitch_deg, self.roll_deg],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> "CameraPose":
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise RegistrationError("camera pose must contain six finite values")
        return cls(tuple(float(value) for value in values[:3]), *values[3:])


@dataclass(frozen=True, slots=True)
class PoseCalibrationResult:
    pose: CameraPose
    converged: bool
    control_points: int
    jacobian_rank: int
    reprojection_rmse_px: float
    reprojection_p95_px: float
    reprojection_max_px: float
    control_bbox_fraction: float
    residuals_px: np.ndarray
    projected_pixels: np.ndarray
    depths_m: np.ndarray
    iterations: int

    @property
    def passed(self) -> bool:
        return (
            self.converged
            and self.control_points >= 6
            and self.jacobian_rank == 6
            and np.all(self.depths_m > 0.05)
            and self.reprojection_rmse_px <= 2.0
            and self.reprojection_p95_px <= 3.0
            and self.reprojection_max_px <= 5.0
            and self.control_bbox_fraction >= 0.02
        )


def camera_basis(pose: CameraPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, forward in the runtime East-Up-South frame."""

    yaw = math.radians(pose.yaw_deg)
    pitch = math.radians(pose.pitch_deg)
    roll = math.radians(pose.roll_deg)
    forward = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch),
        ],
        dtype=np.float64,
    )
    right = np.array([math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float64)
    up = np.cross(right, forward)
    rolled_right = right * math.cos(roll) + up * math.sin(roll)
    rolled_up = up * math.cos(roll) - right * math.sin(roll)
    return rolled_right, rolled_up, forward


def project_world_points(
    world_eus_m: np.ndarray,
    pose: CameraPose,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Project EUS world points into top-left-origin image pixels."""

    world = np.asarray(world_eus_m, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or not np.isfinite(world).all():
        raise RegistrationError("world points must be a finite n x 3 array")
    right, up, forward = camera_basis(pose)
    delta = world - np.asarray(pose.position_eus_m, dtype=np.float64)
    depth = delta @ forward
    safe_depth = np.where(np.abs(depth) < 1e-9, np.copysign(1e-9, depth), depth)
    ideal = np.column_stack((delta @ right, -(delta @ up))) / safe_depth[:, None]
    distorted = intrinsics.distortion.distort(ideal)
    pixels = np.column_stack(
        (
            intrinsics.focal_x_px * distorted[:, 0] + intrinsics.principal_x_px,
            intrinsics.focal_y_px * distorted[:, 1] + intrinsics.principal_y_px,
        )
    )
    return pixels, depth


def fit_camera_pose(
    world_eus_m: np.ndarray,
    observed_pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
    initial_pose: CameraPose,
    *,
    position_bounds_eus_m: tuple[Sequence[float], Sequence[float]] | None = None,
) -> PoseCalibrationResult:
    """Fit a six-DoF pose with a robust pixel-domain least-squares loss."""

    try:
        from scipy.optimize import least_squares
    except ImportError as error:
        raise RuntimeError("install requirements-terrain.txt for pose fitting") from error

    world = np.asarray(world_eus_m, dtype=np.float64)
    observed = np.asarray(observed_pixels, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3 or observed.shape != (len(world), 2):
        raise RegistrationError("control points must be n x 3 world and n x 2 pixels")
    if len(world) < 6:
        raise RegistrationError("at least six control points are required")
    if not np.isfinite(world).all() or not np.isfinite(observed).all():
        raise RegistrationError("control points must be finite")
    if len(np.unique(np.round(world, decimals=6), axis=0)) != len(world):
        raise RegistrationError("duplicate world control points are not allowed")

    if position_bounds_eus_m is None:
        lower_position = np.min(world, axis=0) - np.array([5_000.0, 1_000.0, 5_000.0])
        upper_position = np.max(world, axis=0) + np.array([5_000.0, 1_000.0, 5_000.0])
    else:
        lower_position = np.asarray(position_bounds_eus_m[0], dtype=np.float64)
        upper_position = np.asarray(position_bounds_eus_m[1], dtype=np.float64)
        if lower_position.shape != (3,) or upper_position.shape != (3,):
            raise RegistrationError("position bounds must contain two 3-vectors")
    lower = np.concatenate((lower_position, [-720.0, -85.0, -45.0]))
    upper = np.concatenate((upper_position, [720.0, 85.0, 45.0]))
    initial = initial_pose.vector()
    if np.any(initial < lower) or np.any(initial > upper):
        raise RegistrationError("initial pose lies outside calibration bounds")

    def residual(parameters: np.ndarray) -> np.ndarray:
        projected, depth = project_world_points(
            world, CameraPose.from_vector(parameters), intrinsics
        )
        difference = projected - observed
        behind = depth <= 0.05
        if np.any(behind):
            difference[behind] += np.sign(difference[behind] + 1e-9) * (
                2_000.0 + np.abs(depth[behind, None]) * 100.0
            )
        return difference.ravel()

    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=2_000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    pose = CameraPose.from_vector(solved.x)
    projected, depth = project_world_points(world, pose, intrinsics)
    vector_residual = projected - observed
    radial = np.linalg.norm(vector_residual, axis=1)
    rank = int(np.linalg.matrix_rank(solved.jac, tol=1e-8))
    pixel_extent = np.ptp(observed, axis=0)
    coverage = float(
        pixel_extent[0]
        * pixel_extent[1]
        / (intrinsics.width_px * intrinsics.height_px)
    )
    return PoseCalibrationResult(
        pose=pose,
        converged=bool(solved.success),
        control_points=len(world),
        jacobian_rank=rank,
        reprojection_rmse_px=float(np.sqrt(np.mean(radial * radial))),
        reprojection_p95_px=float(np.percentile(radial, 95.0)),
        reprojection_max_px=float(np.max(radial)),
        control_bbox_fraction=coverage,
        residuals_px=vector_residual,
        projected_pixels=projected,
        depths_m=depth,
        iterations=int(solved.nfev),
    )


def _record(data: Any, label: str) -> DataRecord:
    if not isinstance(data, Mapping):
        raise RegistrationError(f"{label} requires a provenance record")
    try:
        record = DataRecord.from_dict(data)
    except ValueError as error:
        raise RegistrationError(f"{label}: {error}") from error
    if record.grade not in (
        ConfidenceGrade.MEASURED,
        ConfidenceGrade.RECONSTRUCTED,
    ):
        raise RegistrationError(f"{label} must carry confidence grade A or B")
    return record


def calibrate_registration_document(
    document: Mapping[str, Any],
) -> tuple[PoseCalibrationResult, list[str]]:
    """Validate and solve a versioned photograph-registration document."""

    if document.get("schema_version") != 1:
        raise RegistrationError("registration document must use schema_version 1")
    registration_id = str(document.get("registration_id", "")).strip()
    if not registration_id:
        raise RegistrationError("registration_id is required")
    if document.get("target_event_date") != "2024-10-05":
        raise RegistrationError("registration target_event_date must be 2024-10-05")
    image = document.get("image")
    intrinsics_data = document.get("intrinsics")
    if not isinstance(image, Mapping) or not isinstance(intrinsics_data, Mapping):
        raise RegistrationError("image and intrinsics objects are required")
    _record(image.get("evidence"), "image")
    distortion_values = intrinsics_data.get(
        "distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0]
    )
    if len(distortion_values) != 5:
        raise RegistrationError("distortion_coefficients must be k1,k2,p1,p2,k3")
    distortion = LensDistortion(
        k1=float(distortion_values[0]),
        k2=float(distortion_values[1]),
        p1=float(distortion_values[2]),
        p2=float(distortion_values[3]),
        k3=float(distortion_values[4]),
    )
    intrinsics = CameraIntrinsics.from_physical_camera(
        int(image["width_px"]),
        int(image["height_px"]),
        float(intrinsics_data["focal_length_mm"]),
        float(intrinsics_data["sensor_width_mm"]),
        float(intrinsics_data["sensor_height_mm"]),
        distortion,
    )
    principal = intrinsics_data.get("principal_point_px")
    if principal is not None:
        if len(principal) != 2:
            raise RegistrationError("principal_point_px must contain two values")
        intrinsics = CameraIntrinsics(
            intrinsics.width_px,
            intrinsics.height_px,
            intrinsics.focal_x_px,
            intrinsics.focal_y_px,
            float(principal[0]),
            float(principal[1]),
            intrinsics.distortion,
        )
    controls = document.get("control_points", [])
    if not isinstance(controls, list):
        raise RegistrationError("control_points must be a list")
    ids: list[str] = []
    world: list[Sequence[float]] = []
    pixels: list[Sequence[float]] = []
    for control in controls:
        if not isinstance(control, Mapping):
            raise RegistrationError("control point entries must be objects")
        control_id = str(control.get("control_id", "")).strip()
        if not control_id or control_id in ids:
            raise RegistrationError("control_id values must be non-empty and unique")
        _record(control.get("evidence"), f"control point {control_id}")
        world_point = np.asarray(control["world_eus_m"], dtype=np.float64)
        pixel = np.asarray(control["pixel_xy"], dtype=np.float64)
        if world_point.shape != (3,) or pixel.shape != (2,):
            raise RegistrationError(
                f"control point {control_id} must contain world[3] and pixel[2]"
            )
        if not (
            0.0 <= pixel[0] < intrinsics.width_px
            and 0.0 <= pixel[1] < intrinsics.height_px
        ):
            raise RegistrationError(f"control point {control_id} lies outside the image")
        ids.append(control_id)
        world.append(world_point)
        pixels.append(pixel)
    initial_data = document.get("initial_pose", {})
    initial = CameraPose(
        tuple(float(value) for value in initial_data["position_eus_m"]),
        float(initial_data["yaw_deg"]),
        float(initial_data["pitch_deg"]),
        float(initial_data.get("roll_deg", 0.0)),
    )
    bounds = document.get("position_bounds_eus_m")
    result = fit_camera_pose(
        np.asarray(world, dtype=np.float64),
        np.asarray(pixels, dtype=np.float64),
        intrinsics,
        initial,
        position_bounds_eus_m=(bounds[0], bounds[1]) if bounds else None,
    )
    return result, ids
