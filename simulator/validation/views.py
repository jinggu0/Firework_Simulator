"""Deterministic, non-historical cameras for visual regression captures."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np

from ..camera import FreeCamera


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUAL_VIEWS_PATH = (
    REPOSITORY_ROOT / "assets" / "visual_regression_views.json"
)

_VIEW_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_EXPECTED_SUBJECTS = {
    "terrain_shoreline",
    "grass_close",
    "road_ground",
    "facade_landmark",
    "bridge_structure",
    "water_reflection",
}
_EXPECTED_SURFACES = {"land", "water", "air"}


class VisualViewError(ValueError):
    """Raised when a regression-view suite is ambiguous or stale."""


def _vector3(value: object, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise VisualViewError(f"{field} must contain three numeric values")
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise VisualViewError(f"{field} must contain three numeric values") from error
    if not all(math.isfinite(component) for component in vector):
        raise VisualViewError(f"{field} contains a non-finite value")
    return vector


@dataclass(frozen=True, slots=True)
class VisualRegressionView:
    """One project-defined camera that reveals a specific appearance defect."""

    view_id: str
    subject: str
    position_eus_m: tuple[float, float, float]
    target_eus_m: tuple[float, float, float]
    expected_surface: str
    minimum_ground_clearance_m: float
    notes: str

    @property
    def yaw_deg(self) -> float:
        delta_x = self.target_eus_m[0] - self.position_eus_m[0]
        delta_z = self.target_eus_m[2] - self.position_eus_m[2]
        return math.degrees(math.atan2(delta_x, -delta_z)) % 360.0

    @property
    def pitch_deg(self) -> float:
        delta_x = self.target_eus_m[0] - self.position_eus_m[0]
        delta_y = self.target_eus_m[1] - self.position_eus_m[1]
        delta_z = self.target_eus_m[2] - self.position_eus_m[2]
        return math.degrees(math.atan2(delta_y, math.hypot(delta_x, delta_z)))

    def apply(self, camera: FreeCamera) -> None:
        camera.position_m[:] = np.asarray(self.position_eus_m, dtype=np.float32)
        camera.yaw_deg = self.yaw_deg
        camera.pitch_deg = self.pitch_deg
        camera.velocity_mps.fill(0.0)
        camera.walking = False


@dataclass(frozen=True, slots=True)
class VisualRegressionSuite:
    """Versioned cameras tied to one exact static-scene asset."""

    suite_id: str
    scenario_id: str
    coordinate_system: str
    scene_asset: str
    scene_asset_sha256: str
    display_mode: str
    source: dict[str, str]
    views: tuple[VisualRegressionView, ...]

    def view(self, view_id: str) -> VisualRegressionView:
        for view in self.views:
            if view.view_id == view_id:
                return view
        raise KeyError(view_id)

    def verify_scene_asset(self, repository_root: Path = REPOSITORY_ROOT) -> Path:
        path = repository_root / self.scene_asset
        if not path.is_file():
            raise VisualViewError(f"visual-regression scene asset is missing: {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != self.scene_asset_sha256:
            raise VisualViewError(
                "visual-regression cameras are stale: scene checksum "
                f"{digest} != {self.scene_asset_sha256}"
            )
        return path


def _parse_view(payload: object) -> VisualRegressionView:
    if not isinstance(payload, dict):
        raise VisualViewError("each visual-regression view must be an object")
    view_id = str(payload.get("view_id", ""))
    subject = str(payload.get("subject", ""))
    expected_surface = str(payload.get("expected_surface", ""))
    if not _VIEW_ID.fullmatch(view_id):
        raise VisualViewError(f"invalid visual-regression view_id {view_id!r}")
    if subject not in _EXPECTED_SUBJECTS:
        raise VisualViewError(f"unknown visual-regression subject {subject!r}")
    if expected_surface not in _EXPECTED_SURFACES:
        raise VisualViewError(f"unknown expected_surface {expected_surface!r}")
    position = _vector3(payload.get("position_eus_m"), "position_eus_m")
    target = _vector3(payload.get("target_eus_m"), "target_eus_m")
    if math.dist(position, target) < 0.1:
        raise VisualViewError(f"view {view_id!r} target is coincident with its camera")
    clearance = float(payload.get("minimum_ground_clearance_m", -1.0))
    if not math.isfinite(clearance) or clearance < 0.0:
        raise VisualViewError(
            f"view {view_id!r} has invalid minimum_ground_clearance_m"
        )
    notes = str(payload.get("notes", "")).strip()
    if not notes:
        raise VisualViewError(f"view {view_id!r} must explain its regression purpose")
    view = VisualRegressionView(
        view_id=view_id,
        subject=subject,
        position_eus_m=position,
        target_eus_m=target,
        expected_surface=expected_surface,
        minimum_ground_clearance_m=clearance,
        notes=notes,
    )
    if not -85.0 <= view.pitch_deg <= 85.0:
        raise VisualViewError(f"view {view_id!r} exceeds the runtime pitch limits")
    return view


def load_visual_regression_suite(
    path: Path = DEFAULT_VISUAL_VIEWS_PATH,
) -> VisualRegressionSuite:
    """Load and strictly validate the canonical six-view appearance suite."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VisualViewError(f"cannot load visual-regression views: {error}") from error
    if payload.get("schema_version") != 1:
        raise VisualViewError("visual-regression schema_version must be 1")
    views = tuple(_parse_view(item) for item in payload.get("views", []))
    ids = [view.view_id for view in views]
    subjects = [view.subject for view in views]
    if len(ids) != len(set(ids)):
        raise VisualViewError("visual-regression view_id values must be unique")
    if set(subjects) != _EXPECTED_SUBJECTS or len(subjects) != len(
        _EXPECTED_SUBJECTS
    ):
        raise VisualViewError(
            "visual-regression suite must contain each canonical subject once"
        )
    source = payload.get("source")
    if not isinstance(source, dict):
        raise VisualViewError("visual-regression suite requires source metadata")
    if source.get("confidence_grade") not in {"C", "D"}:
        raise VisualViewError(
            "project-defined regression cameras must remain grade C or D"
        )
    if not str(source.get("source_id", "")).strip():
        raise VisualViewError("visual-regression source_id is required")
    digest = str(payload.get("scene_asset_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise VisualViewError("scene_asset_sha256 must be a lowercase SHA-256")
    if payload.get("display_mode") not in {"human_vision", "physical_camera"}:
        raise VisualViewError("unsupported visual-regression display_mode")
    return VisualRegressionSuite(
        suite_id=str(payload.get("suite_id", "")),
        scenario_id=str(payload.get("scenario_id", "")),
        coordinate_system=str(payload.get("coordinate_system", "")),
        scene_asset=str(payload.get("scene_asset", "")),
        scene_asset_sha256=digest,
        display_mode=str(payload["display_mode"]),
        source={str(key): str(value) for key, value in source.items()},
        views=views,
    )
