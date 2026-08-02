from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .terrain import TerrainSurface


@dataclass(frozen=True, slots=True)
class CameraConfig:
    movement_speed_mps: float = 18.0
    sprint_multiplier: float = 3.0
    walking_speed_mps: float = 4.6
    walking_sprint_multiplier: float = 1.75
    acceleration_response_hz: float = 8.0
    mouse_sensitivity_deg: float = 0.085
    minimum_pitch_deg: float = -85.0
    maximum_pitch_deg: float = 85.0
    standing_camera_height_m: float = 1.68
    camera_collision_radius_m: float = 0.12
    maximum_walkable_slope_deg: float = 38.0
    maximum_step_height_m: float = 0.35


@dataclass(slots=True)
class FreeCamera:
    config: CameraConfig = field(default_factory=CameraConfig)
    position_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 24.0, 235.0], dtype=np.float32)
    )
    yaw_deg: float = 0.0
    pitch_deg: float = 11.55
    velocity_mps: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    walking: bool = False

    @property
    def forward(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        cos_pitch = math.cos(pitch)
        return np.array(
            [
                math.sin(yaw) * cos_pitch,
                math.sin(pitch),
                -math.cos(yaw) * cos_pitch,
            ],
            dtype=np.float32,
        )

    @property
    def horizontal_forward(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        return np.array(
            [math.sin(yaw), 0.0, -math.cos(yaw)], dtype=np.float32
        )

    @property
    def right(self) -> np.ndarray:
        yaw = math.radians(self.yaw_deg)
        return np.array(
            [math.cos(yaw), 0.0, math.sin(yaw)], dtype=np.float32
        )

    def look(self, mouse_delta_x: float, mouse_delta_y: float) -> None:
        sensitivity = self.config.mouse_sensitivity_deg
        self.yaw_deg = (self.yaw_deg + mouse_delta_x * sensitivity) % 360.0
        self.pitch_deg = float(
            np.clip(
                self.pitch_deg - mouse_delta_y * sensitivity,
                self.config.minimum_pitch_deg,
                self.config.maximum_pitch_deg,
            )
        )

    def update(
        self,
        dt_s: float,
        local_input: np.ndarray,
        sprint: bool = False,
        terrain: TerrainSurface | None = None,
    ) -> None:
        """Move from local input, resolving the lens body against terrain.

        Free-flight remains useful for inspecting the show. Walking mode is a
        physical camera operator: vertical input is ignored, water is not a
        walkable floor, slopes and discrete steps are bounded, and the optical
        centre stays at the configured standing height.
        """

        local = np.asarray(local_input, dtype=np.float32)
        if local.shape != (3,):
            raise ValueError("camera input must contain right, up and forward")
        if self.walking:
            local = local.copy()
            local[1] = 0.0
        magnitude = float(np.linalg.norm(local))
        if magnitude > 1.0:
            local = local / magnitude
        direction = (
            self.right * local[0]
            + np.array([0.0, 1.0, 0.0], dtype=np.float32) * local[1]
            + self.horizontal_forward * local[2]
        )
        if self.walking:
            speed = self.config.walking_speed_mps
            if sprint:
                speed *= self.config.walking_sprint_multiplier
        else:
            speed = self.config.movement_speed_mps
            if sprint:
                speed *= self.config.sprint_multiplier
        target_velocity = direction * speed
        response = 1.0 - math.exp(
            -self.config.acceleration_response_hz * max(dt_s, 0.0)
        )
        self.velocity_mps += (target_velocity - self.velocity_mps) * response
        previous = self.position_m.copy()
        proposed = previous + self.velocity_mps * max(dt_s, 0.0)
        if terrain is None:
            self.position_m[:] = proposed
            return
        if self.walking:
            proposed = self._resolve_walking(previous, proposed, terrain)
        else:
            floor = terrain.collision_height_at(proposed[0], proposed[2])
            minimum_y = floor + self.config.camera_collision_radius_m
            if proposed[1] < minimum_y:
                proposed[1] = minimum_y
                self.velocity_mps[1] = max(float(self.velocity_mps[1]), 0.0)
        self.position_m[:] = proposed

    def set_walking(self, enabled: bool, terrain: TerrainSurface) -> bool:
        """Enter operator-height navigation when standing over solid land."""

        if enabled and terrain.is_water(self.position_m[0], self.position_m[2]):
            return False
        self.walking = bool(enabled)
        if self.walking:
            self.position_m[1] = (
                terrain.height_at(self.position_m[0], self.position_m[2])
                + self.config.standing_camera_height_m
            )
            self.velocity_mps[1] = 0.0
        return True

    def _resolve_walking(
        self,
        previous: np.ndarray,
        proposed: np.ndarray,
        terrain: TerrainSurface,
    ) -> np.ndarray:
        if terrain.is_water(proposed[0], proposed[2]):
            self.velocity_mps[[0, 2]] = 0.0
            proposed[[0, 2]] = previous[[0, 2]]
        previous_height = terrain.height_at(previous[0], previous[2])
        proposed_height = terrain.height_at(proposed[0], proposed[2])
        horizontal_distance = float(
            np.linalg.norm(proposed[[0, 2]] - previous[[0, 2]])
        )
        rise = proposed_height - previous_height
        slope_limit = math.tan(
            math.radians(self.config.maximum_walkable_slope_deg)
        ) * horizontal_distance
        if rise > max(self.config.maximum_step_height_m, slope_limit):
            self.velocity_mps[[0, 2]] = 0.0
            proposed[[0, 2]] = previous[[0, 2]]
            proposed_height = previous_height
        proposed[1] = proposed_height + self.config.standing_camera_height_m
        self.velocity_mps[1] = 0.0
        return proposed
