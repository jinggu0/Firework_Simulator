from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraConfig:
    movement_speed_mps: float = 18.0
    sprint_multiplier: float = 3.0
    acceleration_response_hz: float = 8.0
    mouse_sensitivity_deg: float = 0.085
    minimum_pitch_deg: float = -85.0
    maximum_pitch_deg: float = 85.0


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
    ) -> None:
        """Move from local right/up/forward input with smooth acceleration."""

        local = np.asarray(local_input, dtype=np.float32)
        magnitude = float(np.linalg.norm(local))
        if magnitude > 1.0:
            local = local / magnitude
        direction = (
            self.right * local[0]
            + np.array([0.0, 1.0, 0.0], dtype=np.float32) * local[1]
            + self.horizontal_forward * local[2]
        )
        speed = self.config.movement_speed_mps
        if sprint:
            speed *= self.config.sprint_multiplier
        target_velocity = direction * speed
        response = 1.0 - math.exp(
            -self.config.acceleration_response_hz * max(dt_s, 0.0)
        )
        self.velocity_mps += (target_velocity - self.velocity_mps) * response
        self.position_m += self.velocity_mps * max(dt_s, 0.0)

