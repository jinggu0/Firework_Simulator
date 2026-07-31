from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np

from .atmosphere import moist_air_density, wind_scale_at_height
from .config import AtmosphereConfig


def meteorological_wind_to_eus(
    speed_mps: float, from_direction_deg: float
) -> np.ndarray:
    """Convert wind-from bearing into an East-Up-South velocity vector."""

    bearing = math.radians(from_direction_deg)
    return np.array(
        [-speed_mps * math.sin(bearing), 0.0, speed_mps * math.cos(bearing)],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class EnvironmentTimeline:
    timestamps: np.ndarray
    temperature_c: np.ndarray
    relative_humidity_fraction: np.ndarray
    pressure_hpa: np.ndarray
    cloud_cover_fraction: np.ndarray
    wind_velocity_mps: np.ndarray
    show_start_timestamp: float
    show_end_timestamp: float
    source: str

    @classmethod
    def load(cls, path: Path) -> "EnvironmentTimeline":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data["records"]
        timestamps = np.array(
            [datetime.fromisoformat(record["time"]).timestamp() for record in records],
            dtype=np.float64,
        )
        wind = np.array(
            [
                meteorological_wind_to_eus(
                    record["wind_speed_mps"], record["wind_from_direction_deg"]
                )
                for record in records
            ]
        )
        return cls(
            timestamps=timestamps,
            temperature_c=np.array([r["temperature_c"] for r in records]),
            relative_humidity_fraction=np.array(
                [r["relative_humidity_percent"] / 100.0 for r in records]
            ),
            pressure_hpa=np.array([r["pressure_hpa"] for r in records]),
            cloud_cover_fraction=np.array(
                [r["cloud_cover_oktas"] / 8.0 for r in records]
            ),
            wind_velocity_mps=wind,
            show_start_timestamp=datetime.fromisoformat(
                data["event"]["show_start"]
            ).timestamp(),
            show_end_timestamp=datetime.fromisoformat(
                data["event"]["show_end"]
            ).timestamp(),
            source=data["provenance"]["weather_source"],
        )

    def sample(self, timestamp: float) -> AtmosphereConfig:
        right = int(np.searchsorted(self.timestamps, timestamp, side="right"))
        if right <= 0:
            left = right = 0
            alpha = 0.0
        elif right >= len(self.timestamps):
            left = right = len(self.timestamps) - 1
            alpha = 0.0
        else:
            left = right - 1
            span = self.timestamps[right] - self.timestamps[left]
            alpha = (timestamp - self.timestamps[left]) / span

        def interpolate(values: np.ndarray):
            return values[left] * (1.0 - alpha) + values[right] * alpha

        temperature_k = float(interpolate(self.temperature_c) + 273.15)
        humidity = float(interpolate(self.relative_humidity_fraction))
        pressure_pa = float(interpolate(self.pressure_hpa) * 100.0)
        wind = interpolate(self.wind_velocity_mps)
        # Neutral logarithmic surface-layer profile at the roughness length of
        # the river corridor, replacing a bare 1.4x factor. The two agree to
        # 0.3%, so this names the physics without moving the trajectory.
        upper_wind = wind * wind_scale_at_height(100.0)
        density = moist_air_density(temperature_k, pressure_pa, humidity)
        return AtmosphereConfig(
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            relative_humidity=humidity,
            wind_velocity_mps=tuple(float(value) for value in wind),
            wind_velocity_100m_mps=tuple(float(value) for value in upper_wind),
            air_density_kg_m3=density,
            cloud_cover_fraction=float(interpolate(self.cloud_cover_fraction)),
        )
