from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import astronomy
import numpy as np


@dataclass(frozen=True, slots=True)
class CelestialState:
    sun_altitude_deg: float
    sun_azimuth_deg: float
    sun_direction_eus: np.ndarray
    moon_altitude_deg: float
    moon_azimuth_deg: float
    moon_direction_eus: np.ndarray
    moon_phase_fraction: float
    moon_illuminance_lux: float
    twilight_illuminance_lux: float


def horizontal_direction_eus(azimuth_deg: float, altitude_deg: float) -> np.ndarray:
    """Convert north-clockwise azimuth and altitude to East-Up-South."""

    azimuth = math.radians(azimuth_deg)
    altitude = math.radians(altitude_deg)
    cos_altitude = math.cos(altitude)
    return np.array(
        [
            math.sin(azimuth) * cos_altitude,
            math.sin(altitude),
            -math.cos(azimuth) * cos_altitude,
        ],
        dtype=np.float32,
    )


def twilight_illuminance_lux(sun_altitude_deg: float) -> float:
    """Piecewise log interpolation of horizontal twilight illuminance.

    Anchor values represent orders of magnitude across civil, nautical, and
    astronomical twilight. The model is deliberately bounded at night because
    urban skyglow is handled separately by the renderer.
    """

    altitudes = np.array([0.0, -6.0, -12.0, -18.0], dtype=np.float64)
    lux = np.array([400.0, 3.4, 0.008, 0.0002], dtype=np.float64)
    if sun_altitude_deg >= altitudes[0]:
        return float(lux[0])
    if sun_altitude_deg <= altitudes[-1]:
        return float(lux[-1])
    for index in range(len(altitudes) - 1):
        high_altitude, low_altitude = altitudes[index : index + 2]
        if high_altitude >= sun_altitude_deg >= low_altitude:
            alpha = (sun_altitude_deg - high_altitude) / (
                low_altitude - high_altitude
            )
            return float(
                10.0
                ** (
                    math.log10(lux[index]) * (1.0 - alpha)
                    + math.log10(lux[index + 1]) * alpha
                )
            )
    raise AssertionError("unreachable twilight interval")


class AstronomyModel:
    def __init__(
        self, latitude_deg: float, longitude_deg: float, altitude_m: float
    ) -> None:
        self.observer = astronomy.Observer(
            latitude_deg, longitude_deg, altitude_m
        )

    def _horizontal(
        self, body: astronomy.Body, time: astronomy.Time
    ) -> tuple[float, float]:
        equatorial = astronomy.Equator(
            body, time, self.observer, ofdate=True, aberration=True
        )
        horizontal = astronomy.Horizon(
            time,
            self.observer,
            equatorial.ra,
            equatorial.dec,
            astronomy.Refraction.Normal,
        )
        return horizontal.azimuth, horizontal.altitude

    def sample(self, timestamp: float) -> CelestialState:
        utc = datetime.fromtimestamp(timestamp, timezone.utc)
        time = astronomy.Time(utc.isoformat().replace("+00:00", "Z"))
        sun_azimuth, sun_altitude = self._horizontal(astronomy.Body.Sun, time)
        moon_azimuth, moon_altitude = self._horizontal(astronomy.Body.Moon, time)
        moon_info = astronomy.Illumination(astronomy.Body.Moon, time)
        # Scale apparent lunar magnitude to a clear-sky, near-zenith full-moon
        # upper reference of 0.25 lux. Horizontal illuminance falls with sine
        # of apparent altitude and is zero below the horizon.
        magnitude_ratio = 10.0 ** (-0.4 * (moon_info.mag + 12.74))
        moon_lux = (
            0.25
            * magnitude_ratio
            * max(math.sin(math.radians(moon_altitude)), 0.0)
        )
        return CelestialState(
            sun_altitude_deg=sun_altitude,
            sun_azimuth_deg=sun_azimuth,
            sun_direction_eus=horizontal_direction_eus(
                sun_azimuth, sun_altitude
            ),
            moon_altitude_deg=moon_altitude,
            moon_azimuth_deg=moon_azimuth,
            moon_direction_eus=horizontal_direction_eus(
                moon_azimuth, moon_altitude
            ),
            moon_phase_fraction=moon_info.phase_fraction,
            moon_illuminance_lux=moon_lux,
            twilight_illuminance_lux=twilight_illuminance_lux(sun_altitude),
        )

