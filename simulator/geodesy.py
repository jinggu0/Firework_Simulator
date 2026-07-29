from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# WGS84 ellipsoid parameters.
_A = 6_378_137.0
_F = 1.0 / 298.257_223_563
_E2 = _F * (2.0 - _F)


def _geodetic_to_ecef(
    latitude_deg: float, longitude_deg: float, altitude_m: float
) -> np.ndarray:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_latitude = math.sin(latitude)
    prime_vertical = _A / math.sqrt(1.0 - _E2 * sin_latitude**2)
    return np.array(
        [
            (prime_vertical + altitude_m) * math.cos(latitude) * math.cos(longitude),
            (prime_vertical + altitude_m) * math.cos(latitude) * math.sin(longitude),
            (prime_vertical * (1.0 - _E2) + altitude_m) * sin_latitude,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class LocalTangentPlane:
    """WGS84 to local East-Up-South coordinates measured in metres."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0

    def to_local(
        self, latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0
    ) -> np.ndarray:
        origin = _geodetic_to_ecef(
            self.latitude_deg, self.longitude_deg, self.altitude_m
        )
        point = _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
        delta = point - origin
        latitude = math.radians(self.latitude_deg)
        longitude = math.radians(self.longitude_deg)
        east = np.array([-math.sin(longitude), math.cos(longitude), 0.0])
        north = np.array(
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ]
        )
        up = np.array(
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ]
        )
        # Runtime convention is East, Up, South.
        return np.array(
            [np.dot(east, delta), np.dot(up, delta), -np.dot(north, delta)],
            dtype=np.float64,
        )

