from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# WGS84 ellipsoid parameters.
_A = 6_378_137.0
_F = 1.0 / 298.257_223_563
_E2 = _F * (2.0 - _F)
_B = _A * (1.0 - _F)
# Second eccentricity squared, used by the Bowring inverse solution.
_EP2 = (_A * _A - _B * _B) / (_B * _B)


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


def _ecef_to_geodetic(point: np.ndarray) -> tuple[float, float, float]:
    """Invert ECEF to WGS84 geodetic degrees and ellipsoidal metres.

    Bowring's closed form supplies the starting latitude, followed by two
    fixed-point refinements. At terrestrial altitudes this converges well below
    the 1 mm round-trip tolerance recorded in the validation spec, which is
    itself three orders of magnitude above float64 noise over a 5 km baseline.
    """

    x, y, z = (float(value) for value in point)
    longitude = math.atan2(y, x)
    horizontal = math.hypot(x, y)
    if horizontal < 1e-9:
        # On the polar axis the longitude is undefined; report the meridian
        # implied by atan2 and solve the height directly.
        latitude = math.copysign(math.pi * 0.5, z)
        altitude = abs(z) - _B
        return math.degrees(latitude), math.degrees(longitude), altitude

    theta = math.atan2(z * _A, horizontal * _B)
    latitude = math.atan2(
        z + _EP2 * _B * math.sin(theta) ** 3,
        horizontal - _E2 * _A * math.cos(theta) ** 3,
    )
    altitude = 0.0
    for _ in range(2):
        sin_latitude = math.sin(latitude)
        prime_vertical = _A / math.sqrt(1.0 - _E2 * sin_latitude * sin_latitude)
        altitude = horizontal / math.cos(latitude) - prime_vertical
        latitude = math.atan2(
            z,
            horizontal
            * (1.0 - _E2 * prime_vertical / (prime_vertical + altitude)),
        )
    sin_latitude = math.sin(latitude)
    prime_vertical = _A / math.sqrt(1.0 - _E2 * sin_latitude * sin_latitude)
    altitude = horizontal / math.cos(latitude) - prime_vertical
    return math.degrees(latitude), math.degrees(longitude), altitude


@dataclass(frozen=True, slots=True)
class LocalTangentPlane:
    """WGS84 to local East-Up-South coordinates measured in metres.

    The runtime frame is right-handed with ``+X`` east, ``+Y`` up, and ``+Z``
    **south** (negated north). ``altitude_m`` is ellipsoidal height; the
    project's rendering datum is a separate named local datum recorded in the
    scenario, not the ellipsoid.

    The origin position and the East/North/Up basis are computed once at
    construction. They were previously rebuilt on every ``to_local`` call,
    which cost roughly ten transcendental evaluations per OSM node.
    """

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    _origin_ecef: np.ndarray = field(
        init=False, repr=False, compare=False, default=None  # type: ignore[assignment]
    )
    _basis_enu: np.ndarray = field(
        init=False, repr=False, compare=False, default=None  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        latitude = math.radians(self.latitude_deg)
        longitude = math.radians(self.longitude_deg)
        east = np.array(
            [-math.sin(longitude), math.cos(longitude), 0.0], dtype=np.float64
        )
        north = np.array(
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ],
            dtype=np.float64,
        )
        up = np.array(
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ],
            dtype=np.float64,
        )
        object.__setattr__(
            self,
            "_origin_ecef",
            _geodetic_to_ecef(
                self.latitude_deg, self.longitude_deg, self.altitude_m
            ),
        )
        # Rows are East, North, Up. The runtime South axis is -North.
        object.__setattr__(
            self, "_basis_enu", np.stack((east, north, up), axis=0)
        )

    def to_local(
        self, latitude_deg: float, longitude_deg: float, altitude_m: float = 0.0
    ) -> np.ndarray:
        delta = (
            _geodetic_to_ecef(latitude_deg, longitude_deg, altitude_m)
            - self._origin_ecef
        )
        enu = self._basis_enu @ delta
        # Runtime convention is East, Up, South.
        return np.array([enu[0], enu[2], -enu[1]], dtype=np.float64)

    def to_local_array(
        self,
        latitude_deg: np.ndarray,
        longitude_deg: np.ndarray,
        altitude_m: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """Vectorised ``to_local`` returning an ``(n, 3)`` East-Up-South array."""

        latitude = np.radians(np.asarray(latitude_deg, dtype=np.float64))
        longitude = np.radians(np.asarray(longitude_deg, dtype=np.float64))
        altitude = np.broadcast_to(
            np.asarray(altitude_m, dtype=np.float64), latitude.shape
        )
        sin_latitude = np.sin(latitude)
        prime_vertical = _A / np.sqrt(1.0 - _E2 * sin_latitude * sin_latitude)
        cos_latitude = np.cos(latitude)
        ecef = np.stack(
            (
                (prime_vertical + altitude) * cos_latitude * np.cos(longitude),
                (prime_vertical + altitude) * cos_latitude * np.sin(longitude),
                (prime_vertical * (1.0 - _E2) + altitude) * sin_latitude,
            ),
            axis=-1,
        )
        enu = (ecef - self._origin_ecef) @ self._basis_enu.T
        return np.stack((enu[..., 0], enu[..., 2], -enu[..., 1]), axis=-1)

    def to_geodetic(self, local_eus_m: np.ndarray) -> tuple[float, float, float]:
        """Invert :meth:`to_local`, returning ``(latitude_deg, longitude_deg,
        altitude_m)``.

        Required by every reprojection and position-error metric in the
        validation spec: without it a simulated burst position cannot be
        compared against a triangulated one in a shared reference frame.
        """

        local = np.asarray(local_eus_m, dtype=np.float64)
        if local.shape != (3,):
            raise ValueError(
                f"expected a 3-component East-Up-South vector, got {local.shape}"
            )
        enu = np.array([local[0], -local[2], local[1]], dtype=np.float64)
        return _ecef_to_geodetic(self._basis_enu.T @ enu + self._origin_ecef)

