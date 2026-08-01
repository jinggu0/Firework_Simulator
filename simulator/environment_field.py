"""Atmospheric state as a field over position and time.

``EnvironmentTimeline.sample(t)`` returns one global :class:`AtmosphereConfig`
with no spatial dependence at all, and there has never been a visibility term.
The product requirements ask for ``wind``, ``temperature``, ``humidity``,
``pressure``, and ``visibility`` as functions of ``(x, y, z, t)``. This module
provides that interface and one implementation.

What the implementation can honestly provide is **vertical** structure. A single
station observation carries no horizontal information, so any horizontal
variation would be invented; the field is horizontally uniform and says so. The
vertical profiles are standard relations driven by the observed surface values:

* pressure from the barometric equation with a lapse rate,
* temperature from that lapse rate,
* humidity from a conserved mixing ratio,
* wind from the logarithmic surface-layer profile,
* visibility from aerosol extinction via the Koschmieder relation.

Each has a real consumer. Density falls about 3% by the time a shell reaches
burst altitude, which changes its drag; sound speed follows temperature and
humidity; visibility drives the aerosol term of
:mod:`simulator.atmosphere`, which sets stellar extinction.

A ``PrecomputedWindField`` sampling an offline CFD solution is deliberately
**not** here. Without a solved field it would be an empty class, and the
architecture's own reasoning for choosing offline over runtime CFD was that an
uncalibrated city-scale solve is a grade-C model dressed as a measurement.

References
----------
ISO 2533:1975 International Standard Atmosphere — barometric relation and the
6.5 K/km reference lapse rate.

Monin-Obukhov surface-layer theory, neutral limit — the logarithmic wind
profile ``u(z) = u* / k * ln(z / z0)``.

Koschmieder H., "Theorie der horizontalen Sichtweite", Beitrage zur Physik der
freien Atmosphare 12, 1924 — visibility from the extinction coefficient at a
2% contrast threshold.

Hanel G., "The properties of atmospheric aerosol particles as functions of the
relative humidity", Advances in Geophysics 19, 1976 — hygroscopic growth.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np

from .atmosphere import (
    DRY_TURBIDITY_BETA,
    REFERENCE_WIND_HEIGHT_M,
    STANDARD_LAPSE_RATE_K_PER_M,
    SURFACE_ROUGHNESS_LENGTH_M,
    AtmosphericOptics,
    aerosol_optical_depth,
    pressure_at_height_pa,
    relative_humidity_at_height,
    temperature_at_height_k,
    wind_scale_at_height,
)
from .config import AtmosphereConfig
from .environment import EnvironmentTimeline, moist_air_density

@runtime_checkable
class EnvironmentField(Protocol):
    """Atmospheric state as a function of position and time.

    Position is local East-Up-South metres; time is a POSIX timestamp.
    """

    def wind_eus_mps(
        self, position_m: np.ndarray, timestamp: float
    ) -> np.ndarray: ...

    def temperature_k(self, position_m: np.ndarray, timestamp: float) -> float: ...

    def pressure_pa(self, position_m: np.ndarray, timestamp: float) -> float: ...

    def relative_humidity(
        self, position_m: np.ndarray, timestamp: float
    ) -> float: ...

    def visibility_m(self, position_m: np.ndarray, timestamp: float) -> float: ...

    def air_density_kg_m3(
        self, position_m: np.ndarray, timestamp: float
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class StationTimelineField:
    """Vertical profiles above a single station observation.

    Horizontally uniform by construction: one station tells you nothing about
    how the field varies across the scene, and pretending otherwise would be
    inventing structure. Horizontal variation needs either a mesoscale analysis
    or an offline solve, neither of which this project holds.
    """

    timeline: EnvironmentTimeline
    lapse_rate_k_per_m: float = STANDARD_LAPSE_RATE_K_PER_M
    roughness_length_m: float = SURFACE_ROUGHNESS_LENGTH_M
    dry_turbidity_beta: float = DRY_TURBIDITY_BETA
    angstrom_alpha: float = 1.3

    @property
    def is_horizontally_uniform(self) -> bool:
        return True

    def _surface(self, timestamp: float) -> AtmosphereConfig:
        return self.timeline.sample(timestamp)

    @staticmethod
    def _height(position_m: np.ndarray) -> float:
        return max(float(np.asarray(position_m, dtype=np.float64)[1]), 0.0)

    # -- the field ---------------------------------------------------------

    def wind_eus_mps(
        self, position_m: np.ndarray, timestamp: float
    ) -> np.ndarray:
        surface = self._surface(timestamp)
        scale = wind_scale_at_height(
            self._height(position_m),
            REFERENCE_WIND_HEIGHT_M,
            self.roughness_length_m,
        )
        return np.asarray(surface.wind_velocity_mps, dtype=np.float64) * scale

    def temperature_k(self, position_m: np.ndarray, timestamp: float) -> float:
        surface = self._surface(timestamp)
        return temperature_at_height_k(
            surface.temperature_k, self._height(position_m),
            self.lapse_rate_k_per_m,
        )

    def pressure_pa(self, position_m: np.ndarray, timestamp: float) -> float:
        surface = self._surface(timestamp)
        return pressure_at_height_pa(
            surface.pressure_pa,
            surface.temperature_k,
            self._height(position_m),
            self.lapse_rate_k_per_m,
        )

    def relative_humidity(
        self, position_m: np.ndarray, timestamp: float
    ) -> float:
        surface = self._surface(timestamp)
        return relative_humidity_at_height(
            surface.relative_humidity,
            surface.temperature_k,
            surface.pressure_pa,
            self._height(position_m),
            self.lapse_rate_k_per_m,
        )

    def air_density_kg_m3(
        self, position_m: np.ndarray, timestamp: float
    ) -> float:
        return moist_air_density(
            self.temperature_k(position_m, timestamp),
            self.pressure_pa(position_m, timestamp),
            self.relative_humidity(position_m, timestamp),
        )

    def optics(
        self, position_m: np.ndarray, timestamp: float
    ) -> AtmosphericOptics:
        """Clear-sky optics at a point, driven by the state at that point.

        Building the optics here rather than reimplementing the optical depths
        keeps one definition of extinction: the renderer's haze, the stellar
        extinction, and the visibility reported below all read this object.
        """

        return AtmosphericOptics(
            pressure_pa=self.pressure_pa(position_m, timestamp),
            turbidity_beta=self.dry_turbidity_beta,
            angstrom_alpha=self.angstrom_alpha,
            relative_humidity=self.relative_humidity(position_m, timestamp),
        )

    def aerosol_optical_depth_550nm(
        self, position_m: np.ndarray, timestamp: float
    ) -> float:
        """Ambient aerosol depth, grown from its dry value by humidity."""

        humidity = self.relative_humidity(position_m, timestamp)
        return float(
            aerosol_optical_depth(
                550.0, self.dry_turbidity_beta, self.angstrom_alpha, humidity
            )
        )

    def visibility_m(self, position_m: np.ndarray, timestamp: float) -> float:
        """Meteorological visibility implied by the modelled extinction.

        **Not a measurement.** The event weather record carries no visibility
        observation, so this is what the aerosol model implies rather than what
        was seen. Supplying an observed visibility would invert this relation
        and calibrate the turbidity instead, which is the more useful direction.
        """

        return self.optics(position_m, timestamp).visibility_m()

    def sample(
        self, position_m: np.ndarray, timestamp: float
    ) -> AtmosphereConfig:
        """The full state at a point, in the form the solvers already consume."""

        surface = self._surface(timestamp)
        wind = self.wind_eus_mps(position_m, timestamp)
        return AtmosphereConfig(
            temperature_k=self.temperature_k(position_m, timestamp),
            pressure_pa=self.pressure_pa(position_m, timestamp),
            relative_humidity=self.relative_humidity(position_m, timestamp),
            wind_velocity_mps=tuple(float(value) for value in wind),
            wind_velocity_100m_mps=surface.wind_velocity_100m_mps,
            air_density_kg_m3=self.air_density_kg_m3(position_m, timestamp),
            cloud_cover_fraction=surface.cloud_cover_fraction,
        )

    def summary(self, timestamp: float) -> dict[str, float | bool]:
        ground = np.zeros(3)
        aloft = np.array([0.0, 300.0, 0.0])
        return {
            "horizontally_uniform": self.is_horizontally_uniform,
            "surface_temperature_k": self.temperature_k(ground, timestamp),
            "temperature_300m_k": self.temperature_k(aloft, timestamp),
            "surface_pressure_pa": self.pressure_pa(ground, timestamp),
            "pressure_300m_pa": self.pressure_pa(aloft, timestamp),
            "surface_density_kg_m3": self.air_density_kg_m3(ground, timestamp),
            "density_300m_kg_m3": self.air_density_kg_m3(aloft, timestamp),
            "surface_humidity": self.relative_humidity(ground, timestamp),
            "humidity_300m": self.relative_humidity(aloft, timestamp),
            "wind_scale_300m": wind_scale_at_height(
                300.0, REFERENCE_WIND_HEIGHT_M, self.roughness_length_m
            ),
            "visibility_km": self.visibility_m(ground, timestamp) / 1_000.0,
        }
