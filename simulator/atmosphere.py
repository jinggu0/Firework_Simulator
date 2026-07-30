"""Clear-sky atmospheric optical depth and extinction.

The renderer previously attenuated starlight with a bare
``exp(-0.12 * (air_mass - 1))``, a constant with no recorded source, duplicated
between ``simulator/environmental_optics.py`` and the sky shader. This module
replaces it with a wavelength-resolved optical depth built from named,
published relations, so the same quantity can be checked against tabulated
values instead of taken on trust.

Three terms are modelled:

Rayleigh
    Molecular scattering, from the Bodhaine et al. (1999) fit. This is the
    dominant term in the visible and is the one that can be validated against
    published tables to better than a percent.

Aerosol (Mie)
    Ångström turbidity, ``tau = beta * (lambda / 1 um) ** -alpha``. The form is
    standard; the ``alpha`` and ``beta`` values for Seoul on 2024-10-05 are not
    held, so the defaults are documented urban estimates and carry grade C.

Ozone
    Chappuis-band absorption. **The absorption cross-sections and the column
    ozone amount for the event are not held**, so this term defaults to zero
    and is graded U. It is present as a named, wired-in slot rather than
    silently folded into the aerosol term, because folding it in would make an
    absent measurement look like a modelled one.

References
----------
Bodhaine, Wood, Dutton and Slusser, "On Rayleigh Optical Depth Calculations",
Journal of Atmospheric and Oceanic Technology 16(11), 1999.

Kasten and Young, "Revised optical air mass tables and approximation formula",
Applied Optics 28(22), 1989.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .provenance import ConfidenceGrade

STANDARD_PRESSURE_PA = 101_325.0
"""Sea-level standard pressure, the reference for the Rayleigh fit."""

MAGNITUDES_PER_OPTICAL_DEPTH = 2.5 / math.log(10.0)
"""1.0857..., converting an optical depth to astronomical magnitudes."""

# Wavelengths the renderer's three colour channels are evaluated at. These match
# PhysicalCameraConfig.wavelength_rgb_nm so the extinction a channel receives is
# the extinction at the wavelength its quantum efficiency was defined for.
RGB_WAVELENGTHS_NM = (610.0, 550.0, 460.0)


def rayleigh_optical_depth(
    wavelength_nm: float | np.ndarray,
    pressure_pa: float = STANDARD_PRESSURE_PA,
) -> np.ndarray:
    """Vertical Rayleigh optical depth of the whole atmosphere.

    Uses the Bodhaine et al. (1999) fit, which is stated for standard sea-level
    pressure and scales linearly with station pressure because the optical
    depth is proportional to the molecular column above the observer.

    The numerator and denominator of the fit are individually large and of
    opposite sign near the visible band; they very nearly cancel. This is a
    property of the published rational fit, not a numerical defect.
    """

    micrometres = np.asarray(wavelength_nm, dtype=np.float64) * 1e-3
    inverse_square = 1.0 / (micrometres * micrometres)
    square = micrometres * micrometres
    numerator = (
        1.045_599_6
        - 341.290_61 * inverse_square
        - 0.902_308_50 * square
    )
    denominator = (
        1.0 + 0.002_705_988_9 * inverse_square - 85.968_563 * square
    )
    depth = 0.002_152_0 * numerator / denominator
    return depth * (pressure_pa / STANDARD_PRESSURE_PA)


def aerosol_optical_depth(
    wavelength_nm: float | np.ndarray,
    turbidity_beta: float = 0.12,
    angstrom_alpha: float = 1.3,
) -> np.ndarray:
    """Ångström aerosol optical depth.

    ``turbidity_beta`` is the optical depth at 1 um and ``angstrom_alpha`` the
    wavelength exponent. The functional form is standard; the default values
    are documented estimates for a hazy urban site and are **not** a
    measurement of Seoul on 2024-10-05.
    """

    micrometres = np.asarray(wavelength_nm, dtype=np.float64) * 1e-3
    return max(turbidity_beta, 0.0) * micrometres ** (-angstrom_alpha)


def ozone_optical_depth(
    wavelength_nm: float | np.ndarray,
    column_dobson: float = 0.0,
    cross_section_per_dobson: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Chappuis-band ozone absorption optical depth.

    Returns zero unless both a column amount and a per-wavelength cross-section
    are supplied. Neither is held for the event, so the default is an explicit
    zero rather than a plausible-looking estimate.
    """

    coefficient = np.asarray(cross_section_per_dobson, dtype=np.float64)
    shape = np.shape(np.asarray(wavelength_nm, dtype=np.float64))
    return np.broadcast_to(
        coefficient * max(column_dobson, 0.0), shape
    ).astype(np.float64)


def relative_air_mass(altitude_deg: float) -> float:
    """Kasten-Young (1989) relative optical air mass.

    Remains finite at and slightly below the horizon, where a plane-parallel
    ``1/cos(z)`` diverges.

    The fit returns 0.99971 rather than exactly 1 at the zenith. That 0.03
    percent offset is a property of the published relation, not a defect, and
    is deliberately not special-cased: clamping it would make the function
    disagree with the tables it is taken from.
    """

    zenith_deg = 90.0 - altitude_deg
    cosine = math.cos(math.radians(min(zenith_deg, 90.0)))
    correction = 0.50572 * max(96.07995 - zenith_deg, 1e-3) ** -1.6364
    return 1.0 / max(cosine + correction, 1e-3)


@dataclass(frozen=True, slots=True)
class AtmosphericOptics:
    """Clear-sky extinction for one atmospheric state.

    ``pressure_pa`` sets the molecular column and therefore the Rayleigh term.
    The aerosol and ozone parameters carry their own confidence grades because
    they are not measurements of the event.
    """

    pressure_pa: float = STANDARD_PRESSURE_PA
    turbidity_beta: float = 0.12
    angstrom_alpha: float = 1.3
    ozone_column_dobson: float = 0.0
    ozone_cross_section_per_dobson: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def aerosol_grade(self) -> ConfidenceGrade:
        return ConfidenceGrade.MODELLED

    @property
    def ozone_grade(self) -> ConfidenceGrade:
        """Ozone stays unverified until a column measurement is supplied."""

        if self.ozone_column_dobson > 0.0 and any(
            value > 0.0 for value in self.ozone_cross_section_per_dobson
        ):
            return ConfidenceGrade.MODELLED
        return ConfidenceGrade.UNVERIFIED

    def vertical_optical_depth(
        self, wavelength_nm: float | np.ndarray
    ) -> np.ndarray:
        return (
            rayleigh_optical_depth(wavelength_nm, self.pressure_pa)
            + aerosol_optical_depth(
                wavelength_nm, self.turbidity_beta, self.angstrom_alpha
            )
            + ozone_optical_depth(
                wavelength_nm,
                self.ozone_column_dobson,
                np.asarray(self.ozone_cross_section_per_dobson)
                if np.ndim(wavelength_nm)
                else float(np.mean(self.ozone_cross_section_per_dobson)),
            )
        )

    def rgb_vertical_optical_depth(self) -> np.ndarray:
        return self.vertical_optical_depth(np.asarray(RGB_WAVELENGTHS_NM))

    def transmittance(
        self, wavelength_nm: float | np.ndarray, altitude_deg: float
    ) -> np.ndarray:
        """Direct-beam transmittance along a slant path to ``altitude_deg``."""

        air_mass = relative_air_mass(altitude_deg)
        return np.exp(-self.vertical_optical_depth(wavelength_nm) * air_mass)

    def rgb_transmittance(self, altitude_deg: float) -> np.ndarray:
        return self.transmittance(np.asarray(RGB_WAVELENGTHS_NM), altitude_deg)

    def extinction_magnitudes(self, altitude_deg: float) -> float:
        """Broadband visual extinction in magnitudes at ``altitude_deg``.

        Evaluated at 550 nm, close to the peak of the photopic response, which
        is the band a visual magnitude is defined in.
        """

        depth = float(self.vertical_optical_depth(550.0))
        return (
            MAGNITUDES_PER_OPTICAL_DEPTH
            * depth
            * relative_air_mass(altitude_deg)
        )

    def zenith_extinction_magnitudes(self) -> float:
        """Extinction at the zenith, the figure observing sites publish."""

        return self.extinction_magnitudes(90.0)

    def summary(self) -> dict[str, object]:
        rayleigh = float(rayleigh_optical_depth(550.0, self.pressure_pa))
        aerosol = float(
            aerosol_optical_depth(
                550.0, self.turbidity_beta, self.angstrom_alpha
            )
        )
        return {
            "pressure_pa": self.pressure_pa,
            "rayleigh_optical_depth_550nm": rayleigh,
            "aerosol_optical_depth_550nm": aerosol,
            "ozone_optical_depth_550nm": float(
                ozone_optical_depth(
                    550.0,
                    self.ozone_column_dobson,
                    float(np.mean(self.ozone_cross_section_per_dobson)),
                )
            ),
            "total_optical_depth_550nm": rayleigh
            + aerosol
            + float(
                ozone_optical_depth(
                    550.0,
                    self.ozone_column_dobson,
                    float(np.mean(self.ozone_cross_section_per_dobson)),
                )
            ),
            "zenith_extinction_mag": self.zenith_extinction_magnitudes(),
            "aerosol_grade": self.aerosol_grade.value,
            "ozone_grade": self.ozone_grade.value,
        }


def from_atmosphere_config(config, **overrides) -> AtmosphericOptics:
    """Build optics from the runtime atmospheric state.

    Only pressure is taken from the observed weather; aerosol and ozone are
    not observed and keep their documented defaults unless overridden.
    """

    return AtmosphericOptics(pressure_pa=config.pressure_pa, **overrides)
