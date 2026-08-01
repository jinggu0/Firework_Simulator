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

from dataclasses import dataclass, replace
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


HYGROSCOPIC_GROWTH_EXPONENT = 0.6
"""Exponent in ``f(RH) = (1 - RH)^-gamma`` for continental aerosol.

Follows Hanel, "The properties of atmospheric aerosol particles as functions of
the relative humidity", Advances in Geophysics 19, 1976. Particles take up water
as humidity rises, growing and scattering more.
"""

REFERENCE_HUMIDITY = 0.56
"""Relative humidity the shipped dry turbidity is referenced to.

It is the value observed at the show. Anchoring here means the extinction at
show conditions is unchanged by introducing humidity dependence, so the change
adds a response rather than silently re-tuning a calibration.
"""

DRY_TURBIDITY_BETA = 0.12 / ((1.0 - REFERENCE_HUMIDITY) ** -HYGROSCOPIC_GROWTH_EXPONENT)
"""Dry Angstrom turbidity, chosen so the ambient value at
``REFERENCE_HUMIDITY`` equals the 0.12 previously used as a fixed coefficient.
"""


def hygroscopic_growth_factor(
    relative_humidity: float,
    exponent: float = HYGROSCOPIC_GROWTH_EXPONENT,
) -> float:
    """Aerosol extinction relative to its dry value.

    Capped below saturation because the relation diverges there and fog is a
    different regime that this model does not cover.
    """

    bounded = min(max(relative_humidity, 0.0), 0.95)
    return (1.0 - bounded) ** -exponent


def aerosol_optical_depth(
    wavelength_nm: float | np.ndarray,
    turbidity_beta: float = 0.12,
    angstrom_alpha: float = 1.3,
    relative_humidity: float | None = None,
) -> np.ndarray:
    """Ångström aerosol optical depth.

    ``turbidity_beta`` is the optical depth at 1 um and ``angstrom_alpha`` the
    wavelength exponent. Supplying ``relative_humidity`` treats the turbidity
    as a **dry** value and grows it hygroscopically; omitting it treats the
    turbidity as already ambient.

    The functional form is standard; the coefficients are documented estimates
    for an urban site and are **not** a measurement of Seoul on 2024-10-05.
    """

    micrometres = np.asarray(wavelength_nm, dtype=np.float64) * 1e-3
    depth = max(turbidity_beta, 0.0) * micrometres ** (-angstrom_alpha)
    if relative_humidity is not None:
        depth = depth * hygroscopic_growth_factor(relative_humidity)
    return depth


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


STANDARD_LAPSE_RATE_K_PER_M = 0.0065
"""ISA environmental lapse rate.

A clear October night over water very likely carried a surface **inversion**
instead, which would bend sound downward and extend audible range. No sounding
for the event exists, so the standard profile is used and the inversion is
recorded as unmeasured rather than guessed at.
"""

MOLAR_MASS_DRY_AIR_KG_MOL = 0.028_964_4
UNIVERSAL_GAS_CONSTANT_J_MOL_K = 8.314_462_618
GRAVITY_M_S2 = 9.806_65

SURFACE_ROUGHNESS_LENGTH_M = 0.03
"""Aerodynamic roughness length for the river and its parkland banks.

Open water is nearer 0.0002 m and dense urban fabric nearer 1 m; 0.03 m is the
short-grass value appropriate to the Han River corridor. It is also the value
that reproduces the previously hardcoded 1.4x wind increase from 10 m to 100 m,
which is how that literal is retired without changing the trajectory: the log
profile gives 1.396 at this roughness.
"""

REFERENCE_WIND_HEIGHT_M = 10.0
"""Height of a standard meteorological wind observation."""

AEROSOL_SCALE_HEIGHT_M = 1_200.0
"""Height over which aerosol extinction falls by 1/e.

Aerosol is confined near the surface, unlike the 8,500 m molecular scale
height, which is why haze thickens toward the horizon faster than Rayleigh
scattering does.
"""

MOLECULAR_SCALE_HEIGHT_M = 8_500.0

KOSCHMIEDER_CONSTANT = 3.912
"""``-ln(0.02)``: the 2% contrast threshold defining meteorological visibility."""

def temperature_at_height_k(
    surface_temperature_k: float,
    height_m: float,
    lapse_rate_k_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
) -> float:
    return max(surface_temperature_k - lapse_rate_k_per_m * height_m, 1.0)


def pressure_at_height_pa(
    surface_pressure_pa: float,
    surface_temperature_k: float,
    height_m: float,
    lapse_rate_k_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
) -> float:
    """Barometric pressure with a constant lapse rate.

    Falls to the isothermal exponential when the lapse rate is zero, which is
    the limit an inversion-free night approaches.
    """

    if height_m == 0.0:
        return surface_pressure_pa
    exponent_scale = (
        GRAVITY_M_S2
        * MOLAR_MASS_DRY_AIR_KG_MOL
        / UNIVERSAL_GAS_CONSTANT_J_MOL_K
    )
    if lapse_rate_k_per_m <= 0.0:
        return surface_pressure_pa * math.exp(
            -exponent_scale * height_m / surface_temperature_k
        )
    ratio = max(
        1.0 - lapse_rate_k_per_m * height_m / surface_temperature_k, 1e-6
    )
    return surface_pressure_pa * ratio ** (exponent_scale / lapse_rate_k_per_m)


def wind_scale_at_height(
    height_m: float,
    reference_height_m: float = REFERENCE_WIND_HEIGHT_M,
    roughness_length_m: float = SURFACE_ROUGHNESS_LENGTH_M,
) -> float:
    """Neutral logarithmic surface-layer profile, relative to the reference."""

    z0 = max(roughness_length_m, 1e-6)
    height = max(height_m, z0 * 1.001)
    reference = max(reference_height_m, z0 * 1.001)
    return math.log(height / z0) / math.log(reference / z0)


def relative_humidity_at_height(
    surface_relative_humidity: float,
    surface_temperature_k: float,
    surface_pressure_pa: float,
    height_m: float,
    lapse_rate_k_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
) -> float:
    """Humidity aloft at a conserved water-vapour mixing ratio.

    Air carrying the same vapour becomes relatively more humid as it cools, so
    humidity rises with height even with no extra moisture, until it saturates.
    """

    temperature = temperature_at_height_k(
        surface_temperature_k, height_m, lapse_rate_k_per_m
    )
    pressure = pressure_at_height_pa(
        surface_pressure_pa, surface_temperature_k, height_m, lapse_rate_k_per_m
    )
    surface_saturation = _saturation_vapour_pressure_pa(surface_temperature_k)
    saturation = _saturation_vapour_pressure_pa(temperature)
    vapour = surface_relative_humidity * surface_saturation
    # Partial pressure scales with total pressure at fixed mixing ratio.
    vapour *= pressure / max(surface_pressure_pa, 1e-6)
    return min(max(vapour / max(saturation, 1e-6), 0.0), 1.0)


def _saturation_vapour_pressure_pa(temperature_k: float) -> float:
    """Magnus-Tetens form, matching :mod:`simulator.environment`."""

    celsius = temperature_k - 273.15
    return 611.2 * math.exp(17.67 * celsius / (celsius + 243.5))


def surface_extinction_per_m(
    column_optical_depth: float | np.ndarray, scale_height_m: float
) -> np.ndarray:
    """Extinction coefficient at the surface, from a whole-column depth.

    A column depth is the integral of the coefficient over an exponential
    profile, ``tau = sigma(0) * H``, so the surface value is the depth divided
    by the scale height of the species that produced it. Aerosol and molecules
    must be divided separately: they carry very different scale heights, and
    dividing their sum by either one would misattribute the extinction.
    """

    return np.maximum(np.asarray(column_optical_depth, dtype=np.float64), 0.0) / max(
        scale_height_m, 1.0
    )


LEVEL_PATH_RISE_M = 1e-3
"""Rise below which a segment is treated as level.

The closed form is singular at zero rise and a level path is its own limit.
``air_extinction.glsl`` uses the same threshold, so the CPU reference and the
shader take the same branch on the same inputs.
"""


def _monotonic_profile_mean(
    scale_height_m: float, low_m: float, high_m: float
) -> float:
    """Mean of ``exp(-z/H)`` over a segment climbing from ``low_m`` to ``high_m``.

    Symmetric in its arguments: swapping them flips the sign of both the
    numerator and the denominator.
    """

    rise = high_m - low_m
    at_start = math.exp(-low_m / scale_height_m)
    if abs(rise) < LEVEL_PATH_RISE_M:
        return at_start
    return (
        scale_height_m
        * (at_start - math.exp(-high_m / scale_height_m))
        / rise
    )


def slant_optical_depth(
    surface_extinction: float | np.ndarray,
    scale_height_m: float,
    start_height_m: float,
    end_height_m: float,
    path_length_m: float,
) -> np.ndarray:
    """Optical depth along a straight path through an exponential atmosphere.

    The integral ``∫ sigma(0) exp(-z/H) ds`` over a segment whose height varies
    linearly from ``start_height_m`` to ``end_height_m`` has a closed form,

    ``sigma(0) * L * H / dz * (exp(-z0/H) - exp(-z1/H))``,

    which degenerates to ``sigma(0) * L * exp(-z0/H)`` for a level path. Using
    it rather than a constant surface density matters for the fireworks: at a
    1.5 km range a 300 m break sits behind 12 percent less air than the level
    approximation claims, and a 600 m break behind 22 percent less.

    A segment that **crosses the water datum** is split there. The planar
    reflection draws stars and geometry from a camera mirrored below the datum,
    so the straight line to the image point has the length of the real
    two-segment path but its height falls to zero and rises again. Taking the
    endpoint magnitudes and assuming a monotonic climb is wrong by 1-2 percent
    of the optical depth over typical firework geometry; splitting at the
    crossing reproduces numerical quadrature to float precision.
    """

    extinction = np.asarray(surface_extinction, dtype=np.float64)
    height_scale = max(scale_height_m, 1.0)
    low = abs(start_height_m)
    high = abs(end_height_m)
    if start_height_m * end_height_m >= 0.0:
        profile = _monotonic_profile_mean(height_scale, low, high)
    else:
        # Weighting by path fraction is exact: each piece contributes its own
        # mean over its own share of the length.
        below = low / max(low + high, 1e-6)
        profile = below * _monotonic_profile_mean(
            height_scale, 0.0, low
        ) + (1.0 - below) * _monotonic_profile_mean(height_scale, 0.0, high)
    return extinction * max(path_length_m, 0.0) * profile


def visibility_m(
    aerosol_optical_depth_550nm: float,
    rayleigh_optical_depth_550nm: float,
    aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M,
    molecular_scale_height_m: float = MOLECULAR_SCALE_HEIGHT_M,
) -> float:
    """Koschmieder meteorological visibility from column optical depths.

    Surface extinction is recovered by dividing each column depth by the scale
    height of the species that produced it, because aerosol sits far lower in
    the atmosphere than the molecules do.
    """

    extinction_per_m = float(
        surface_extinction_per_m(
            aerosol_optical_depth_550nm, aerosol_scale_height_m
        )
        + surface_extinction_per_m(
            rayleigh_optical_depth_550nm, molecular_scale_height_m
        )
    )
    return KOSCHMIEDER_CONSTANT / max(extinction_per_m, 1e-12)


@dataclass(frozen=True, slots=True)
class SurfaceExtinction:
    """Per-channel extinction at the surface, with each species' scale height.

    This is the quantity the renderer needs and the quantity a visibility
    observation constrains, so it is the single object the haze, star, smoke,
    and point-light paths all read. Keeping aerosol and molecules apart rather
    than summing them lets each keep its own vertical profile: haze thickens
    toward the horizon roughly seven times faster than Rayleigh scattering
    does, which is why the two cannot share one exponential.
    """

    aerosol_per_m: tuple[float, float, float]
    molecular_per_m: tuple[float, float, float]
    aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M
    molecular_scale_height_m: float = MOLECULAR_SCALE_HEIGHT_M

    @property
    def total_per_m(self) -> tuple[float, float, float]:
        return tuple(
            float(aerosol + molecular)
            for aerosol, molecular in zip(
                self.aerosol_per_m, self.molecular_per_m
            )
        )

    @property
    def visibility_m(self) -> float:
        """Koschmieder range at 550 nm, the green channel this is evaluated at.

        Inverting this relation is how an observed visibility would calibrate
        the turbidity, which is the direction the project would prefer to run
        it in once such an observation exists.
        """

        return KOSCHMIEDER_CONSTANT / max(self.total_per_m[1], 1e-12)

    def transmittance(
        self,
        start_height_m: float,
        end_height_m: float,
        path_length_m: float,
    ) -> np.ndarray:
        """Per-channel transmittance along one straight path.

        The CPU reference for what ``haze.frag``, ``particle.vert``, and
        ``smoke.frag`` compute on the GPU, so the shader arithmetic has
        something testable to agree with.
        """

        depth = slant_optical_depth(
            np.asarray(self.aerosol_per_m),
            self.aerosol_scale_height_m,
            start_height_m,
            end_height_m,
            path_length_m,
        ) + slant_optical_depth(
            np.asarray(self.molecular_per_m),
            self.molecular_scale_height_m,
            start_height_m,
            end_height_m,
            path_length_m,
        )
        return np.exp(-depth)

    def uniforms(self) -> dict[str, tuple[float, float, float] | float]:
        """The uniform names every shader that consumes extinction declares.

        One mapping rather than five call sites means a renamed uniform fails
        to link instead of silently leaving one pass on a stale value.
        """

        return {
            "aerosol_extinction_per_m": self.aerosol_per_m,
            "molecular_extinction_per_m": self.molecular_per_m,
            "aerosol_scale_height_m": self.aerosol_scale_height_m,
            "molecular_scale_height_m": self.molecular_scale_height_m,
        }


DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
WATER_VAPOUR_GAS_CONSTANT_J_KG_K = 461.495


def moist_air_density(
    temperature_k: float, pressure_pa: float, relative_humidity: float
) -> float:
    """Density of humid air, as the sum of its dry and vapour partial densities.

    Moist air is lighter than dry air at the same pressure, because water
    vapour has a lower molar mass than the nitrogen and oxygen it displaces.
    """

    vapour_pressure = relative_humidity * _saturation_vapour_pressure_pa(
        temperature_k
    )
    dry_pressure = pressure_pa - vapour_pressure
    return dry_pressure / (
        DRY_AIR_GAS_CONSTANT_J_KG_K * temperature_k
    ) + vapour_pressure / (
        WATER_VAPOUR_GAS_CONSTANT_J_KG_K * temperature_k
    )


def moist_air_density_at_height_kg_m3(
    surface_temperature_k: float,
    surface_pressure_pa: float,
    surface_relative_humidity: float,
    height_m: float,
    lapse_rate_k_per_m: float = STANDARD_LAPSE_RATE_K_PER_M,
) -> float:
    """Air density aloft from surface observations."""

    return moist_air_density(
        temperature_at_height_k(
            surface_temperature_k, height_m, lapse_rate_k_per_m
        ),
        pressure_at_height_pa(
            surface_pressure_pa,
            surface_temperature_k,
            height_m,
            lapse_rate_k_per_m,
        ),
        relative_humidity_at_height(
            surface_relative_humidity,
            surface_temperature_k,
            surface_pressure_pa,
            height_m,
            lapse_rate_k_per_m,
        ),
    )


@dataclass(frozen=True, slots=True)
class AtmosphericOptics:
    """Clear-sky extinction for one atmospheric state.

    ``pressure_pa`` sets the molecular column and therefore the Rayleigh term.
    The aerosol and ozone parameters carry their own confidence grades because
    they are not measurements of the event.
    """

    pressure_pa: float = STANDARD_PRESSURE_PA
    turbidity_beta: float = DRY_TURBIDITY_BETA
    """Dry Angstrom turbidity; the ambient value grows with humidity."""

    angstrom_alpha: float = 1.3
    relative_humidity: float = REFERENCE_HUMIDITY
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
                wavelength_nm,
                self.turbidity_beta,
                self.angstrom_alpha,
                self.relative_humidity,
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

    def surface_extinction(
        self,
        aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M,
        molecular_scale_height_m: float = MOLECULAR_SCALE_HEIGHT_M,
    ) -> SurfaceExtinction:
        """Per-channel extinction at the surface, for the renderer's haze.

        The same optical depths that redden and dim the stars, divided by their
        scale heights. Aerial perspective and stellar extinction therefore
        cannot disagree: one weather state produces both.
        """

        wavelengths = np.asarray(RGB_WAVELENGTHS_NM)
        aerosol = surface_extinction_per_m(
            aerosol_optical_depth(
                wavelengths,
                self.turbidity_beta,
                self.angstrom_alpha,
                self.relative_humidity,
            ),
            aerosol_scale_height_m,
        )
        molecular = surface_extinction_per_m(
            rayleigh_optical_depth(wavelengths, self.pressure_pa),
            molecular_scale_height_m,
        )
        return SurfaceExtinction(
            aerosol_per_m=tuple(float(value) for value in aerosol),
            molecular_per_m=tuple(float(value) for value in molecular),
            aerosol_scale_height_m=aerosol_scale_height_m,
            molecular_scale_height_m=molecular_scale_height_m,
        )

    def visibility_m(
        self,
        aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M,
        molecular_scale_height_m: float = MOLECULAR_SCALE_HEIGHT_M,
    ) -> float:
        """Koschmieder visibility implied by this atmospheric state.

        **Not a measurement.** The event weather record carries no visibility
        observation; this is what the aerosol model implies.
        """

        return visibility_m(
            float(
                aerosol_optical_depth(
                    550.0,
                    self.turbidity_beta,
                    self.angstrom_alpha,
                    self.relative_humidity,
                )
            ),
            float(rayleigh_optical_depth(550.0, self.pressure_pa)),
            aerosol_scale_height_m,
            molecular_scale_height_m,
        )

    def with_visibility_m(
        self,
        observed_visibility_m: float,
        aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M,
        molecular_scale_height_m: float = MOLECULAR_SCALE_HEIGHT_M,
    ) -> "AtmosphericOptics":
        """Calibrate the dry turbidity against an observed visibility.

        The adapter for a dataset the project does not hold. Everywhere else
        the turbidity is a documented urban estimate and the visibility is
        whatever it implies; a reported visibility runs the relation the other
        way, which is the direction that turns a grade C estimate into a
        grade A constraint. The molecular term is fixed by the observed
        pressure and is subtracted first, so only the aerosol absorbs the
        residual — and the aerosol floors at zero rather than going negative
        when a reported visibility exceeds the Rayleigh limit.

        Humidity still modulates the result: the calibration sets the *dry*
        turbidity such that the ambient value at the current humidity matches
        the observation, so a later hour with different humidity moves as the
        model says it should.
        """

        molecular = float(
            surface_extinction_per_m(
                rayleigh_optical_depth(550.0, self.pressure_pa),
                molecular_scale_height_m,
            )
        )
        target = KOSCHMIEDER_CONSTANT / max(observed_visibility_m, 1e-6)
        aerosol_depth = max(target - molecular, 0.0) * aerosol_scale_height_m
        growth = hygroscopic_growth_factor(self.relative_humidity)
        dry_beta = aerosol_depth / (
            growth * (0.55 ** -self.angstrom_alpha)
        )
        return replace(self, turbidity_beta=dry_beta)

    def summary(self) -> dict[str, object]:
        rayleigh = float(rayleigh_optical_depth(550.0, self.pressure_pa))
        aerosol = float(
            aerosol_optical_depth(
                550.0,
                self.turbidity_beta,
                self.angstrom_alpha,
                self.relative_humidity,
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
            "surface_extinction_550nm_per_m": self.surface_extinction().total_per_m[
                1
            ],
            "visibility_km": self.visibility_m() / 1_000.0,
            "relative_humidity": self.relative_humidity,
            "hygroscopic_growth": hygroscopic_growth_factor(
                self.relative_humidity
            ),
            "aerosol_grade": self.aerosol_grade.value,
            "ozone_grade": self.ozone_grade.value,
        }


def from_atmosphere_config(config, **overrides) -> AtmosphericOptics:
    """Build optics from the runtime atmospheric state.

    Pressure and humidity are taken from the observed weather: pressure sets
    the molecular column and humidity grows the aerosol hygroscopically. The dry
    turbidity and the ozone terms are not observed and keep their documented
    defaults unless overridden.
    """

    return AtmosphericOptics(
        pressure_pa=config.pressure_pa,
        relative_humidity=config.relative_humidity,
        **overrides,
    )
