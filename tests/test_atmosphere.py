import math

import numpy as np
import pytest

from simulator.atmosphere import (
    MAGNITUDES_PER_OPTICAL_DEPTH,
    RGB_WAVELENGTHS_NM,
    STANDARD_PRESSURE_PA,
    AtmosphericOptics,
    aerosol_optical_depth,
    from_atmosphere_config,
    ozone_optical_depth,
    rayleigh_optical_depth,
    relative_air_mass,
)
from simulator.config import AtmosphereConfig
from simulator.provenance import ConfidenceGrade

# Tabulated Rayleigh optical depths at standard sea-level pressure.
PUBLISHED = {
    400.0: 0.3602,
    500.0: 0.1434,
    550.0: 0.0971,
    700.0: 0.0364,
}


def test_rayleigh_matches_published_optical_depths() -> None:
    # The Bodhaine et al. (1999) fit reproduces tabulated values to better than
    # 0.1 percent, so a 1 percent bound catches a transcription or unit error.
    for wavelength_nm, published in PUBLISHED.items():
        value = float(rayleigh_optical_depth(wavelength_nm))
        assert abs(value - published) / published < 1e-2, wavelength_nm


def test_rayleigh_follows_an_inverse_fourth_power() -> None:
    ratio = float(
        rayleigh_optical_depth(400.0) / rayleigh_optical_depth(800.0)
    )
    # Exactly 16 for a pure lambda^-4; the excess comes from the dispersion of
    # air's refractive index, which the fit carries.
    assert 16.0 < ratio < 17.5


def test_rayleigh_scales_linearly_with_station_pressure() -> None:
    # Optical depth is proportional to the molecular column above the observer.
    full = float(rayleigh_optical_depth(550.0, STANDARD_PRESSURE_PA))
    half = float(rayleigh_optical_depth(550.0, STANDARD_PRESSURE_PA * 0.5))
    assert half == pytest.approx(full * 0.5, rel=1e-12)


def test_rayleigh_is_vectorised() -> None:
    values = rayleigh_optical_depth(np.array([400.0, 550.0, 700.0]))
    assert values.shape == (3,)
    assert values[0] > values[1] > values[2]


def test_aerosol_follows_the_angstrom_exponent() -> None:
    # tau = beta * lambda^-alpha with lambda in micrometres, so beta is by
    # definition the optical depth at 1 um.
    assert float(aerosol_optical_depth(1000.0, 0.12, 1.3)) == pytest.approx(
        0.12, rel=1e-9
    )
    short = float(aerosol_optical_depth(400.0, 0.12, 1.3))
    long = float(aerosol_optical_depth(800.0, 0.12, 1.3))
    assert short / long == pytest.approx(2.0**1.3, rel=1e-9)


def test_aerosol_is_far_greyer_than_rayleigh() -> None:
    # An Angstrom exponent near 1.3 is much flatter than lambda^-4, which is
    # why haze whitens the sky while molecules blue it.
    rayleigh_ratio = float(
        rayleigh_optical_depth(400.0) / rayleigh_optical_depth(700.0)
    )
    aerosol_ratio = float(
        aerosol_optical_depth(400.0) / aerosol_optical_depth(700.0)
    )
    # Roughly 9.9 against 2.1 over the same band.
    assert rayleigh_ratio > 4.0 * aerosol_ratio


def test_ozone_is_zero_until_measured() -> None:
    # Neither the column amount nor the cross-sections are held for the event,
    # so the term must be an explicit zero rather than a plausible estimate.
    assert float(ozone_optical_depth(550.0)) == 0.0
    assert float(ozone_optical_depth(550.0, column_dobson=300.0)) == 0.0
    assert AtmosphericOptics().ozone_grade is ConfidenceGrade.UNVERIFIED
    supplied = AtmosphericOptics(
        ozone_column_dobson=300.0,
        ozone_cross_section_per_dobson=(1e-4, 1e-4, 1e-4),
    )
    assert supplied.ozone_grade is ConfidenceGrade.MODELLED


def test_aerosol_defaults_are_modelled_not_measured() -> None:
    assert AtmosphericOptics().aerosol_grade is ConfidenceGrade.MODELLED
    assert not AtmosphericOptics().aerosol_grade.is_evidence


def test_air_mass_matches_kasten_young_landmarks() -> None:
    assert relative_air_mass(90.0) == pytest.approx(1.0, abs=1e-3)
    assert relative_air_mass(30.0) == pytest.approx(2.0, abs=0.01)
    # Kasten-Young stays finite at the horizon where 1/cos(z) diverges.
    horizon = relative_air_mass(0.0)
    assert 30.0 < horizon < 45.0
    assert math.isfinite(relative_air_mass(-0.5))


def test_air_mass_increases_monotonically_toward_the_horizon() -> None:
    altitudes = [90.0, 60.0, 30.0, 15.0, 5.0, 0.0]
    masses = [relative_air_mass(value) for value in altitudes]
    assert all(a < b for a, b in zip(masses, masses[1:]))


def test_transmittance_is_bounded_and_wavelength_ordered() -> None:
    optics = AtmosphericOptics()
    zenith = optics.rgb_transmittance(90.0)
    horizon = optics.rgb_transmittance(5.0)
    assert np.all((zenith > 0.0) & (zenith <= 1.0))
    assert np.all(horizon < zenith)
    # Blue is scattered hardest, so it transmits least.
    assert zenith[2] < zenith[1] < zenith[0]
    assert RGB_WAVELENGTHS_NM[2] < RGB_WAVELENGTHS_NM[1] < RGB_WAVELENGTHS_NM[0]


def test_extinction_magnitudes_are_physically_plausible_for_a_city() -> None:
    optics = AtmosphericOptics()
    zenith = optics.zenith_extinction_magnitudes()
    # Dark sites measure roughly 0.15-0.20 mag at the zenith; a hazy urban site
    # sits well above that but below a magnitude.
    assert 0.2 < zenith < 0.8
    assert optics.extinction_magnitudes(20.0) > zenith
    # Kasten-Young returns 0.99971 rather than exactly 1 at the zenith, a known
    # 0.03 percent property of the fit. The extinction must carry that through
    # rather than special-casing the zenith.
    assert relative_air_mass(90.0) == pytest.approx(0.99971, abs=1e-5)
    assert zenith == pytest.approx(
        MAGNITUDES_PER_OPTICAL_DEPTH
        * float(optics.vertical_optical_depth(550.0))
        * relative_air_mass(90.0),
        rel=1e-9,
    )


def test_optics_track_the_observed_station_pressure() -> None:
    low = from_atmosphere_config(AtmosphereConfig(pressure_pa=95_000.0))
    high = from_atmosphere_config(AtmosphereConfig(pressure_pa=103_000.0))
    assert low.vertical_optical_depth(550.0) < high.vertical_optical_depth(550.0)
    # Only pressure is observed; aerosol and ozone keep their documented
    # defaults rather than being invented from the weather record.
    assert low.turbidity_beta == AtmosphericOptics().turbidity_beta
    assert low.ozone_column_dobson == 0.0


def test_summary_reports_each_term_and_its_grade() -> None:
    summary = AtmosphericOptics().summary()
    assert summary["rayleigh_optical_depth_550nm"] == pytest.approx(
        0.0971, abs=1e-3
    )
    assert summary["ozone_optical_depth_550nm"] == 0.0
    assert summary["aerosol_grade"] == "C"
    assert summary["ozone_grade"] == "U"
    assert summary["total_optical_depth_550nm"] > summary[
        "rayleigh_optical_depth_550nm"
    ]
