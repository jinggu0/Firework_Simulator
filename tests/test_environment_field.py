from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from simulator.atmosphere import (
    AEROSOL_SCALE_HEIGHT_M,
    DRY_TURBIDITY_BETA,
    KOSCHMIEDER_CONSTANT,
    MOLECULAR_SCALE_HEIGHT_M,
    REFERENCE_HUMIDITY,
    REFERENCE_WIND_HEIGHT_M,
    SURFACE_ROUGHNESS_LENGTH_M,
    AtmosphericOptics,
    hygroscopic_growth_factor,
    moist_air_density,
    moist_air_density_at_height_kg_m3,
    pressure_at_height_pa,
    relative_humidity_at_height,
    temperature_at_height_k,
    visibility_m,
    wind_scale_at_height,
)
from simulator.config import AtmosphereConfig
from simulator.environment import EnvironmentTimeline
from simulator.environment_field import EnvironmentField, StationTimelineField

ASSET = Path(__file__).resolve().parent.parent / "assets" / (
    "yeouido_2024-10-05_environment.json"
)
SHOW_EPOCH = datetime.fromisoformat("2024-10-05T19:30:00+09:00").timestamp()
GROUND = np.zeros(3)
BURST = np.array([0.0, 160.0, 0.0])
HIGH = np.array([0.0, 300.0, 0.0])


@pytest.fixture(scope="module")
def field() -> StationTimelineField:
    return StationTimelineField(EnvironmentTimeline.load(ASSET))


# --- vertical profiles ------------------------------------------------------


def test_temperature_falls_at_the_lapse_rate() -> None:
    assert temperature_at_height_k(288.15, 1_000.0) == pytest.approx(281.65)
    assert temperature_at_height_k(288.15, 0.0) == 288.15


def test_pressure_follows_the_barometric_relation() -> None:
    # ISA: 1013.25 hPa at sea level gives about 898.7 hPa at 1 km.
    pressure = pressure_at_height_pa(101_325.0, 288.15, 1_000.0)
    assert pressure / 100.0 == pytest.approx(898.7, abs=1.0)
    assert pressure_at_height_pa(101_325.0, 288.15, 0.0) == 101_325.0


def test_pressure_reduces_to_the_isothermal_case() -> None:
    # With no lapse rate the relation must become the exponential atmosphere.
    isothermal = pressure_at_height_pa(101_325.0, 288.15, 500.0, 0.0)
    assert isothermal < 101_325.0
    assert isothermal == pytest.approx(101_325.0 * np.exp(-500.0 / 8_434.5), rel=1e-3)


def test_wind_log_law_reproduces_the_literal_it_replaced() -> None:
    # The previous code raised the 10 m wind by a bare factor of 1.4 at 100 m.
    # At the river corridor's roughness length the log profile gives 1.396, so
    # naming the physics does not move the trajectory.
    assert wind_scale_at_height(100.0) == pytest.approx(1.4, abs=0.01)
    assert wind_scale_at_height(REFERENCE_WIND_HEIGHT_M) == pytest.approx(1.0)


def test_wind_vanishes_at_the_roughness_height() -> None:
    # No-slip: the surface-layer profile goes to zero at z0, not at z = 0.
    # The implementation floors the height just above z0 rather than at it,
    # because the logarithm is zero there and the profile is undefined below,
    # so the value is a residual 1.7e-4 rather than an exact zero.
    assert wind_scale_at_height(SURFACE_ROUGHNESS_LENGTH_M) < 1e-3
    assert wind_scale_at_height(0.0) < 1e-3
    assert wind_scale_at_height(0.0) >= 0.0


def test_wind_increases_monotonically_with_height() -> None:
    scales = [wind_scale_at_height(h) for h in (1, 10, 50, 100, 300, 1_000)]
    assert all(a < b for a, b in zip(scales, scales[1:]))


def test_humidity_rises_with_height_at_constant_mixing_ratio() -> None:
    # The same vapour in cooler air is relatively more humid.
    surface = 0.56
    aloft = relative_humidity_at_height(surface, 291.75, 102_125.0, 300.0)
    assert surface < aloft < 1.0


def test_humidity_stays_bounded() -> None:
    assert relative_humidity_at_height(0.99, 280.0, 101_325.0, 3_000.0) <= 1.0
    assert relative_humidity_at_height(0.0, 291.0, 101_325.0, 100.0) == 0.0


def test_density_falls_with_height() -> None:
    surface = moist_air_density_at_height_kg_m3(291.75, 102_125.0, 0.56, 0.0)
    burst = moist_air_density_at_height_kg_m3(291.75, 102_125.0, 0.56, 160.0)
    high = moist_air_density_at_height_kg_m3(291.75, 102_125.0, 0.56, 300.0)
    assert surface > burst > high
    # About 1.5% thinner at a typical break and 2.8% at 300 m.
    assert 0.010 < (surface - burst) / surface < 0.020
    assert 0.020 < (surface - high) / surface < 0.035


def test_moist_air_is_lighter_than_dry_air() -> None:
    # Water vapour displaces heavier nitrogen and oxygen.
    dry = moist_air_density(293.15, 101_325.0, 0.0)
    humid = moist_air_density(293.15, 101_325.0, 0.9)
    assert humid < dry


# --- aerosol, humidity, visibility ------------------------------------------


def test_hygroscopic_growth_increases_with_humidity() -> None:
    assert hygroscopic_growth_factor(0.0) == pytest.approx(1.0)
    values = [hygroscopic_growth_factor(rh) for rh in (0.2, 0.5, 0.7, 0.9)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_growth_is_capped_below_saturation() -> None:
    # The relation diverges at saturation; fog is a regime this does not model.
    assert hygroscopic_growth_factor(1.0) == hygroscopic_growth_factor(0.95)
    assert np.isfinite(hygroscopic_growth_factor(1.0))


def test_dry_turbidity_reproduces_the_previous_fixed_coefficient() -> None:
    # Anchoring at the observed humidity means introducing humidity dependence
    # adds a response rather than silently re-tuning the calibration.
    ambient = DRY_TURBIDITY_BETA * hygroscopic_growth_factor(REFERENCE_HUMIDITY)
    assert ambient == pytest.approx(0.12, rel=1e-9)


def test_extinction_now_responds_to_humidity() -> None:
    dry = AtmosphericOptics(relative_humidity=0.30)
    damp = AtmosphericOptics(relative_humidity=0.80)
    assert damp.zenith_extinction_magnitudes() > (
        dry.zenith_extinction_magnitudes()
    )
    # At the reference humidity the value is unchanged from before.
    reference = AtmosphericOptics(relative_humidity=REFERENCE_HUMIDITY)
    assert reference.summary()["aerosol_optical_depth_550nm"] == pytest.approx(
        0.12 * (0.55**-1.3), rel=1e-6
    )


def test_visibility_follows_koschmieder() -> None:
    # Halving the extinction doubles the visual range.
    near = visibility_m(0.30, 0.097)
    far = visibility_m(0.15, 0.097)
    assert far > near
    only_aerosol = visibility_m(0.26, 0.0)
    assert only_aerosol == pytest.approx(
        KOSCHMIEDER_CONSTANT / (0.26 / AEROSOL_SCALE_HEIGHT_M), rel=1e-9
    )


def test_aerosol_dominates_visibility_over_molecules() -> None:
    # Aerosol sits in a 1.2 km layer against the 8.5 km molecular scale height,
    # so haze closes the view down far faster than Rayleigh scattering does.
    assert AEROSOL_SCALE_HEIGHT_M < MOLECULAR_SCALE_HEIGHT_M
    aerosol_only = visibility_m(0.26, 0.0)
    molecules_only = visibility_m(0.0, 0.097)
    assert aerosol_only < molecules_only


# --- the field --------------------------------------------------------------


def test_station_field_satisfies_the_protocol(field) -> None:
    assert isinstance(field, EnvironmentField)


def test_field_is_horizontally_uniform_and_says_so(field) -> None:
    # One station carries no horizontal information; inventing structure from
    # it would be fabrication.
    assert field.is_horizontally_uniform
    east = np.array([2_000.0, 100.0, 0.0])
    south = np.array([0.0, 100.0, 2_000.0])
    assert field.temperature_k(east, SHOW_EPOCH) == field.temperature_k(
        south, SHOW_EPOCH
    )
    assert field.pressure_pa(east, SHOW_EPOCH) == field.pressure_pa(
        south, SHOW_EPOCH
    )


def test_field_varies_vertically(field) -> None:
    assert field.temperature_k(HIGH, SHOW_EPOCH) < field.temperature_k(
        GROUND, SHOW_EPOCH
    )
    assert field.pressure_pa(HIGH, SHOW_EPOCH) < field.pressure_pa(
        GROUND, SHOW_EPOCH
    )
    assert field.air_density_kg_m3(HIGH, SHOW_EPOCH) < field.air_density_kg_m3(
        GROUND, SHOW_EPOCH
    )
    assert np.linalg.norm(
        field.wind_eus_mps(HIGH, SHOW_EPOCH)
    ) > np.linalg.norm(field.wind_eus_mps(BURST, SHOW_EPOCH))


def test_field_varies_in_time(field) -> None:
    early = field.temperature_k(GROUND, SHOW_EPOCH)
    late = field.temperature_k(GROUND, SHOW_EPOCH + 3_600.0)
    # The record cools through the evening.
    assert late < early


def test_visibility_is_reported_and_plausible(field) -> None:
    kilometres = field.visibility_m(GROUND, SHOW_EPOCH) / 1_000.0
    # A clear autumn evening after a front; not fog and not pristine.
    assert 5.0 < kilometres < 60.0


def test_visibility_falls_as_humidity_rises(field) -> None:
    from dataclasses import replace

    damp = replace(field, dry_turbidity_beta=field.dry_turbidity_beta * 2.0)
    assert damp.visibility_m(GROUND, SHOW_EPOCH) < field.visibility_m(
        GROUND, SHOW_EPOCH
    )


def test_sample_returns_the_form_the_solvers_consume(field) -> None:
    state = field.sample(BURST, SHOW_EPOCH)
    assert isinstance(state, AtmosphereConfig)
    surface = field.sample(GROUND, SHOW_EPOCH)
    assert state.air_density_kg_m3 < surface.air_density_kg_m3
    assert state.pressure_pa < surface.pressure_pa


def test_summary_reports_both_ends_of_the_profile(field) -> None:
    summary = field.summary(SHOW_EPOCH)
    assert summary["horizontally_uniform"] is True
    assert summary["temperature_300m_k"] < summary["surface_temperature_k"]
    assert summary["density_300m_kg_m3"] < summary["surface_density_kg_m3"]
    assert summary["wind_scale_300m"] > 1.0
    assert summary["visibility_km"] > 0.0


def test_config_exposes_density_aloft() -> None:
    # The consumer that matters: shell drag through the climb.
    config = AtmosphereConfig(
        temperature_k=291.75, pressure_pa=102_125.0, relative_humidity=0.56
    )
    assert config.air_density_at_height_kg_m3(
        300.0
    ) < config.air_density_at_height_kg_m3(0.0)
