"""Surface extinction, the slant-path integral, and the haze pass's contract.

The GPU side is checked by V-22, which renders the same frame with and without
the atmosphere and predicts the first from the second. These tests cover the
CPU reference that prediction is built from, so a formula error is caught here
rather than showing up as an unexplained residual there.
"""

from datetime import datetime
import math
from pathlib import Path

import numpy as np
import pytest

from simulator.atmosphere import (
    AEROSOL_SCALE_HEIGHT_M,
    KOSCHMIEDER_CONSTANT,
    MOLECULAR_SCALE_HEIGHT_M,
    RGB_WAVELENGTHS_NM,
    AtmosphericOptics,
    SurfaceExtinction,
    from_atmosphere_config,
    slant_optical_depth,
    surface_extinction_per_m,
)
from simulator.environment import EnvironmentTimeline
from simulator.environment_field import StationTimelineField
from simulator.passes import haze, particles

ASSET = Path(__file__).resolve().parent.parent / "assets" / (
    "yeouido_2024-10-05_environment.json"
)
SHOW_EPOCH = datetime.fromisoformat("2024-10-05T19:30:00+09:00").timestamp()


@pytest.fixture(scope="module")
def optics() -> AtmosphericOptics:
    timeline = EnvironmentTimeline.load(ASSET)
    return from_atmosphere_config(timeline.sample(SHOW_EPOCH))


# --- the surface coefficient ------------------------------------------------


def test_surface_extinction_inverts_the_column_integral() -> None:
    # tau = sigma(0) * H is the integral of an exponential profile, so the
    # surface value is the depth divided by that species' scale height.
    assert surface_extinction_per_m(0.26, 1_300.0) == pytest.approx(2e-4)
    assert surface_extinction_per_m(-1.0, 1_000.0) == 0.0


def test_the_two_species_keep_separate_scale_heights(optics) -> None:
    # Aerosol sits in a 1.2 km layer against 8.5 km of molecules. Summing the
    # depths first and dividing once would misattribute the extinction.
    extinction = optics.surface_extinction()
    assert extinction.aerosol_scale_height_m == AEROSOL_SCALE_HEIGHT_M
    assert extinction.molecular_scale_height_m == MOLECULAR_SCALE_HEIGHT_M
    for aerosol, molecular in zip(
        extinction.aerosol_per_m, extinction.molecular_per_m
    ):
        assert aerosol > molecular


def test_extinction_is_stronger_at_shorter_wavelengths(optics) -> None:
    # Channels are ordered red, green, blue.
    red, green, blue = optics.surface_extinction().total_per_m
    assert red < green < blue
    assert RGB_WAVELENGTHS_NM[0] > RGB_WAVELENGTHS_NM[2]


def test_visibility_agrees_across_every_path_that_reports_it(optics) -> None:
    # Three call sites once computed this three ways; they now share one.
    field = StationTimelineField(EnvironmentTimeline.load(ASSET))
    from_field = field.visibility_m(np.zeros(3), SHOW_EPOCH)
    assert optics.visibility_m() == pytest.approx(from_field, rel=1e-12)
    assert optics.surface_extinction().visibility_m == pytest.approx(
        from_field, rel=1e-12
    )
    assert 15_000.0 < from_field < 20_000.0


def test_visibility_is_the_koschmieder_inverse(optics) -> None:
    extinction = optics.surface_extinction()
    assert extinction.visibility_m == pytest.approx(
        KOSCHMIEDER_CONSTANT / extinction.total_per_m[1], rel=1e-12
    )


def test_the_retired_literal_would_have_doubled_the_visible_range(
    optics,
) -> None:
    # The renderer's old fixed 0.00012 per metre implies 32.6 km. Recording the
    # gap is the point: it was not a small correction.
    retired = KOSCHMIEDER_CONSTANT / 0.00012
    assert retired / optics.visibility_m() > 1.7


# --- the slant path ---------------------------------------------------------


def _quadrature(
    extinction: float, scale_height_m: float, start: float, end: float, length: float
) -> float:
    samples = np.linspace(0.0, length, 200_001)
    heights = abs(start) + (abs(end) - abs(start)) * samples / max(length, 1e-12)
    return float(
        np.trapezoid(extinction * np.exp(-heights / scale_height_m), samples)
    )


@pytest.mark.parametrize(
    "start,end,length",
    [(10.0, 10.0, 1_500.0), (10.0, 300.0, 1_520.0), (2.0, 160.0, 900.0),
     (10.0, 0.0, 2_500.0), (150.0, 5.0, 800.0), (24.0, -220.0, 1_100.0)],
)
def test_slant_optical_depth_matches_numerical_quadrature(
    start, end, length
) -> None:
    # The closed form is the whole reason the shader can do this in a few
    # instructions, so it has to be the same integral.
    analytic = float(
        slant_optical_depth(2.2904e-4, AEROSOL_SCALE_HEIGHT_M, start, end, length)
    )
    assert analytic == pytest.approx(
        _quadrature(2.2904e-4, AEROSOL_SCALE_HEIGHT_M, start, end, length),
        rel=1e-9,
    )


def test_a_level_path_is_the_zero_rise_limit() -> None:
    # The quotient is singular at equal heights; approaching it must not jump.
    height, length = 40.0, 1_000.0
    level = float(
        slant_optical_depth(1e-4, AEROSOL_SCALE_HEIGHT_M, height, height, length)
    )
    nearly = float(
        slant_optical_depth(
            1e-4, AEROSOL_SCALE_HEIGHT_M, height, height + 1e-4, length
        )
    )
    assert level == pytest.approx(nearly, rel=1e-6)
    assert level == pytest.approx(
        1e-4 * length * math.exp(-height / AEROSOL_SCALE_HEIGHT_M)
    )


def test_a_mirrored_reflection_path_crosses_real_air() -> None:
    # The water reflection draws stars at negative height. Taking heights as
    # magnitudes keeps the profile above the datum instead of inflating the
    # density below it.
    above = slant_optical_depth(1e-4, AEROSOL_SCALE_HEIGHT_M, 10.0, 200.0, 900.0)
    mirrored = slant_optical_depth(
        1e-4, AEROSOL_SCALE_HEIGHT_M, 10.0, -200.0, 900.0
    )
    assert float(mirrored) == pytest.approx(float(above))


def test_height_structure_thins_the_path_to_a_high_break(optics) -> None:
    # A level approximation over the same path length would overstate the haze
    # in front of a burst by about a tenth at 300 m.
    extinction = optics.surface_extinction()
    range_m = 1_500.0
    high = extinction.transmittance(10.0, 300.0, math.hypot(range_m, 290.0))
    level = extinction.transmittance(10.0, 10.0, math.hypot(range_m, 290.0))
    assert np.all(high > level)
    depth_ratio = np.log(high) / np.log(level)
    assert 0.85 < float(depth_ratio[1]) < 0.92


def test_transmittance_falls_to_the_contrast_threshold_at_the_visible_range(
    optics,
) -> None:
    # Koschmieder's definition: 2% of the original contrast at the visibility.
    # The published constant is 3.912, a rounding of -ln(0.02) = 3.9120230, so
    # the recovered threshold is 0.0200005. Keeping the tabulated value rather
    # than the exact logarithm is deliberate; the 2.3e-5 relative offset is the
    # constant's own precision.
    extinction = optics.surface_extinction()
    level = extinction.transmittance(0.0, 0.0, extinction.visibility_m)
    assert float(level[1]) == pytest.approx(0.02, rel=1e-4)


def test_a_vacuum_transmits_everything() -> None:
    vacuum = SurfaceExtinction((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert np.allclose(vacuum.transmittance(10.0, 500.0, 5_000.0), 1.0)
    assert math.isinf(vacuum.visibility_m) or vacuum.visibility_m > 1e10


# --- calibrating against an observation -------------------------------------


def test_an_observed_visibility_round_trips_through_the_turbidity(
    optics,
) -> None:
    # The adapter for a dataset the project does not hold: reported visibility
    # in, calibrated dry turbidity out, and the model then reproduces it.
    for observed_km in (4.0, 10.0, 17.0, 30.0):
        calibrated = optics.with_visibility_m(observed_km * 1_000.0)
        assert calibrated.visibility_m() == pytest.approx(
            observed_km * 1_000.0, rel=1e-9
        )


def test_calibration_leaves_the_humidity_response_intact(optics) -> None:
    # Calibration sets the *dry* turbidity, so a damper hour still hazes more.
    from dataclasses import replace

    calibrated = optics.with_visibility_m(12_000.0)
    damper = replace(calibrated, relative_humidity=0.85)
    assert damper.visibility_m() < calibrated.visibility_m()


def test_calibration_floors_the_aerosol_rather_than_going_negative(
    optics,
) -> None:
    # A reported visibility beyond the Rayleigh limit cannot be met by removing
    # more aerosol than exists.
    beyond = optics.with_visibility_m(10_000_000.0)
    assert beyond.turbidity_beta == 0.0
    assert beyond.visibility_m() < 10_000_000.0


# --- the renderer's contract ------------------------------------------------


def test_every_air_path_declares_the_same_uniform_names(optics) -> None:
    # set_air_extinction pushes this mapping to five programs; a name that
    # drifts in one shader must fail to link, not render stale weather.
    uniforms = optics.surface_extinction().uniforms()
    assert set(uniforms) == {
        "aerosol_extinction_per_m",
        "molecular_extinction_per_m",
        "aerosol_scale_height_m",
        "molecular_scale_height_m",
    }
    assert len(uniforms["aerosol_extinction_per_m"]) == 3
    assert isinstance(uniforms["aerosol_scale_height_m"], float)


def test_set_air_extinction_writes_every_program(optics) -> None:
    class FakeUniform:
        def __init__(self) -> None:
            self.value = None

    class FakeProgram(dict):
        def __missing__(self, key):
            uniform = FakeUniform()
            self[key] = uniform
            return uniform

    programs = (FakeProgram(), FakeProgram())
    extinction = optics.surface_extinction()
    haze.set_air_extinction(programs, extinction)
    for program in programs:
        assert program["aerosol_extinction_per_m"].value == (
            extinction.aerosol_per_m
        )
        assert program["molecular_scale_height_m"].value == (
            MOLECULAR_SCALE_HEIGHT_M
        )


def test_the_haze_stages_are_ordered_and_distinct() -> None:
    # Multiplying by the transmittance after adding the airlight would scale
    # the airlight too, so the constants must not be interchangeable.
    assert haze.EXTINCTION_STAGE != haze.AIRLIGHT_STAGE
    assert haze.EXTINCTION_STAGE == 0.0


def test_the_particle_pass_takes_a_camera_position() -> None:
    # Stars are attenuated over their own path in the vertex stage, which needs
    # the origin of that path.
    assert hasattr(particles.ParticlePass, "set_camera_position")
