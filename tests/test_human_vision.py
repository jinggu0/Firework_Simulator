import math
import re

import pytest

from simulator import shaders
from simulator.human_vision import (
    ACUITY_E2_DEG,
    DARK_ADAPTATION_TIME_S,
    LIGHT_ADAPTATION_TIME_S,
    PHOTOPIC_FLOOR_CD_M2,
    PHOTOPIC_PEAK_NM,
    SCOTOPIC_CEILING_CD_M2,
    SCOTOPIC_PEAK_NM,
    HumanVisionState,
    acuity_fraction,
    adapt,
    mesopic_factor,
    pupil_area_mm2,
    pupil_diameter_mm,
    retinal_illuminance_td,
    veiling_luminance_cd_m2,
)
from simulator.passes.post import DisplayMode


# --- pupil ------------------------------------------------------------------


def test_pupil_spans_the_physiological_range() -> None:
    # Stanley and Davies (1995). A dark-adapted pupil reaches 7-8 mm; a
    # photopic one closes to 2-3 mm.
    dark = pupil_diameter_mm(1e-4)
    bright = pupil_diameter_mm(1_000.0)
    assert 7.0 < dark < 8.0
    assert 2.0 < bright < 3.5
    assert dark > bright


def test_pupil_narrows_monotonically_with_luminance() -> None:
    luminances = [1e-4, 1e-2, 1.0, 100.0, 10_000.0]
    diameters = [pupil_diameter_mm(value) for value in luminances]
    assert all(a > b for a, b in zip(diameters, diameters[1:]))


def test_pupil_area_follows_the_diameter() -> None:
    assert pupil_area_mm2(4.0) == pytest.approx(math.pi * 4.0)


def test_a_wider_pupil_delivers_more_light_but_sublinearly() -> None:
    # The Stiles-Crawford effect: light entering near the pupil margin excites
    # cones less efficiently, so trolands grow more slowly than pupil area.
    dark_diameter = pupil_diameter_mm(1e-4)
    geometric_ratio = pupil_area_mm2(dark_diameter) / pupil_area_mm2(3.0)
    effective_ratio = retinal_illuminance_td(1.0, dark_diameter) / (
        retinal_illuminance_td(1.0, 3.0)
    )
    assert effective_ratio < geometric_ratio


# --- mesopic range ----------------------------------------------------------


def test_mesopic_bounds_follow_cie_191() -> None:
    assert mesopic_factor(SCOTOPIC_CEILING_CD_M2 * 0.5) == 0.0
    assert mesopic_factor(PHOTOPIC_FLOOR_CD_M2 * 2.0) == 1.0
    assert 0.0 < mesopic_factor(0.1) < 1.0


def test_cone_contribution_increases_with_luminance() -> None:
    values = [mesopic_factor(v) for v in (1e-4, 0.01, 0.1, 1.0, 10.0)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_the_purkinje_peaks_are_recorded_and_separated() -> None:
    # The 48 nm gap is why rod vision favours short wavelengths.
    assert SCOTOPIC_PEAK_NM < PHOTOPIC_PEAK_NM
    assert PHOTOPIC_PEAK_NM - SCOTOPIC_PEAK_NM == pytest.approx(48.0)


# --- adaptation -------------------------------------------------------------


def test_adaptation_is_asymmetric() -> None:
    # Brightening is fast and darkening is slow, which is what makes a burst
    # dazzle and the recovery take the gap between shells.
    assert LIGHT_ADAPTATION_TIME_S < DARK_ADAPTATION_TIME_S
    brighten = adapt(1.0, 100.0, 1.0)
    darken = adapt(100.0, 1.0, 1.0)
    assert (brighten - 1.0) / 99.0 > (100.0 - darken) / 99.0


def test_adaptation_converges_to_its_target() -> None:
    state = 0.01
    for _ in range(2_000):
        state = adapt(state, 5.0, 0.1)
    assert state == pytest.approx(5.0, rel=1e-3)


def test_zero_timestep_does_not_move_the_state() -> None:
    assert adapt(3.0, 100.0, 0.0) == 3.0


def test_a_burst_constricts_the_pupil_within_a_tenth_of_a_second() -> None:
    state = HumanVisionState()
    before = state.pupil_diameter_mm
    for _ in range(6):
        state.update(1_000.0, 1.0 / 60.0)
    after = state.pupil_diameter_mm
    assert before > 7.0
    assert after < 5.0


def test_recovery_after_a_burst_is_slow() -> None:
    state = HumanVisionState()
    for _ in range(6):
        state.update(1_000.0, 1.0 / 60.0)
    dazzled = state.adapting_luminance_cd_m2
    for _ in range(60):  # one second of darkness
        state.update(0.05, 1.0 / 60.0)
    # Still far above the ambient level it is heading toward.
    assert state.adapting_luminance_cd_m2 > dazzled * 0.9
    assert state.adapting_luminance_cd_m2 < dazzled


def test_observer_at_the_show_is_mesopic() -> None:
    # The show's ambient illuminance places the observer between rod and cone
    # vision, which is the regime the whole mode exists to represent.
    state = HumanVisionState()
    for _ in range(600):
        state.update(0.5, 1.0 / 60.0)
    assert 0.0 < state.cone_fraction < 1.0
    assert state.summary()["regime"] == "mesopic"


# --- acuity and glare -------------------------------------------------------


def test_acuity_halves_at_the_stated_eccentricity() -> None:
    assert acuity_fraction(0.0) == 1.0
    assert acuity_fraction(ACUITY_E2_DEG) == pytest.approx(0.5)
    assert acuity_fraction(30.0) < 0.1


def test_glare_follows_an_inverse_square_in_angle() -> None:
    # Stiles-Holladay: halving the separation quadruples the veil.
    near = veiling_luminance_cd_m2(100.0, 2.0)
    far = veiling_luminance_cd_m2(100.0, 4.0)
    assert near / far == pytest.approx(4.0)


def test_glare_stays_finite_at_zero_separation() -> None:
    assert math.isfinite(veiling_luminance_cd_m2(1e6, 0.0))


def test_glare_scales_with_source_illuminance() -> None:
    assert veiling_luminance_cd_m2(200.0, 3.0) == pytest.approx(
        2.0 * veiling_luminance_cd_m2(100.0, 3.0)
    )


# --- shader contract --------------------------------------------------------


def test_every_state_uniform_is_declared_by_the_shader() -> None:
    source = shaders.source("human_vision.frag")
    for name in HumanVisionState().uniforms():
        assert re.search(rf"uniform\s+\w+\s+{name}\s*;", source), name


def test_the_two_display_paths_are_separate_shaders() -> None:
    assert "human_vision.frag" in shaders.available()
    assert "tonemap.frag" in shaders.available()
    assert "adaptation.frag" in shaders.available()
    # The observer path must not carry sensor concepts.
    vision = shaders.source("human_vision.frag")
    for sensor_term in (
        "full_well_electrons",
        "read_noise_electrons",
        "photon_to_electron",
        "analog_gain",
    ):
        assert sensor_term not in vision, sensor_term


def test_the_camera_path_carries_no_observer_concepts() -> None:
    camera = shaders.source("tonemap.frag")
    for observer_term in ("pupil_gain", "cone_fraction", "acuity_e2_deg"):
        assert observer_term not in camera, observer_term


def test_display_modes_are_distinct_and_switchable() -> None:
    assert DisplayMode.PHYSICAL_CAMERA is not DisplayMode.HUMAN_VISION
    assert {mode.value for mode in DisplayMode} == {
        "physical_camera",
        "human_vision",
    }


def test_uniform_payload_is_all_scalars() -> None:
    # The shader consumes psychophysics as plain floats; anything structured
    # would mean the model had leaked into GLSL.
    for name, value in HumanVisionState().uniforms().items():
        assert isinstance(value, float), name
