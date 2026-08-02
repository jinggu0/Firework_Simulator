"""Lens distortion, white balance, and the calibration adapter.

The GPU side is checked by V-23, which reproduces the whole display transform
on the CPU from the linear buffers. These tests cover the model that reference
is built from, plus the guards on loading someone else's calibration.
"""

from dataclasses import replace
import json
import re

import numpy as np
import pytest

from simulator import shaders
from simulator.camera_optics import (
    DISTORTION_INVERSE_ITERATIONS,
    LensDistortion,
    analog_gain,
    frame_half_extent,
    load_lens_calibration,
    photon_to_electron_scale,
    white_balance_gains,
)
from simulator.color import (
    _XYZ_TO_LINEAR_SRGB,
    blackbody_rgb,
    cie_xyz_at_wavelength,
    planckian_linear_srgb,
    planckian_spectral_radiance,
)
from simulator.config import PhysicalCameraConfig

CONFIG = PhysicalCameraConfig()
HALF_EXTENT = frame_half_extent(CONFIG)


# --- the Planckian reference ------------------------------------------------


def test_planck_law_peaks_where_wien_says_it_should() -> None:
    # Wien's displacement: lambda_max * T = 2.898e-3 m K. At 5000 K that is
    # 580 nm, which is a check on the law rather than on the integration.
    wavelengths = np.arange(300.0, 1_200.0, 0.5)
    radiance = planckian_spectral_radiance(wavelengths, 5_000.0)
    peak_nm = float(wavelengths[int(np.argmax(radiance))])
    assert peak_nm == pytest.approx(2.897_771e-3 / 5_000.0 * 1e9, rel=1e-3)


def test_a_hotter_radiator_is_bluer() -> None:
    ratios = [
        planckian_linear_srgb(T)[2] / planckian_linear_srgb(T)[0]
        for T in (2_000.0, 3_200.0, 5_000.0, 6_504.0)
    ]
    assert all(a < b for a, b in zip(ratios, ratios[1:]))


def test_the_integration_agrees_with_the_shipped_curve_fit() -> None:
    # blackbody_rgb is a convenient fit to the same locus. Agreement to a few
    # percent is what makes it usable for star hue; the integration is used
    # where a calibration depends on it.
    for temperature in (3_200.0, 5_000.0, 6_504.0, 10_000.0):
        integrated = planckian_linear_srgb(temperature)
        fitted = np.asarray(blackbody_rgb(temperature), dtype=np.float64)
        assert np.abs(integrated - fitted).max() < 0.06, temperature


def test_the_reference_is_normalised_to_its_peak() -> None:
    for temperature in (1_500.0, 6_504.0, 20_000.0):
        assert planckian_linear_srgb(temperature).max() == pytest.approx(1.0)


def test_a_6504k_planckian_is_not_exactly_the_srgb_white() -> None:
    # D65 is a daylight illuminant with solar and atmospheric line structure;
    # a black body at the same correlated temperature is not the same spectrum.
    # The 5% gap is why the white balance gains are not simply 1/QE.
    reference = planckian_linear_srgb(6_504.0)
    assert np.abs(reference - 1.0).max() > 0.02
    assert np.abs(reference - 1.0).max() < 0.10


# --- white balance ----------------------------------------------------------


def test_the_sensor_response_was_leaving_a_yellow_green_cast() -> None:
    # This is the defect the balance corrects: quantum efficiency and photon
    # energy together turn a neutral scene into a strongly green-weighted
    # electron count.
    response = photon_to_electron_scale(CONFIG).astype(np.float64)
    normalised = response / response[1]
    assert normalised[0] == pytest.approx(0.896, abs=0.005)
    assert normalised[2] == pytest.approx(0.579, abs=0.005)


def test_the_balance_renders_the_reference_illuminant_neutral() -> None:
    # The whole point, stated as a round trip: scene radiance equal to the
    # reference illuminant must leave the sensor stage achromatic.
    gains = white_balance_gains(CONFIG).astype(np.float64)
    reference = planckian_linear_srgb(CONFIG.white_balance_temperature_k)
    signal = reference * photon_to_electron_scale(CONFIG).astype(np.float64) * gains
    assert signal.max() / signal.min() == pytest.approx(1.0, rel=1e-6)


def test_green_is_the_unity_gain_channel() -> None:
    # Exposure is set on the luminance-carrying channel, so the balance must
    # not silently change it.
    assert white_balance_gains(CONFIG)[1] == pytest.approx(1.0)


def test_a_warmer_balance_needs_more_blue() -> None:
    warm = white_balance_gains(
        replace(CONFIG, white_balance_temperature_k=3_200.0)
    )
    daylight = white_balance_gains(CONFIG)
    assert warm[2] > daylight[2] > 1.0
    assert warm[0] < daylight[0]


def test_the_shipped_balance_matches_the_documented_figure() -> None:
    # Quoted in docs/ARCHITECTURE.md and in the commit that introduced it.
    assert np.allclose(
        white_balance_gains(CONFIG), (1.0548, 1.0, 1.6420), atol=5e-4
    )


def test_analog_gain_is_untouched_by_the_balance() -> None:
    # Two independent stages; conflating them would make ISO change hue.
    assert analog_gain(CONFIG) == pytest.approx(8.0)


def test_a_camera_colour_matrix_would_be_a_second_conversion() -> None:
    # Recorded as a test because it is the evidence for *not* shipping one:
    # the renderer's values already carry sRGB primaries, so a matrix built
    # from the channel wavelengths and the CIE observer is far from identity.
    basis = np.array(
        [cie_xyz_at_wavelength(float(w)) for w in CONFIG.wavelength_rgb_nm]
    ).T
    matrix = _XYZ_TO_LINEAR_SRGB @ basis
    assert np.abs(np.diag(matrix) - 1.0).max() > 0.4
    assert matrix[0, 0] == pytest.approx(2.472, abs=0.01)


# --- lens distortion --------------------------------------------------------


def test_the_shipped_lens_is_identity_and_exactly_so() -> None:
    # No calibration exists. An invented coefficient would put a fabricated
    # optical claim into every frame, and a merely near-identity default would
    # perturb the camera path for no reason.
    lens = LensDistortion.from_config(CONFIG)
    assert lens.is_identity
    points = np.array([[0.3, -0.2], [0.0, 0.0], [0.75, 0.421875]])
    assert np.array_equal(lens.undistort(points), points)
    assert np.array_equal(lens.distort(points), points)
    assert lens.frame_coverage(HALF_EXTENT) == 1.0
    assert lens.inverse_residual(HALF_EXTENT) == 0.0


def test_the_forward_model_follows_the_opencv_convention() -> None:
    # Hand-evaluated so a transposed or reordered coefficient is caught:
    # x_d = x (1 + k1 r^2) + 2 p1 x y + p2 (r^2 + 2 x^2).
    lens = LensDistortion(k1=-0.2, p1=0.01, p2=0.02)
    x, y = 0.4, 0.3
    r2 = x * x + y * y
    expected_x = x * (1.0 + lens.k1 * r2) + 2.0 * lens.p1 * x * y + lens.p2 * (
        r2 + 2.0 * x * x
    )
    expected_y = y * (1.0 + lens.k1 * r2) + lens.p1 * (
        r2 + 2.0 * y * y
    ) + 2.0 * lens.p2 * x * y
    assert lens.distort(np.array([x, y])) == pytest.approx(
        [expected_x, expected_y]
    )


def test_undistort_inverts_distort_for_an_ordinary_lens() -> None:
    lens = LensDistortion(k1=-0.12, k2=0.03, p1=1e-4, p2=-2e-4)
    grid = LensDistortion._frame_grid(HALF_EXTENT)
    assert np.abs(lens.distort(lens.undistort(grid)) - grid).max() < 1e-4


def test_a_strong_barrel_does_not_converge_and_says_so() -> None:
    # Five fixed-point steps is what OpenCV uses and it is not universal. The
    # residual is reported rather than assumed so a calibration that needs more
    # iterations is a number, not a silently warped image. V-23 gates on it.
    assert LensDistortion(k1=-0.25).inverse_residual(HALF_EXTENT) > 1e-4
    assert LensDistortion(k1=-0.05).inverse_residual(HALF_EXTENT) < 1e-6


def test_barrel_distortion_needs_overscan_and_pincushion_does_not() -> None:
    # Barrel pulls the image inward, so the output corners ask for scene that
    # an ideal render does not contain.
    barrel = LensDistortion(k1=-0.12)
    assert barrel.frame_coverage(HALF_EXTENT) < 0.9
    assert LensDistortion(k1=0.08).frame_coverage(HALF_EXTENT) == 1.0


def test_the_shipped_lens_asks_for_no_overscan_at_all() -> None:
    # Exactly 1.0, not merely close: any other value would resize every render
    # target and stop the camera path being bit-identical.
    assert LensDistortion().required_overscan(HALF_EXTENT) == 1.0
    assert LensDistortion(k1=0.08).required_overscan(HALF_EXTENT) == 1.0


def test_overscan_is_derived_from_the_lens_and_restores_full_coverage() -> None:
    # The whole point: widen the rendered field until every output pixel has
    # something real to sample.
    for lens in (
        LensDistortion(k1=-0.12, k2=0.03),
        LensDistortion(k1=-0.18, k2=0.03),
        LensDistortion(k1=-0.12, k2=0.03, p1=1e-4, p2=-2e-4),
    ):
        overscan = lens.required_overscan(HALF_EXTENT)
        assert overscan > 1.0
        widened = (HALF_EXTENT[0] * overscan, HALF_EXTENT[1] * overscan)
        assert lens.frame_coverage(HALF_EXTENT) < 0.9
        assert lens.frame_coverage(HALF_EXTENT, widened) == 1.0


def test_overscan_costs_area_not_just_width() -> None:
    # Recording the price: the render is widened on both axes, so a 9 percent
    # wider field is a 19 percent larger frame.
    overscan = LensDistortion(k1=-0.12, k2=0.03).required_overscan(HALF_EXTENT)
    assert overscan == pytest.approx(1.0897, abs=1e-3)
    assert overscan**2 == pytest.approx(1.187, abs=1e-3)


def test_distortion_is_a_visible_effect_when_calibrated() -> None:
    # Recording the scale: a mild barrel moves the frame corner by tens of
    # pixels, which is why leaving the stage out was a real gap rather than a
    # formality.
    lens = LensDistortion(k1=-0.18, k2=0.03)
    corner = np.asarray(HALF_EXTENT)
    shift = (lens.distort(corner) - corner) / corner * np.array([640.0, 360.0])
    assert abs(shift[0]) > 30.0


def test_frame_half_extent_matches_the_projection() -> None:
    # The shader builds the same quantity from tan_half_fov and aspect; if the
    # two disagreed the CPU and GPU would distort different coordinates.
    assert HALF_EXTENT[1] == pytest.approx(
        CONFIG.sensor_height_mm / (2.0 * CONFIG.focal_length_mm)
    )
    assert HALF_EXTENT[0] / HALF_EXTENT[1] == pytest.approx(
        CONFIG.sensor_width_mm / CONFIG.sensor_height_mm
    )


def test_the_shader_iterates_the_same_number_of_times() -> None:
    source = shaders.source("tonemap.frag")
    match = re.search(r"const int DISTORTION_ITERATIONS = (\d+);", source)
    assert match is not None
    assert int(match.group(1)) == DISTORTION_INVERSE_ITERATIONS


def test_the_distortion_uniforms_are_the_ones_the_shader_declares() -> None:
    source = shaders.source("tonemap.frag")
    for name in LensDistortion().uniforms():
        assert re.search(rf"^uniform \w+ {name};", source, re.MULTILINE), name
    assert "uniform vec3 white_balance_gain;" in source


# --- the calibration adapter ------------------------------------------------


def _calibration(**overrides) -> dict:
    payload = {
        "source_id": "checkerboard-2026-01-04",
        "source_url": "https://example.invalid/calibration",
        "license": "CC0-1.0",
        "captured_at": "2026-01-04T11:00:00+09:00",
        "image_width_px": 1280,
        "image_height_px": 720,
        "principal_point_px": [640.0, 360.0],
        "distortion_coefficients": [-0.12, 0.03, 1e-4, -2e-4, 0.001],
        "reprojection_error_px": "0.28",
        "notes": "synthetic fixture",
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload) -> "object":
    path = tmp_path / "lens.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_calibration_loads_with_its_provenance(tmp_path) -> None:
    lens, record = load_lens_calibration(_write(tmp_path, _calibration()))
    assert (lens.k1, lens.k2, lens.p1, lens.p2, lens.k3) == (
        -0.12, 0.03, 1e-4, -2e-4, 0.001
    )
    assert record.source_id == "checkerboard-2026-01-04"
    assert record.checksum.startswith("sha256:")
    assert "OpenCV" in record.units


def test_a_calibration_without_provenance_is_refused(tmp_path) -> None:
    payload = _calibration()
    del payload["license"]
    del payload["captured_at"]
    with pytest.raises(ValueError, match="missing required provenance"):
        load_lens_calibration(_write(tmp_path, payload))


def test_a_naive_capture_time_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="no UTC offset"):
        load_lens_calibration(
            _write(tmp_path, _calibration(captured_at="2026-01-04T11:00:00"))
        )


def test_an_off_centre_principal_point_is_refused(tmp_path) -> None:
    # Only the distortion polynomial is applied; a decentred calibration also
    # displaces the projection, and applying half of it would be worse than
    # refusing it.
    with pytest.raises(ValueError, match="off centre"):
        load_lens_calibration(
            _write(tmp_path, _calibration(principal_point_px=[652.0, 360.0]))
        )


def test_a_half_pixel_offset_is_tolerated(tmp_path) -> None:
    # That is the quantisation of the calibration's own image grid.
    lens, _ = load_lens_calibration(
        _write(tmp_path, _calibration(principal_point_px=[640.4, 359.7]))
    )
    assert lens.k1 == -0.12


def test_too_few_coefficients_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="OpenCV order"):
        load_lens_calibration(
            _write(tmp_path, _calibration(distortion_coefficients=[-0.12, 0.03]))
        )


def test_a_four_coefficient_calibration_leaves_k3_zero(tmp_path) -> None:
    lens, _ = load_lens_calibration(
        _write(
            tmp_path,
            _calibration(distortion_coefficients=[-0.12, 0.03, 1e-4, -2e-4]),
        )
    )
    assert lens.k3 == 0.0
