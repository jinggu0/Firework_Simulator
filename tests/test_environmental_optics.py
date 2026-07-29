from __future__ import annotations

import numpy as np

from simulator.environmental_optics import (
    beer_lambert_transmittance,
    grass_tip_displacement_m,
    periodic_cloud_noise,
    procedural_star_catalogue,
    relative_air_mass,
    star_contrast_visibility,
)


def test_beer_lambert_spectral_transmission_is_monotonic() -> None:
    short = beer_lambert_transmittance([0.18, 0.08, 0.035], 1.0)
    long = beer_lambert_transmittance([0.18, 0.08, 0.035], 4.0)
    assert np.all(long < short)
    assert long[0] < long[1] < long[2]


def test_star_visibility_falls_with_background_cloud_and_air_mass() -> None:
    clear = star_contrast_visibility(0.0004, 0.002, 0.0, 1.0)
    bright = star_contrast_visibility(0.0004, 0.02, 0.0, 1.0)
    cloud = star_contrast_visibility(0.0004, 0.002, 3.0, 1.0)
    horizon = star_contrast_visibility(0.0004, 0.002, 0.0, 8.0)
    assert clear > bright
    assert clear > cloud
    assert clear > horizon


def test_relative_air_mass_increases_toward_horizon() -> None:
    assert relative_air_mass(np.deg2rad(8.0)) > relative_air_mass(
        np.deg2rad(70.0)
    )


def test_grass_base_is_fixed_and_tip_response_is_bounded() -> None:
    assert grass_tip_displacement_m(8.0, 0.0, 1.0) == 0.0
    moderate = grass_tip_displacement_m(2.0, 1.0, 1.0)
    strong = grass_tip_displacement_m(8.0, 1.0, 1.0)
    assert 0.0 < moderate < strong <= 0.22


def test_environment_textures_are_deterministic_and_bounded() -> None:
    first = periodic_cloud_noise(64, 32, seed=7)
    second = periodic_cloud_noise(64, 32, seed=7)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    catalogue = procedural_star_catalogue(128, 64, 80, seed=7)
    assert catalogue.shape == (64, 128, 3)
    assert np.isfinite(catalogue).all()
    assert np.count_nonzero(catalogue) > 80
