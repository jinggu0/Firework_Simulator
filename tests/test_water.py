import numpy as np

from simulator.water import WaterConfig, build_directional_spectrum, build_water_mesh


def test_wave_spectrum_is_deterministic_and_finite() -> None:
    config = WaterConfig()
    a = build_directional_spectrum(config)
    b = build_directional_spectrum(config)
    np.testing.assert_array_equal(a.components, b.components)
    assert np.isfinite(a.components).all()
    assert 0.0 < a.significant_wave_height_m < 2.0


def test_stronger_wind_produces_more_wave_energy() -> None:
    calm = build_directional_spectrum(WaterConfig(wind_speed_mps=1.5))
    wind = build_directional_spectrum(WaterConfig(wind_speed_mps=5.0))
    assert wind.significant_wave_height_m > calm.significant_wave_height_m


def test_water_mesh_index_bounds() -> None:
    config = WaterConfig(grid_size=(17, 9))
    vertices, indices = build_water_mesh(config)
    assert len(vertices) == 17 * 9
    assert int(indices.max()) < len(vertices)
    assert len(indices) == (17 - 1) * (9 - 1) * 6

