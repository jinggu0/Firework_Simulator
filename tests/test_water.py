from pathlib import Path

import numpy as np

from simulator.environment import EnvironmentTimeline
from simulator.scene import load_scene
from simulator.water import (
    WaterConfig,
    build_directional_spectrum,
    build_water_mesh,
    estimate_fetch_length_m,
    relax_wave_spectrum,
)


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


def test_wave_relaxation_preserves_phase_and_moves_toward_new_wind() -> None:
    calm = build_directional_spectrum(WaterConfig(wind_speed_mps=1.5))
    wind = build_directional_spectrum(WaterConfig(wind_speed_mps=5.0))
    relaxed = relax_wave_spectrum(calm, wind, 30.0, 180.0)
    np.testing.assert_array_equal(relaxed.phases, calm.phases)
    assert (
        calm.significant_wave_height_m
        < relaxed.significant_wave_height_m
        < wind.significant_wave_height_m
    )
    np.testing.assert_allclose(
        np.linalg.norm(relaxed.components[:, :2], axis=1),
        1.0,
        atol=1e-6,
    )


def test_water_mesh_index_bounds() -> None:
    config = WaterConfig(grid_size=(17, 9))
    vertices, indices = build_water_mesh(config)
    assert len(vertices) == 17 * 9
    assert int(indices.max()) < len(vertices)
    assert len(indices) == (17 - 1) * (9 - 1) * 6


def test_north_wind_waves_travel_south() -> None:
    spectrum = build_directional_spectrum(
        WaterConfig(wind_speed_mps=4.0, wind_direction_deg=360.0)
    )
    weights = spectrum.components[:, 3]
    mean_south = np.average(spectrum.components[:, 1], weights=weights)
    assert mean_south > 0.8


def test_fetch_stops_at_upwind_riverbank() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    bounds = np.array([-100.0, -100.0, 100.0, 100.0], dtype=np.float32)
    fetch = estimate_fetch_length_m(
        mask, bounds, np.array([2.0, 0.0]), (0.0, 0.0)
    )
    assert 59.0 <= fetch <= 62.0


def test_far_water_mesh_excludes_near_patch() -> None:
    config = WaterConfig(
        grid_size=(9, 7),
        extent_m=(80.0, 60.0),
        far_grid_size=(11, 9),
        far_extent_m=(200.0, 160.0),
    )
    _, full_indices = build_water_mesh(
        config, config.far_grid_size, config.far_extent_m
    )
    vertices, ring_indices = build_water_mesh(
        config,
        config.far_grid_size,
        config.far_extent_m,
        config.extent_m,
    )
    assert len(ring_indices) < len(full_indices)
    centres = vertices[ring_indices.reshape(-1, 3)].mean(axis=1)
    cell_width = config.far_extent_m[0] / (config.far_grid_size[0] - 1)
    cell_depth = config.far_extent_m[1] / (config.far_grid_size[1] - 1)
    assert np.all(
        (
            np.abs(centres[:, 0])
            >= config.extent_m[0] * 0.5 - cell_width
        )
        | (
            np.abs(centres[:, 1])
            >= config.extent_m[1] * 0.5 - cell_depth
        )
    )


def test_event_water_uses_historical_wind_and_river_fetch() -> None:
    project_root = Path(__file__).resolve().parent.parent
    timeline = EnvironmentTimeline.load(
        project_root / "assets" / "yeouido_2024-10-05_environment.json"
    )
    atmosphere = timeline.sample(timeline.show_start_timestamp)
    scene = load_scene(project_root / "assets" / "yeouido_scene.npz")
    wind_xz = np.asarray(atmosphere.wind_velocity_mps)[[0, 2]]
    wind_speed = float(np.linalg.norm(wind_xz))
    fetch = estimate_fetch_length_m(
        scene.water_mask, scene.water_mask_bounds, wind_xz
    )
    spectrum = build_directional_spectrum(
        WaterConfig(wind_speed_mps=wind_speed, fetch_length_m=fetch)
    )

    assert np.isclose(wind_speed, 0.4714045, atol=1e-6)
    assert 2_090.0 <= fetch <= 2_100.0
    assert 0.012 <= spectrum.significant_wave_height_m <= 0.014
