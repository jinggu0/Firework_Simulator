from pathlib import Path

import numpy as np

from simulator.environment import (
    EnvironmentTimeline,
    meteorological_wind_to_eus,
    moist_air_density,
)


def test_north_wind_moves_toward_south() -> None:
    velocity = meteorological_wind_to_eus(3.0, 360.0)
    np.testing.assert_allclose(velocity, [0.0, 0.0, 3.0], atol=1e-6)


def test_west_wind_moves_toward_east() -> None:
    velocity = meteorological_wind_to_eus(4.0, 270.0)
    np.testing.assert_allclose(velocity, [4.0, 0.0, 0.0], atol=1e-6)


def test_moist_air_density_is_physical() -> None:
    density = moist_air_density(293.15, 101_325.0, 0.5)
    assert 1.15 < density < 1.25


def test_event_timeline_interpolates_observations() -> None:
    path = Path(__file__).resolve().parent.parent / "assets" / (
        "yeouido_2024-10-05_environment.json"
    )
    timeline = EnvironmentTimeline.load(path)
    midpoint = (timeline.timestamps[1] + timeline.timestamps[2]) * 0.5
    state = timeline.sample(midpoint)
    expected_c = (19.3 + 17.9) * 0.5
    assert abs(state.temperature_k - (expected_c + 273.15)) < 1e-6
    assert 1.15 < state.air_density_kg_m3 < 1.3
    assert np.linalg.norm(state.wind_velocity_100m_mps) > np.linalg.norm(
        state.wind_velocity_mps
    )
