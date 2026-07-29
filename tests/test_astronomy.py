from datetime import datetime

import numpy as np

from simulator.astronomy import (
    AstronomyModel,
    horizontal_direction_eus,
    twilight_illuminance_lux,
)


def test_cardinal_horizontal_directions_use_eus_axes() -> None:
    np.testing.assert_allclose(horizontal_direction_eus(0.0, 0.0), [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(horizontal_direction_eus(90.0, 0.0), [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(horizontal_direction_eus(180.0, 0.0), [0, 0, 1], atol=1e-6)


def test_event_start_celestial_state() -> None:
    timestamp = datetime.fromisoformat("2024-10-05T19:20:00+09:00").timestamp()
    state = AstronomyModel(37.529, 126.935, 5.0).sample(timestamp)
    assert abs(state.sun_altitude_deg - (-14.28)) < 0.1
    assert abs(state.sun_azimuth_deg - 275.13) < 0.1
    assert state.moon_altitude_deg < 0.0
    assert abs(state.moon_phase_fraction - 0.0623) < 0.001
    assert state.moon_illuminance_lux == 0.0
    end = AstronomyModel(37.529, 126.935, 5.0).sample(
        datetime.fromisoformat("2024-10-05T20:30:00+09:00").timestamp()
    )
    assert end.sun_altitude_deg < state.sun_altitude_deg
    assert end.twilight_illuminance_lux < state.twilight_illuminance_lux


def test_twilight_illuminance_decreases_monotonically() -> None:
    values = [twilight_illuminance_lux(alt) for alt in (0, -6, -12, -18, -24)]
    assert all(a >= b for a, b in zip(values, values[1:]))
