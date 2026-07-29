import numpy as np

from simulator.terrain import _mercator_pixel


def test_mercator_pixel_matches_known_yeouido_tile() -> None:
    x, y = _mercator_pixel(np.array([37.529]), np.array([126.935]), 12)
    assert int(x[0] // 256) == 3492
    assert int(y[0] // 256) == 1586


def test_mercator_y_decreases_toward_north() -> None:
    _, y = _mercator_pixel(
        np.array([37.52, 37.54]), np.array([126.935, 126.935]), 12
    )
    assert y[1] < y[0]
