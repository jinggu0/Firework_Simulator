import numpy as np

import numpy as np

from simulator.terrain import TerrainSurface, _mercator_pixel


def _surface() -> TerrainSurface:
    return TerrainSurface(
        np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32),
        np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
    )


def test_height_field_samples_geographic_texel_centres_bilinearly() -> None:
    surface = _surface()
    assert surface.height_at(0.0, 0.0) == 0.0
    assert surface.height_at(10.0, 10.0) == 30.0
    assert surface.height_at(5.0, 5.0) == 15.0
    assert surface.height_at(-100.0, 5.0) == 10.0


def test_height_field_normal_tracks_a_planar_slope() -> None:
    surface = TerrainSurface(
        np.array(
            [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [0.0, 10.0, 20.0]],
            dtype=np.float32,
        ),
        np.array([0.0, 0.0, 20.0, 20.0], dtype=np.float32),
    )
    expected = np.array([-1.0, 1.0, 0.0]) / np.sqrt(2.0)
    assert np.allclose(surface.normal_at(10.0, 10.0), expected)


def test_water_mask_owns_the_collision_surface_at_the_river_datum() -> None:
    surface = TerrainSurface(
        np.full((2, 2), -1.0, dtype=np.float32),
        np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
        np.array([[255, 0], [0, 0]], dtype=np.uint8),
    )
    assert surface.is_water(0.0, 0.0)
    assert surface.collision_height_at(0.0, 0.0) == 0.0
    assert not surface.is_water(10.0, 10.0)
    assert surface.collision_height_at(10.0, 10.0) == -1.0


def test_mercator_pixel_matches_known_yeouido_tile() -> None:
    x, y = _mercator_pixel(np.array([37.529]), np.array([126.935]), 12)
    assert int(x[0] // 256) == 3492
    assert int(y[0] // 256) == 1586


def test_mercator_y_decreases_toward_north() -> None:
    _, y = _mercator_pixel(
        np.array([37.52, 37.54]), np.array([126.935, 126.935]), 12
    )
    assert y[1] < y[0]
