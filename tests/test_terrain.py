import numpy as np

import numpy as np

from simulator.terrain import (
    TerrainSurface,
    _mercator_pixel,
    constrained_heightmap,
    sample_heightmap_array,
)


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


def test_vector_height_sampling_matches_the_scalar_surface() -> None:
    surface = _surface()
    positions = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
    sampled = sample_heightmap_array(surface.height_m, surface.bounds, positions)
    assert np.allclose(sampled, [surface.height_at(*point) for point in positions])


def test_official_constraints_create_a_bounded_piecewise_linear_surface() -> None:
    positions = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    # Plane y = x + 2z, sampled only at its four official constraints.
    heights = positions[:, 0] + 2.0 * positions[:, 1]
    refined, support = constrained_heightmap(
        np.full((2, 2), 99.0, dtype=np.float32),
        np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
        positions,
        heights,
        np.zeros((2, 2), dtype=np.uint8),
        (5, 5),
    )
    expected_x, expected_z = np.meshgrid(
        np.linspace(0.0, 10.0, 5), np.linspace(0.0, 10.0, 5)
    )
    assert support == 1.0
    assert np.allclose(refined, expected_x + 2.0 * expected_z)
    assert refined.min() >= heights.min()
    assert refined.max() <= heights.max()


def test_refined_terrain_keeps_the_river_on_the_reference_plane() -> None:
    refined, _ = constrained_heightmap(
        np.ones((2, 2), dtype=np.float32),
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.ones(4),
        np.array([[255, 0], [0, 0]], dtype=np.uint8),
        (2, 2),
    )
    assert refined[0, 0] == 0.0
    assert np.all(refined[np.array([[False, True], [True, True]])] == 1.0)


def test_mercator_pixel_matches_known_yeouido_tile() -> None:
    x, y = _mercator_pixel(np.array([37.529]), np.array([126.935]), 12)
    assert int(x[0] // 256) == 3492
    assert int(y[0] // 256) == 1586


def test_mercator_y_decreases_toward_north() -> None:
    _, y = _mercator_pixel(
        np.array([37.52, 37.54]), np.array([126.935, 126.935]), 12
    )
    assert y[1] < y[0]
