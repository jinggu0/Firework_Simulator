import numpy as np

from simulator.volume import active_volume_bounds, box_vertices


def test_active_volume_bounds_conservatively_wraps_density() -> None:
    volume = np.zeros((6, 8, 10), dtype=np.float32)
    volume[2:4, 3:5, 4:7] = 0.1
    bounds = active_volume_bounds(
        volume,
        np.array([-5.0, 0.0, -3.0]),
        np.array([5.0, 8.0, 3.0]),
    )
    assert bounds is not None
    minimum, maximum = bounds
    np.testing.assert_allclose(minimum, [-2.0, 2.0, -2.0])
    np.testing.assert_allclose(maximum, [3.0, 6.0, 2.0])


def test_box_vertices_span_requested_aabb() -> None:
    vertices = box_vertices((-2.0, 1.0, -3.0), (4.0, 5.0, 6.0))
    np.testing.assert_array_equal(vertices.min(axis=0), [-2.0, 1.0, -3.0])
    np.testing.assert_array_equal(vertices.max(axis=0), [4.0, 5.0, 6.0])
