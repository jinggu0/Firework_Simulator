import numpy as np

from simulator.renderer import _look_at, _perspective
from simulator.volume import (
    active_slice_bounds,
    active_volume_bounds,
    box_vertices,
    opaque_ray_limit,
    reconstruct_world_position,
)


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


def test_active_slice_bounds_preserves_full_analytic_depth() -> None:
    field = np.zeros((8, 10), dtype=np.float32)
    field[3:5, 4:7] = 0.1
    bounds = active_slice_bounds(
        field, (-5.0, 0.0), (5.0, 8.0), -3.0, 3.0
    )
    assert bounds is not None
    minimum, maximum = bounds
    np.testing.assert_allclose(minimum, [-2.0, 2.0, -3.0])
    np.testing.assert_allclose(maximum, [3.0, 6.0, 3.0])


def test_depth_sample_reconstructs_original_world_position() -> None:
    eye = np.array([12.0, 18.0, 35.0], dtype=np.float32)
    target = np.array([-4.0, 9.0, -20.0], dtype=np.float32)
    view_projection = _perspective(47.0, 16.0 / 9.0, 0.1, 2500.0) @ (
        _look_at(eye, target)
    )
    world = np.array([1.0, 11.0, -8.0, 1.0], dtype=np.float64)
    clip = view_projection.astype(np.float64) @ world
    ndc = clip[:3] / clip[3]
    reconstructed = reconstruct_world_position(
        ndc[:2] * 0.5 + 0.5,
        ndc[2] * 0.5 + 0.5,
        np.linalg.inv(view_projection.astype(np.float64)),
    )
    np.testing.assert_allclose(reconstructed, world[:3], atol=2e-4)


def test_opaque_ray_limit_stops_before_surface() -> None:
    camera = np.array([0.0, 2.0, 5.0])
    ray = np.array([0.0, 0.0, -2.0])
    surface = np.array([0.0, 2.0, -15.0])
    assert opaque_ray_limit(camera, ray, surface, 0.2) == 19.8
