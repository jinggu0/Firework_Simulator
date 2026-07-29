import numpy as np

from simulator.geodesy import LocalTangentPlane


def test_local_origin_maps_to_zero() -> None:
    plane = LocalTangentPlane(37.5, 126.9, 12.0)
    np.testing.assert_allclose(plane.to_local(37.5, 126.9, 12.0), 0.0, atol=1e-8)


def test_coordinate_axes_follow_east_up_south_convention() -> None:
    plane = LocalTangentPlane(37.5, 126.9)
    east = plane.to_local(37.5, 126.901)
    north = plane.to_local(37.501, 126.9)
    high = plane.to_local(37.5, 126.9, 10.0)
    assert east[0] > 80.0
    assert north[2] < -100.0
    assert high[1] > 9.99

