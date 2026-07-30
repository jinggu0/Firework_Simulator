import numpy as np
import pytest

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


def test_geodetic_round_trip_is_within_one_millimetre() -> None:
    # Validation metric V-03. float64 ECEF retains roughly a micrometre over a
    # 5 km baseline, so a 1 mm bound is three orders above numerical noise:
    # exceeding it indicates a formula error rather than precision loss.
    plane = LocalTangentPlane(37.529, 126.935, 5.0)
    for local in (
        (0.0, 0.0, 0.0),
        (2_500.0, 400.0, -2_000.0),
        (-2_500.0, -10.0, 2_000.0),
        (137.4, 322.0, -871.2),
    ):
        target = np.array(local, dtype=np.float64)
        latitude, longitude, altitude = plane.to_geodetic(target)
        recovered = plane.to_local(latitude, longitude, altitude)
        assert float(np.linalg.norm(recovered - target)) < 1e-3


def test_geodetic_inverse_recovers_the_source_coordinates() -> None:
    plane = LocalTangentPlane(37.529, 126.935)
    latitude, longitude, altitude = plane.to_geodetic(
        plane.to_local(37.5401, 126.9187, 63.5)
    )
    assert latitude == pytest.approx(37.5401, abs=1e-9)
    assert longitude == pytest.approx(126.9187, abs=1e-9)
    assert altitude == pytest.approx(63.5, abs=1e-3)


def test_geodetic_inverse_rejects_wrongly_shaped_input() -> None:
    plane = LocalTangentPlane(37.529, 126.935)
    with pytest.raises(ValueError, match="3-component"):
        plane.to_geodetic(np.zeros(2))


def test_vectorised_conversion_matches_the_scalar_path() -> None:
    plane = LocalTangentPlane(37.529, 126.935, 5.0)
    latitudes = np.array([37.515, 37.529, 37.545])
    longitudes = np.array([126.910, 126.935, 126.960])
    altitudes = np.array([0.0, 12.0, 252.0])
    batched = plane.to_local_array(latitudes, longitudes, altitudes)
    assert batched.shape == (3, 3)
    for index in range(3):
        np.testing.assert_allclose(
            batched[index],
            plane.to_local(
                latitudes[index], longitudes[index], altitudes[index]
            ),
            atol=1e-9,
        )

