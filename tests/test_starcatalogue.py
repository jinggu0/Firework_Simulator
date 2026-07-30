from datetime import datetime, timedelta, timezone

import astronomy
import numpy as np
import pytest

from simulator.astronomy import (
    NAKED_EYE_PLANETS,
    TELESCOPIC_PLANETS,
    AstronomyModel,
)
from simulator.color import blackbody_rgb, color_temperature_from_bv
from simulator.provenance import ConfidenceGrade
from simulator.starcatalogue import (
    DEFAULT_CATALOGUE_PATH,
    J2000_EPOCH,
    StarCatalogue,
    annual_aberration_velocity,
    eus_to_equatorial_matrix,
)

EPOCH = datetime.fromisoformat("2024-10-05T19:30:00+09:00")
OBSERVER = (37.529, 126.935, 5.0)

catalogue_required = pytest.mark.skipif(
    not DEFAULT_CATALOGUE_PATH.exists(),
    reason="run 'python -m tools.import_star_catalogue' to enable",
)


@pytest.fixture(scope="module")
def imported() -> StarCatalogue:
    loaded = StarCatalogue.load_if_present()
    if loaded is None:
        pytest.skip("no star catalogue imported")
    return loaded


def synthetic() -> StarCatalogue:
    """A tiny catalogue with no proper motion, for frame and shape checks."""

    return StarCatalogue(
        right_ascension_deg=np.array([0.0, 90.0, 180.0, 270.0]),
        declination_deg=np.array([0.0, 45.0, -45.0, 89.0]),
        proper_motion_ra_cosdec_arcsec_yr=np.zeros(4, dtype=np.float32),
        proper_motion_dec_arcsec_yr=np.zeros(4, dtype=np.float32),
        visual_magnitude=np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
        color_index_bv=np.array([0.0, 0.65, 1.4, -0.2], dtype=np.float32),
        source_id="synthetic-fixture",
        confidence_grade=ConfidenceGrade.ARTISTIC,
    )


# --- colour ----------------------------------------------------------------


def test_bv_to_temperature_reproduces_known_stars() -> None:
    # The Ballesteros (2012) relation is anchored by these two cases: the Sun
    # at B-V 0.65 and an A0V star at B-V 0.
    assert float(color_temperature_from_bv(0.65)) == pytest.approx(5778, abs=15)
    assert 9_000 < float(color_temperature_from_bv(0.0)) < 11_000
    # Redder means cooler, monotonically.
    values = [float(color_temperature_from_bv(bv)) for bv in (-0.3, 0.0, 0.65, 1.5)]
    assert all(a > b for a, b in zip(values, values[1:]))


def test_catalogue_colours_follow_the_colour_index() -> None:
    colours = synthetic().linear_colors()
    assert colours.shape == (4, 3)
    hot = colours[3]  # B-V = -0.2
    cool = colours[2]  # B-V = 1.4
    assert hot[2] / max(hot[0], 1e-6) > cool[2] / max(cool[0], 1e-6)
    np.testing.assert_allclose(colours[1], blackbody_rgb(5778.0), atol=0.02)


# --- structure --------------------------------------------------------------


def test_mismatched_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        StarCatalogue(
            right_ascension_deg=np.zeros(3),
            declination_deg=np.zeros(2),
            proper_motion_ra_cosdec_arcsec_yr=np.zeros(3),
            proper_motion_dec_arcsec_yr=np.zeros(3),
            visual_magnitude=np.zeros(3),
            color_index_bv=np.zeros(3),
        )


def test_missing_catalogue_reports_absence_rather_than_raising(tmp_path) -> None:
    assert StarCatalogue.load_if_present(tmp_path / "absent.npz") is None
    with pytest.raises(FileNotFoundError, match="star catalogue not found"):
        StarCatalogue.load(tmp_path / "absent.npz")


def test_catalogue_round_trips_through_disk(tmp_path) -> None:
    original = synthetic()
    restored = StarCatalogue.load(original.save(tmp_path / "cat.npz"))
    np.testing.assert_array_equal(
        restored.right_ascension_deg, original.right_ascension_deg
    )
    assert restored.source_id == "synthetic-fixture"
    assert restored.confidence_grade is ConfidenceGrade.ARTISTIC
    assert not restored.is_measured


def test_relative_flux_follows_the_magnitude_scale() -> None:
    flux = synthetic().relative_flux()
    # Five magnitudes is a factor of exactly 100.
    assert flux[0] / flux[2] == pytest.approx(100.0 ** (4.0 / 5.0), rel=1e-9)
    assert flux[0] > flux[1] > flux[2] > flux[3]


# --- astrometry -------------------------------------------------------------


def test_proper_motion_moves_a_star_by_its_declared_rate() -> None:
    catalogue = StarCatalogue(
        right_ascension_deg=np.array([100.0]),
        declination_deg=np.array([0.0]),
        # 1 arcsec/yr in declination; at the equator RA is not compressed.
        proper_motion_ra_cosdec_arcsec_yr=np.array([1.0]),
        proper_motion_dec_arcsec_yr=np.array([1.0]),
        visual_magnitude=np.array([3.0]),
        color_index_bv=np.array([0.5]),
    )
    later = J2000_EPOCH + timedelta(days=365.25 * 100.0)
    right_ascension, declination = catalogue.positions_at(later)
    assert declination[0] == pytest.approx(100.0 / 3600.0, rel=1e-6)
    assert right_ascension[0] == pytest.approx(100.0 + 100.0 / 3600.0, rel=1e-9)


def test_positions_require_an_aware_epoch() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        synthetic().positions_at(datetime(2024, 10, 5, 19, 30))


def test_unit_vectors_are_normalised() -> None:
    vectors = synthetic().unit_vectors_at(EPOCH)
    assert vectors.shape == (4, 3)
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-12
    )


def test_aberration_velocity_has_the_classical_magnitude() -> None:
    velocity = annual_aberration_velocity(
        astronomy.Time("2024-10-05T10:30:00Z")
    )
    arcseconds = float(np.linalg.norm(velocity)) * 206_264.806
    # The constant of aberration is 20.49 arcseconds.
    assert 20.0 < arcseconds < 21.0


def test_eus_to_equatorial_matrix_is_a_rotation() -> None:
    matrix = eus_to_equatorial_matrix(
        astronomy.Time("2024-10-05T10:30:00Z"),
        astronomy.Observer(*OBSERVER),
    )
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(matrix)) == pytest.approx(1.0, abs=1e-12)


def test_zenith_ray_maps_to_the_observer_meridian() -> None:
    # Straight up in East-Up-South must land at a declination equal to the
    # observer's latitude, since the zenith lies on the observer's meridian.
    matrix = eus_to_equatorial_matrix(
        astronomy.Time("2024-10-05T10:30:00Z"),
        astronomy.Observer(*OBSERVER),
    )
    zenith = matrix @ np.array([0.0, 1.0, 0.0])
    declination = np.degrees(np.arcsin(np.clip(zenith[2], -1.0, 1.0)))
    assert declination == pytest.approx(OBSERVER[0], abs=0.2)


@catalogue_required
def test_imported_catalogue_is_measured_astrometry(imported) -> None:
    assert len(imported) > 5_000
    assert imported.is_measured
    assert imported.confidence_grade is ConfidenceGrade.MEASURED
    assert imported.source_url
    assert imported.reference_epoch == "J2000.0"
    assert imported.retrieved_at


@catalogue_required
def test_catalogue_agrees_with_the_ephemeris_library(imported) -> None:
    # Validation metric V-20. Both paths apply the same precession, nutation,
    # and diurnal rotation, so they must agree to numerical precision once
    # annual aberration is included.
    time = astronomy.Time("2024-10-05T10:30:00Z")
    observer = astronomy.Observer(*OBSERVER)
    right_ascension, declination = imported.positions_at(EPOCH)
    azimuth, altitude = imported.apparent_horizontal(
        EPOCH, *OBSERVER, refraction=astronomy.Refraction.Airless
    )
    generator = np.random.default_rng(4)
    for slot, index in enumerate(
        generator.choice(len(imported), 8, replace=False)
    ):
        body = getattr(astronomy.Body, f"Star{slot + 1}")
        astronomy.DefineStar(
            body,
            float(right_ascension[index]) / 15.0,
            float(declination[index]),
            100.0,
        )
        equatorial = astronomy.Equator(
            body, time, observer, ofdate=True, aberration=True
        )
        horizon = astronomy.Horizon(
            time,
            observer,
            equatorial.ra,
            equatorial.dec,
            astronomy.Refraction.Airless,
        )
        assert abs(horizon.altitude - altitude[index]) * 3_600.0 < 0.5


@catalogue_required
def test_refraction_lifts_low_stars_and_leaves_the_zenith_alone(imported) -> None:
    airless_az, airless_alt = imported.apparent_horizontal(
        EPOCH, *OBSERVER, refraction=astronomy.Refraction.Airless
    )
    refracted_az, refracted_alt = imported.apparent_horizontal(
        EPOCH, *OBSERVER, refraction=astronomy.Refraction.Normal
    )
    lift = refracted_alt - airless_alt
    near_zenith = airless_alt > 85.0
    near_horizon = (airless_alt > 0.0) & (airless_alt < 3.0)
    assert np.all(lift >= -1e-9)
    if near_zenith.any():
        assert np.all(lift[near_zenith] < 0.01)
    if near_horizon.any():
        # Refraction near the horizon is of order half a degree.
        assert lift[near_horizon].max() > 0.1
    np.testing.assert_allclose(refracted_az, airless_az, atol=1e-12)


@catalogue_required
def test_equatorial_flux_map_places_stars_in_the_celestial_frame(
    imported,
) -> None:
    field = imported.equatorial_flux_map(512, 256)
    assert field.shape == (256, 512, 3)
    assert np.isfinite(field).all()
    assert (field >= 0.0).all()
    # A magnitude-limited catalogue leaves most of the sky empty.
    occupied = np.count_nonzero(field.max(axis=2))
    assert 0 < occupied < field.shape[0] * field.shape[1] // 2


@catalogue_required
def test_the_sky_rotates_between_the_start_and_end_of_the_show(
    imported,
) -> None:
    start = datetime.fromisoformat("2024-10-05T19:20:00+09:00")
    end = datetime.fromisoformat("2024-10-05T20:30:00+09:00")
    _, altitude_start = imported.apparent_horizontal(start, *OBSERVER)
    _, altitude_end = imported.apparent_horizontal(end, *OBSERVER)
    # 70 minutes is 17.5 degrees of diurnal rotation, so altitudes must move.
    assert np.abs(altitude_end - altitude_start).max() > 5.0


# --- planets ----------------------------------------------------------------


def test_all_planets_are_sampled_with_finite_geometry() -> None:
    state = AstronomyModel(*OBSERVER).sample(EPOCH.timestamp())
    names = tuple(planet.name for planet in state.planets)
    assert names == NAKED_EYE_PLANETS + TELESCOPIC_PLANETS
    for planet in state.planets:
        assert -90.0 <= planet.altitude_deg <= 90.0
        assert 0.0 <= planet.azimuth_deg < 360.0
        assert np.isfinite(planet.apparent_magnitude)
        assert 0.0 <= planet.illuminated_fraction <= 1.0
        np.testing.assert_allclose(
            np.linalg.norm(planet.direction_eus), 1.0, atol=1e-5
        )


def test_planets_can_be_skipped() -> None:
    state = AstronomyModel(*OBSERVER).sample(
        EPOCH.timestamp(), include_planets=False
    )
    assert state.planets == ()


def test_planet_lookup_and_visibility_gate() -> None:
    state = AstronomyModel(*OBSERVER).sample(EPOCH.timestamp())
    saturn = state.planet("Saturn")
    assert saturn.above_horizon
    with pytest.raises(KeyError, match="not sampled"):
        state.planet("Ceres")
    visible = state.visible_planets()
    # Neptune is above the horizon at this instant but far below naked-eye.
    assert state.planet("Neptune").above_horizon
    assert "Neptune" not in [planet.name for planet in visible]
    assert all(planet.above_horizon for planet in visible)


def test_planet_direction_matches_its_reported_azimuth_and_altitude() -> None:
    from simulator.astronomy import horizontal_direction_eus

    state = AstronomyModel(*OBSERVER).sample(EPOCH.timestamp())
    for planet in state.planets:
        np.testing.assert_allclose(
            planet.direction_eus,
            horizontal_direction_eus(planet.azimuth_deg, planet.altitude_deg),
            atol=1e-6,
        )


def test_celestial_state_carries_the_equatorial_frame() -> None:
    state = AstronomyModel(*OBSERVER).sample(EPOCH.timestamp())
    matrix = state.equatorial_from_eus
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
