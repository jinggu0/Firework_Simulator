import math

import numpy as np
import pytest

from simulator.color import cie_xyz_at_wavelength, wavelength_rgb
from simulator.config import AtmosphereConfig, ShellConfig
from simulator.physics import FireworkWorld, tube_direction_eus
from simulator.provenance import ConfidenceGrade
from simulator.shells import (
    SHELL_LIBRARY,
    BreakPattern,
    FlickerMode,
    SecondaryBreak,
    ShellLibrary,
    ShellProfile,
    emission_directions,
)

REQUIRED_PATTERNS = {
    BreakPattern.PEONY,
    BreakPattern.CHRYSANTHEMUM,
    BreakPattern.WILLOW,
    BreakPattern.PALM,
    BreakPattern.RING,
    BreakPattern.CROSSETTE,
    BreakPattern.HORSETAIL,
    BreakPattern.COMET,
    BreakPattern.MINE,
    BreakPattern.FAN,
    BreakPattern.WATERFALL,
}


def generator(seed: int = 5) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- colour ----------------------------------------------------------------


def test_cie_fit_peaks_where_the_matching_functions_do() -> None:
    # y-bar peaks near 555 nm, the photopic luminous efficiency maximum.
    samples = np.arange(400.0, 700.0, 1.0)
    luminance = [cie_xyz_at_wavelength(nm)[1] for nm in samples]
    assert abs(samples[int(np.argmax(luminance))] - 555.0) < 6.0


def test_emission_wavelengths_produce_the_expected_hue() -> None:
    red = wavelength_rgb(640.0)
    green = wavelength_rgb(515.0)
    blue = wavelength_rgb(452.0)
    assert red[0] > red[1] and red[0] > red[2]
    assert green[1] > green[0] and green[1] > green[2]
    assert blue[2] > blue[0] and blue[2] > blue[1]
    for colour in (red, green, blue):
        # Max-channel normalisation matches the black-body convention the
        # renderer already calibrates against.
        assert colour.max() == pytest.approx(1.0, abs=1e-6)
        assert (colour >= 0.0).all()


# --- library integrity -----------------------------------------------------


def test_library_covers_every_required_break_pattern() -> None:
    assert REQUIRED_PATTERNS <= SHELL_LIBRARY.patterns()


def test_library_covers_the_required_optical_modifiers() -> None:
    flickers = {profile.flicker for profile in SHELL_LIBRARY}
    assert FlickerMode.STROBE in flickers
    assert FlickerMode.CRACKLE in flickers
    assert any(profile.changes_color for profile in SHELL_LIBRARY)
    assert any(profile.secondary is not None for profile in SHELL_LIBRARY)


def test_every_shipped_profile_is_graded_artistic() -> None:
    # No measured shell record for the 2024 performance has been obtained, so
    # nothing in the library may claim to be evidence.
    for profile in SHELL_LIBRARY:
        assert profile.confidence_grade is ConfidenceGrade.ARTISTIC
        assert not profile.confidence_grade.is_evidence
        assert profile.source_note


def test_profile_ids_are_unique_and_sorted_lookup_works() -> None:
    ids = SHELL_LIBRARY.ids()
    assert len(ids) == len(set(ids)) == len(SHELL_LIBRARY)
    assert SHELL_LIBRARY.get(ids[0]).profile_id == ids[0]
    with pytest.raises(KeyError, match="unknown shell profile"):
        SHELL_LIBRARY.get("not_a_shell")


def test_duplicate_profile_ids_are_rejected() -> None:
    profile = SHELL_LIBRARY.get("peony_100mm_gold")
    with pytest.raises(ValueError, match="duplicate profile id"):
        ShellLibrary((profile, profile))


def test_invalid_profiles_are_rejected() -> None:
    with pytest.raises(ValueError, match="burst_star_count"):
        ShellProfile(profile_id="x", display_name="x", burst_star_count=0)
    with pytest.raises(ValueError, match="requires a positive flicker_hz"):
        ShellProfile(
            profile_id="x",
            display_name="x",
            flicker=FlickerMode.STROBE,
            flicker_hz=0.0,
        )
    with pytest.raises(ValueError, match="star_count"):
        SecondaryBreak(pattern=BreakPattern.RING, star_count=0)
    with pytest.raises(ValueError, match="parent_fraction"):
        SecondaryBreak(
            pattern=BreakPattern.RING, star_count=4, parent_fraction=1.5
        )


def test_legacy_shell_config_round_trips_unchanged() -> None:
    config = ShellConfig()
    profile = ShellProfile.from_shell_config(config)
    assert profile.pattern is BreakPattern.PEONY
    for name in (
        "dry_mass_kg", "diameter_m", "launch_speed_mps", "fuse_delay_s",
        "drag_coefficient", "burst_star_count", "star_speed_mean_mps",
        "star_lifetime_mean_s", "star_drag_time_s", "star_composition_mass_kg",
        "burst_charge_mass_kg", "smoke_yield_fraction",
    ):
        assert getattr(profile, name) == getattr(config, name)
    assert ShellProfile.coerce(profile) is profile


# --- break geometry --------------------------------------------------------


def test_peony_is_isotropic_and_unit_length() -> None:
    directions, scale = emission_directions(
        BreakPattern.PEONY, 4_000, generator()
    )
    lengths = np.linalg.norm(directions, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)
    np.testing.assert_allclose(scale, 1.0)
    # A uniform sphere has a mean direction near zero on every axis.
    assert np.abs(directions.mean(axis=0)).max() < 0.05


def test_ring_is_planar() -> None:
    directions, _ = emission_directions(
        BreakPattern.RING, 2_000, generator(), spread_deg=4.0
    )
    # The ring's plane normal is up by default, so the vertical component must
    # stay within the declared angular thickness.
    assert np.abs(directions[:, 1]).max() < math.sin(math.radians(6.0))
    # Azimuthal coverage must be complete rather than clustered.
    angle = np.arctan2(directions[:, 2], directions[:, 0])
    histogram, _ = np.histogram(angle, bins=12, range=(-math.pi, math.pi))
    assert histogram.min() > 0


def test_mine_and_waterfall_point_in_opposite_vertical_senses() -> None:
    mine, mine_scale = emission_directions(
        BreakPattern.MINE, 1_500, generator(), spread_deg=26.0
    )
    waterfall, _ = emission_directions(
        BreakPattern.WATERFALL, 1_500, generator()
    )
    assert mine[:, 1].mean() > 0.8
    assert waterfall[:, 1].mean() < -0.5
    # A mine's spray is graded, not a uniform shell.
    assert mine_scale.min() < mine_scale.max()


def test_horsetail_hangs_below_the_break() -> None:
    directions, scale = emission_directions(
        BreakPattern.HORSETAIL, 1_500, generator()
    )
    assert directions[:, 1].mean() < -0.4
    assert scale.max() <= 1.0


def test_willow_opens_upward_before_it_droops() -> None:
    directions, _ = emission_directions(
        BreakPattern.WILLOW, 2_000, generator()
    )
    assert 0.1 < directions[:, 1].mean() < 0.6


def test_fan_stays_within_its_sector() -> None:
    directions, _ = emission_directions(
        BreakPattern.FAN, 1_500, generator(), spread_deg=30.0
    )
    # The fan's plane is spanned by the two vectors orthogonal to the axis, so
    # out-of-plane deviation must remain small.
    assert np.abs(directions[:, 1]).max() < math.sin(math.radians(10.0))


def test_comet_is_a_narrow_beam() -> None:
    directions, _ = emission_directions(
        BreakPattern.COMET, 800, generator(), spread_deg=6.0
    )
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    cosines = directions @ axis
    assert cosines.min() > math.cos(math.radians(2.0))


def test_palm_forms_discrete_spokes() -> None:
    directions, scale = emission_directions(
        BreakPattern.PALM, 3_000, generator(), spread_deg=32.0
    )
    assert directions[:, 1].min() > 0.0  # every spoke rises
    angle = np.arctan2(directions[:, 2], directions[:, 0])
    histogram, _ = np.histogram(angle, bins=48, range=(-math.pi, math.pi))
    # Spokes leave empty azimuthal gaps; an isotropic shell would not.
    assert (histogram == 0).sum() > 4
    assert scale.min() < scale.max()


def test_all_patterns_produce_finite_unit_directions() -> None:
    for pattern in BreakPattern:
        directions, scale = emission_directions(pattern, 400, generator())
        assert directions.shape == (400, 3)
        assert np.isfinite(directions).all()
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1), 1.0, atol=1e-4
        )
        assert (scale > 0.0).all()


def test_zero_count_returns_empty_arrays() -> None:
    directions, scale = emission_directions(BreakPattern.PEONY, 0, generator())
    assert directions.shape == (0, 3)
    assert scale.shape == (0,)


def test_emission_is_deterministic_for_a_fixed_seed() -> None:
    for pattern in BreakPattern:
        first, first_scale = emission_directions(pattern, 256, generator(11))
        second, second_scale = emission_directions(pattern, 256, generator(11))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_scale, second_scale)


# --- launch geometry -------------------------------------------------------


def test_vertical_tube_reproduces_the_previous_launch_direction() -> None:
    direction = tube_direction_eus(0.0, 90.0)
    np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1e-6)


def test_tube_heading_follows_the_north_clockwise_convention() -> None:
    # Azimuth 90 is east, 180 is south, in the East-Up-South frame.
    np.testing.assert_allclose(
        tube_direction_eus(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        tube_direction_eus(180.0, 0.0), [0.0, 0.0, 1.0], atol=1e-6
    )


def test_tilted_launch_moves_the_shell_off_the_vertical() -> None:
    world = FireworkWorld(
        AtmosphereConfig(wind_velocity_mps=(0.0, 0.0, 0.0),
                         wind_velocity_100m_mps=(0.0, 0.0, 0.0)),
        ShellConfig(burst_star_count=16),
        1_024,
        3,
    )
    tilted = world.launch(azimuth_deg=90.0, elevation_deg=60.0)
    for _ in range(120):
        world.update(1.0 / 120.0)
    # Fired east at 60 degrees: substantial eastward travel, still climbing.
    assert tilted.position_m[0] > 30.0
    assert tilted.position_m[1] > 30.0
    assert abs(tilted.position_m[2]) < 1.0


# --- star field behaviour --------------------------------------------------


def make_world(profile: ShellProfile, seed: int = 7) -> FireworkWorld:
    return FireworkWorld(
        AtmosphereConfig(
            wind_velocity_mps=(0.0, 0.0, 0.0),
            wind_velocity_100m_mps=(0.0, 0.0, 0.0),
        ),
        profile,
        60_000,
        seed,
    )


def test_colour_changing_star_crosses_over_during_its_burn() -> None:
    profile = SHELL_LIBRARY.get("colorchange_125mm_green_red")
    world = make_world(profile)
    world.launch()
    steps = int(profile.fuse_delay_s * 120) + 2
    for _ in range(steps):
        world.update(1.0 / 120.0)
    early = world.stars.current_color_linear().mean(axis=0).copy()
    for _ in range(int(profile.star_lifetime_mean_s * 0.9 * 120)):
        world.update(1.0 / 120.0)
    late = world.stars.current_color_linear().mean(axis=0)
    assert early[1] > early[0]  # opens green
    assert late[0] > late[1]  # ends red


def test_steady_stars_return_the_stored_colour_without_copying() -> None:
    world = make_world(SHELL_LIBRARY.get("peony_100mm_gold"))
    world.launch()
    for _ in range(int(3.05 * 120) + 2):
        world.update(1.0 / 120.0)
    assert world.stars.count > 0
    # A steady break must return a view of the stored colours, not a per-frame
    # blend: the renderer reads this every frame for up to 250,000 stars.
    assert np.shares_memory(
        world.stars.current_color_linear(), world.stars.color_linear
    )


def test_strobe_modulation_is_deep_and_crackle_is_shallow() -> None:
    for profile_id, expected_peak in (
        ("strobe_125mm_white", 2.0),
        ("crackling_100mm_gold", 1.2),
    ):
        profile = SHELL_LIBRARY.get(profile_id)
        world = make_world(profile)
        world.launch()
        for _ in range(int(profile.fuse_delay_s * 120) + 2):
            world.update(1.0 / 120.0)
        peaks = []
        for _ in range(60):
            world.update(1.0 / 120.0)
            intensity = world.stars.intensity()
            if intensity.size:
                peaks.append(float(intensity.max() / max(intensity.mean(), 1e-9)))
        assert max(peaks) > expected_peak


def test_flicker_modulation_preserves_mean_output() -> None:
    from simulator.physics import (
        CRACKLE_MODULATION_DEPTH,
        STROBE_DUTY_CYCLE,
    )

    # sin^2 has mean 1/2 over its support, so the strobe normalisation of
    # 2/duty gives unit mean over a whole number of periods.
    cycle = np.linspace(0.0, 1.0, 100_001)
    edge = np.clip(cycle / STROBE_DUTY_CYCLE, 0.0, 1.0)
    strobe = (2.0 / STROBE_DUTY_CYCLE) * np.sin(np.pi * edge) ** 2
    assert np.trapezoid(strobe, cycle) == pytest.approx(1.0, rel=2e-3)
    crackle = 1.0 + CRACKLE_MODULATION_DEPTH * np.sin(2.0 * np.pi * cycle)
    assert np.trapezoid(crackle, cycle) == pytest.approx(1.0, rel=1e-6)


def test_crossette_children_are_released_at_the_end_of_the_parent_burn() -> None:
    profile = SHELL_LIBRARY.get("crossette_125mm_blue")
    world = make_world(profile)
    world.launch()
    for _ in range(int(profile.fuse_delay_s * 120) + 2):
        world.update(1.0 / 120.0)
    primaries = world.stars.count
    assert primaries == profile.burst_star_count
    for _ in range(int((profile.star_lifetime_mean_s + 0.3) * 120)):
        world.update(1.0 / 120.0)
    # Every primary splits into star_count children.
    assert world.stars.count > primaries
    assert world.stars.count <= primaries * profile.secondary.star_count


def test_partial_multibreak_only_splits_the_declared_fraction() -> None:
    profile = SHELL_LIBRARY.get("multibreak_150mm_red_green")
    world = make_world(profile)
    world.launch()
    for _ in range(int(profile.fuse_delay_s * 120) + 2):
        world.update(1.0 / 120.0)
    carriers = int((world.stars.secondary_index[: world.stars.count] >= 0).sum())
    expected = profile.secondary.parent_fraction * profile.burst_star_count
    # Bernoulli sampling: 3 standard deviations of the binomial count.
    spread = 3.0 * math.sqrt(
        profile.burst_star_count
        * profile.secondary.parent_fraction
        * (1.0 - profile.secondary.parent_fraction)
    )
    assert abs(carriers - expected) < spread


def test_secondary_break_draws_from_its_own_declared_mass() -> None:
    profile = SHELL_LIBRARY.get("crossette_125mm_blue")
    world = make_world(profile)
    world.launch()
    for _ in range(int((profile.fuse_delay_s + 2.4) * 120)):
        world.update(1.0 / 120.0)
    children = world.stars.count
    if children:
        total_child_fuel = float(
            world.stars.fuel_mass_kg[:children].sum(dtype=np.float64)
        )
        # Children never exceed the secondary's declared composition mass, so a
        # break cannot inflate the parent shell's energy budget.
        assert total_child_fuel <= profile.secondary.composition_mass_kg * 1.001


def test_every_profile_launches_and_conserves_star_fuel() -> None:
    for profile in SHELL_LIBRARY:
        world = make_world(profile, seed=13)
        world.launch()
        emitted_kg = 0.0
        horizon_s = profile.fuse_delay_s + profile.star_lifetime_mean_s * 3.0 + 2.0
        for step in range(int(horizon_s * 120)):
            world.update(1.0 / 120.0)
            if (step + 1) % 4 == 0:
                for emission in world.consume_combustion_emissions():
                    emitted_kg += float(emission.smoke_mass_kg.sum())
        budget_kg = (
            profile.total_composition_mass_kg()
            * profile.star_smoke_yield_fraction
        )
        assert emitted_kg <= budget_kg * 1.001, profile.profile_id
        assert np.isfinite(world.stars.position_m[: world.stars.count]).all()
