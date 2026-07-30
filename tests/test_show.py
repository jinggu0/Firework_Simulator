import copy
import json
from datetime import timedelta

import numpy as np
import pytest

from simulator.provenance import ConfidenceGrade
from simulator.scenario import Scenario, load_default_scenario
from simulator.shells import SHELL_LIBRARY, BreakPattern
from simulator.show import ShowScheduler
from simulator.timebase import PlaybackMode, parse_event_time

DEMO_PATH = (
    load_default_scenario().source_path.parent / "scenario_demo_synthetic.json"
)


@pytest.fixture(scope="module")
def demo() -> Scenario:
    return Scenario.load(DEMO_PATH)


@pytest.fixture()
def demo_dict() -> dict:
    return json.loads(DEMO_PATH.read_text(encoding="utf-8"))


# --- the historical scenario has no timeline -------------------------------


def test_historical_scenario_schedules_nothing() -> None:
    # No dated firing record for 2024-10-05 has been obtained, so the
    # reconstruction must fire nothing rather than invent a sequence.
    scheduler = ShowScheduler(load_default_scenario())
    assert len(scheduler) == 0
    assert scheduler.finished
    assert scheduler.next_launch_time() is None
    assert scheduler.due(parse_event_time("2024-10-05T20:00:00+09:00")) == []


# --- the synthetic demo ----------------------------------------------------


def test_demo_scenario_is_unmistakably_synthetic(demo: Scenario) -> None:
    assert demo.scenario_id == "demo-synthetic-not-historical"
    assert demo.provenance.grade_for("scenario") is ConfidenceGrade.ARTISTIC
    assert demo.provenance.grade_for("events") is ConfidenceGrade.ARTISTIC
    assert demo.provenance.grade_for("launch_sites") is ConfidenceGrade.ARTISTIC
    for event in demo.events:
        assert event.confidence_grade is ConfidenceGrade.ARTISTIC
    assert "NOT" in demo.provenance.record_for("scenario").notes


def test_demo_exercises_every_break_pattern(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    patterns = {launch.profile.pattern for launch in scheduler.launches}
    assert patterns == SHELL_LIBRARY.patterns()
    assert BreakPattern.WATERFALL in patterns


def test_demo_sites_fall_inside_the_river_mask(demo: Scenario) -> None:
    from simulator.scene import load_scene

    scene = load_scene(DEMO_PATH.parent / "yeouido_scene.npz")
    mask, bounds = scene.water_mask, scene.water_mask_bounds
    for site in demo.launch_sites:
        east, _, south = demo.launch_site_position_eus_m(site.site_id)
        column = int(
            (east - bounds[0]) / (bounds[2] - bounds[0]) * (mask.shape[1] - 1)
        )
        row = int(
            (south - bounds[1]) / (bounds[3] - bounds[1]) * (mask.shape[0] - 1)
        )
        assert mask[row, column] > 127, site.site_id


# --- scheduling ------------------------------------------------------------


def test_events_fire_in_time_order(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    times = [launch.event.launch_time_utc for launch in scheduler.launches]
    assert times == sorted(times)


def test_nothing_fires_before_its_launch_time(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    first = scheduler.next_launch_time()
    assert scheduler.due(first - timedelta(milliseconds=1)) == []
    released = scheduler.due(first)
    assert len(released) == 1
    assert released[0].event.launch_time_utc == first


def test_stepping_the_clock_releases_every_event_exactly_once(
    demo: Scenario,
) -> None:
    scheduler = ShowScheduler(demo)
    clock = demo.make_clock(120, PlaybackMode.DETERMINISTIC)
    fired: list[str] = []
    for _ in range(120 * 120):  # two minutes of show time
        clock.advance_steps(1)
        fired.extend(
            launch.event.event_id for launch in scheduler.due(clock.absolute_time)
        )
    assert len(fired) == len(demo.events)
    assert len(set(fired)) == len(fired)
    assert fired == sorted(fired)
    assert scheduler.finished
    assert scheduler.remaining == 0


def test_simultaneous_events_both_release_in_file_order(demo: Scenario) -> None:
    # d16 and d17 share a launch time; d20 and d21 also do.
    scheduler = ShowScheduler(demo)
    released = scheduler.due(parse_event_time("2024-10-05T19:20:47+09:00"))
    identifiers = [launch.event_id for launch in released]
    assert "d16" in identifiers and "d17" in identifiers
    assert identifiers.index("d16") < identifiers.index("d17")


def test_seeking_forward_consumes_without_firing(demo: Scenario) -> None:
    # Seeking must not dump every earlier shell into one frame.
    scheduler = ShowScheduler(demo)
    scheduler.seek_to(parse_event_time("2024-10-05T19:20:30+09:00"))
    assert scheduler.fired_count > 0
    assert scheduler.remaining > 0
    remaining_before = scheduler.remaining
    released = scheduler.due(parse_event_time("2024-10-05T19:20:30+09:00"))
    assert released == []
    assert scheduler.remaining == remaining_before


def test_reset_replays_the_show(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    first = scheduler.due(parse_event_time("2024-10-05T20:30:00+09:00"))
    scheduler.reset()
    second = scheduler.due(parse_event_time("2024-10-05T20:30:00+09:00"))
    assert [launch.event_id for launch in first] == [
        launch.event_id for launch in second
    ]
    assert len(first) == len(demo.events)


def test_scheduler_rejects_a_naive_instant(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        scheduler.due(datetime(2024, 10, 5, 19, 20))
    with pytest.raises(ValueError, match="timezone-aware"):
        scheduler.seek_to(datetime(2024, 10, 5, 19, 20))


def test_scheduling_is_deterministic(demo: Scenario) -> None:
    # The scheduler holds no randomness; two runs must agree exactly.
    def run() -> list[tuple[str, float, float, float]]:
        scheduler = ShowScheduler(demo)
        return [
            (
                launch.event_id,
                float(launch.position_eus_m[0]),
                launch.azimuth_deg,
                launch.elevation_deg,
            )
            for launch in scheduler.due(
                parse_event_time("2024-10-05T20:30:00+09:00")
            )
        ]

    assert run() == run()


# --- launch geometry and overrides -----------------------------------------


def test_unrecorded_tube_heading_defaults_to_vertical(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    plain = next(
        launch
        for launch in scheduler.launches
        if launch.event.tube_azimuth_deg is None
        and launch.event.tube_elevation_deg is None
    )
    assert plain.azimuth_deg == 0.0
    assert plain.elevation_deg == 90.0


def test_recorded_tube_heading_is_used(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    fan = next(
        launch for launch in scheduler.launches if launch.event_id == "d18"
    )
    assert fan.azimuth_deg == 270.0
    assert fan.elevation_deg == 72.0


def test_per_event_overrides_replace_profile_values(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    override = next(
        launch for launch in scheduler.launches if launch.event_id == "d22"
    )
    base = SHELL_LIBRARY.get("peony_150mm_red")
    assert override.profile.diameter_m == pytest.approx(0.200)
    assert override.profile.launch_speed_mps == 112.0
    assert override.profile.fuse_delay_s == 5.2
    # Untouched fields keep the library archetype's values.
    assert override.profile.burst_star_count == base.burst_star_count
    assert override.profile.star_composition_mass_kg == (
        base.star_composition_mass_kg
    )
    # The library itself is not mutated by an override.
    assert SHELL_LIBRARY.get("peony_150mm_red").diameter_m == 0.15


def test_launch_positions_differ_between_sites(demo: Scenario) -> None:
    scheduler = ShowScheduler(demo)
    positions = {
        launch.event.launch_site_id: tuple(launch.position_eus_m)
        for launch in scheduler.launches
    }
    assert len(positions) == 3
    east_values = sorted(position[0] for position in positions.values())
    # Sites are 0 m, 350 m, and 700 m east of the scene origin.
    assert east_values[1] - east_values[0] == pytest.approx(350.0, abs=1.0)
    assert east_values[2] - east_values[1] == pytest.approx(350.0, abs=1.0)


def test_summary_reports_the_programme(demo: Scenario) -> None:
    summary = ShowScheduler(demo).summary()
    assert summary["event_count"] == len(demo.events)
    assert summary["duration_s"] > 0.0
    assert sum(summary["by_pattern"].values()) == len(demo.events)
    assert summary["total_stars"] > 0


# --- integration with the physics world -------------------------------------


def test_scheduled_launches_reach_the_physics_world(demo: Scenario) -> None:
    from simulator.config import AtmosphereConfig
    from simulator.physics import FireworkWorld

    world = FireworkWorld(
        AtmosphereConfig(),
        SHELL_LIBRARY.get("peony_100mm_gold"),
        200_000,
        demo.seeds.derive("shell_burst"),
    )
    scheduler = ShowScheduler(demo)
    clock = demo.make_clock(120, PlaybackMode.DETERMINISTIC)
    bursts: list[str] = []
    # The window must cover the last launch plus its fuse delay, otherwise a
    # shell still in flight would look like one that never fired.
    horizon_s = (
        max(
            (
                launch.event.launch_time_utc - demo.show_start
            ).total_seconds()
            + launch.profile.fuse_delay_s
            for launch in scheduler.launches
        )
        + 2.0
    )
    for _ in range(int(horizon_s * 120)):
        clock.advance_steps(1)
        for launch in scheduler.due(clock.absolute_time):
            world.launch(
                tuple(float(value) for value in launch.position_eus_m),
                profile=launch.profile,
                azimuth_deg=launch.azimuth_deg,
                elevation_deg=launch.elevation_deg,
                event_id=launch.event_id,
            )
        world.update(1.0 / 120.0)
        bursts.extend(
            event.event_id for event in world.consume_burst_events()
        )
    # Every scheduled shell reaches its break, including mines and curtains,
    # which have no lift charge and open at the tube.
    assert set(bursts) == {launch.event_id for launch in scheduler.launches}
    assert len(bursts) == len(set(bursts))
    assert np.isfinite(world.stars.position_m[: world.stars.count]).all()


def test_a_curtain_without_a_burst_charge_produces_a_finite_arrival() -> None:
    # The waterfall profile declares no burst charge, so its burst energy is
    # zero. The acoustic path must stay finite rather than taking log10(0).
    from simulator.acoustics import FireworkAcoustics
    from simulator.config import AcousticConfig, AtmosphereConfig

    acoustics = FireworkAcoustics(AcousticConfig(), 3)
    arrival = acoustics.predict_arrival(
        np.array([0.0, 40.0, 0.0], dtype=np.float32),
        0.0,
        np.array([0.0, 24.0, 235.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        AtmosphereConfig(),
    )
    assert np.isfinite(arrival.sound_pressure_level_db)
    assert np.isfinite(arrival.propagation_delay_s)
    assert arrival.rms_pressure_pa == 0.0


def test_unknown_profile_in_a_scenario_is_rejected(demo_dict: dict) -> None:
    broken = copy.deepcopy(demo_dict)
    broken["events"][0]["shell_profile_id"] = "not_a_real_shell"
    with pytest.raises(ValueError, match="unknown shell profile"):
        Scenario.from_dict(broken)
