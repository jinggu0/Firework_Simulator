from datetime import datetime, timedelta, timezone

import pytest

from simulator.timebase import (
    KST,
    PlaybackMode,
    SeedRegistry,
    SimulationClock,
    format_kst,
    parse_event_time,
)

EPOCH = datetime.fromisoformat("2024-10-05T19:30:00+09:00")


def test_epoch_must_be_timezone_aware() -> None:
    # There is deliberately no default epoch: the previous implementation fell
    # back to POSIX zero and evaluated the sky at 1970-01-01T00:00:00Z.
    with pytest.raises(ValueError, match="timezone-aware epoch"):
        SimulationClock(datetime(2024, 10, 5, 19, 30))


def test_parse_event_time_rejects_naive_input() -> None:
    with pytest.raises(ValueError, match="no UTC offset"):
        parse_event_time("2024-10-05T19:30:00")
    assert parse_event_time("2024-10-05T19:30:00+09:00") == EPOCH


def test_absolute_time_is_stored_utc_and_displayed_kst() -> None:
    clock = SimulationClock(EPOCH, physics_hz=120)
    assert clock.epoch.tzinfo is timezone.utc
    assert clock.epoch.isoformat() == "2024-10-05T10:30:00+00:00"
    assert format_kst(clock.epoch) == "2024-10-05T19:30:00+09:00"
    assert clock.local_time.utcoffset() == timedelta(hours=9)
    assert KST.utcoffset(None) == timedelta(hours=9)


def test_playback_position_is_exact_over_many_steps() -> None:
    # Position is reconstructed from an integer step count, so 12 000 steps of
    # 1/120 s must land on exactly 100 s with no accumulated float drift.
    clock = SimulationClock(EPOCH, physics_hz=120)
    for _ in range(12_000):
        clock.advance_steps(1)
    assert clock.step_index == 12_000
    assert clock.playback_time_s == 100.0
    assert clock.absolute_time == EPOCH + timedelta(seconds=100)


def test_deterministic_mode_ignores_wall_clock_frame_time() -> None:
    steady = SimulationClock(EPOCH, 120, PlaybackMode.DETERMINISTIC)
    jittery = SimulationClock(EPOCH, 120, PlaybackMode.DETERMINISTIC)
    for index in range(600):
        steady.consume_frame(1.0 / 60.0)
        jittery.consume_frame(0.0 if index % 3 else 0.2)
    assert steady.step_index == jittery.step_index == 600
    assert steady.playback_time_s == jittery.playback_time_s
    assert steady.absolute_time == jittery.absolute_time


def test_realtime_mode_caps_catch_up_and_clamps_frame_time() -> None:
    clock = SimulationClock(EPOCH, 120, PlaybackMode.REALTIME, max_catch_up_steps=8)
    # A one-second stall would otherwise demand 120 steps; the cap keeps the
    # frame bounded, matching FixedStepClock's existing spiral-of-death guard.
    assert clock.consume_frame(1.0) == 8
    assert clock.step_index == 8


def test_pause_stops_both_advance_paths() -> None:
    clock = SimulationClock(EPOCH, 120, PlaybackMode.DETERMINISTIC)
    clock.advance_steps(10)
    clock.set_paused(True)
    assert clock.consume_frame(1.0 / 60.0) == 0
    assert clock.advance_steps(5) == 0
    assert clock.step_index == 10
    clock.set_paused(False)
    clock.advance_steps(1)
    assert clock.step_index == 11


def test_rate_scales_realtime_playback() -> None:
    clock = SimulationClock(EPOCH, 120, PlaybackMode.REALTIME)
    clock.set_rate(2.0)
    assert clock.consume_frame(1.0 / 60.0) == 4
    with pytest.raises(ValueError, match="non-negative"):
        clock.set_rate(-1.0)


def test_seeking_by_absolute_posix_and_playback_agree() -> None:
    clock = SimulationClock(EPOCH, physics_hz=120)
    target = EPOCH + timedelta(seconds=42.5)
    clock.seek_to_absolute(target)
    assert clock.playback_time_s == pytest.approx(42.5, abs=1e-9)
    by_posix = SimulationClock(EPOCH, physics_hz=120)
    by_posix.seek_to_posix(target.timestamp())
    assert by_posix.step_index == clock.step_index
    by_playback = SimulationClock(EPOCH, physics_hz=120)
    by_playback.seek_to_playback_s(42.5)
    assert by_playback.step_index == clock.step_index


def test_seeking_rejects_naive_and_negative_targets() -> None:
    clock = SimulationClock(EPOCH, physics_hz=120)
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.seek_to_absolute(datetime(2024, 10, 5, 19, 30))
    with pytest.raises(ValueError, match="non-negative"):
        clock.seek_to_playback_s(-1.0)


def test_seed_registry_is_stable_and_independent_per_name() -> None:
    registry = SeedRegistry(20241005)
    # Stability across calls and across processes: blake2b, not the built-in
    # randomised string hash.
    assert registry.derive("shell_burst") == registry.derive("shell_burst")
    assert SeedRegistry(20241005).derive("smoke") == registry.derive("smoke")
    names = ("shell_burst", "smoke", "cloud_field", "star_catalogue", "acoustics")
    values = [registry.derive(name) for name in names]
    assert len(set(values)) == len(names)
    assert all(0 <= value < 2**31 for value in values)
    assert SeedRegistry(20241006).derive("smoke") != registry.derive("smoke")


def test_state_snapshot_reports_both_timezones() -> None:
    clock = SimulationClock(EPOCH, 120, PlaybackMode.DETERMINISTIC)
    clock.advance_steps(240)
    state = clock.state()
    assert state["playback_time_s"] == 2.0
    assert state["absolute_time_utc"] == "2024-10-05T10:30:02+00:00"
    assert state["absolute_time_kst"] == "2024-10-05T19:30:02+09:00"
    assert state["mode"] == "deterministic"
