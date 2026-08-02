import copy
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from simulator.provenance import ConfidenceGrade
from simulator.scenario import (
    DEFAULT_SCENARIO_PATH,
    SCHEMA_VERSION,
    GeodeticPosition,
    Scenario,
    ShowEvent,
    load_default_scenario,
)
from simulator.timebase import PlaybackMode


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return load_default_scenario()


@pytest.fixture()
def scenario_dict() -> dict:
    return json.loads(DEFAULT_SCENARIO_PATH.read_text(encoding="utf-8"))


def test_shipped_scenario_matches_the_official_show_window(scenario: Scenario) -> None:
    assert scenario.scenario_id == "yeouido-2024-10-05"
    assert scenario.schema_version == SCHEMA_VERSION
    assert scenario.show_start == datetime.fromisoformat(
        "2024-10-05T19:20:00+09:00"
    )
    assert scenario.show_end == datetime.fromisoformat(
        "2024-10-05T20:30:00+09:00"
    )
    assert scenario.show_duration_s() == 4_200.0
    assert scenario.show_start <= scenario.reference_epoch <= scenario.show_end


def test_reference_epoch_is_the_documented_instant(scenario: Scenario) -> None:
    assert scenario.reference_epoch == datetime.fromisoformat(
        "2024-10-05T19:30:00+09:00"
    )
    assert scenario.reference_epoch.timestamp() == 1_728_124_200.0


def test_observer_replaces_the_previously_hardcoded_position(
    scenario: Scenario,
) -> None:
    # These are the values that used to live in simulator/app.py and that
    # tests/test_astronomy.py still asserts against, so moving them into the
    # scenario must not change them.
    observer = scenario.default_observer
    assert observer.position.latitude_deg == 37.529
    assert observer.position.longitude_deg == 126.935
    assert observer.position.altitude_m == 5.0
    assert scenario.observer(observer.observer_id) == observer
    with pytest.raises(KeyError, match="unknown observer"):
        scenario.observer("nobody")


def test_observer_resolves_to_local_east_up_south_metres(
    scenario: Scenario,
) -> None:
    local = scenario.observer_position_eus_m("scene_origin_reference")
    # The observer sits on the scene origin, 5 m above the ellipsoidal datum.
    assert abs(local[0]) < 1e-6
    assert local[1] == pytest.approx(5.0, abs=1e-6)
    assert abs(local[2]) < 1e-6


def test_geodetic_round_trip_is_within_one_millimetre(scenario: Scenario) -> None:
    # Validation metric V-03. float64 ECEF over a 5 km baseline retains about a
    # micrometre, so 1 mm is three orders above numerical noise: a failure here
    # means a formula error, not precision loss.
    plane = scenario.tangent_plane
    for east, up, south in (
        (0.0, 0.0, 0.0),
        (2_500.0, 120.0, -2_000.0),
        (-2_500.0, -5.0, 2_000.0),
        (1_234.5, 322.0, 987.6),
    ):
        local = np.array([east, up, south], dtype=np.float64)
        latitude, longitude, altitude = plane.to_geodetic(local)
        recovered = plane.to_local(latitude, longitude, altitude)
        assert float(np.linalg.norm(recovered - local)) < 1e-3


def test_clock_playback_origin_is_the_show_start(scenario: Scenario) -> None:
    # Playback position zero is when the performance begins. The reference
    # epoch is a separate reporting instant ten minutes later; anchoring
    # playback there would make the show start a negative position.
    clock = scenario.make_clock(120, PlaybackMode.DETERMINISTIC)
    assert clock.absolute_time == scenario.show_start
    clock.advance_steps(120)
    assert clock.absolute_time == scenario.show_start + timedelta(seconds=1)


def test_clock_can_seek_forward_to_the_reference_epoch(scenario: Scenario) -> None:
    clock = scenario.make_clock(120, PlaybackMode.DETERMINISTIC)
    clock.seek_to_absolute(scenario.reference_epoch)
    assert clock.absolute_time == scenario.reference_epoch
    assert clock.playback_time_s == 600.0


def test_clock_epoch_can_be_overridden(scenario: Scenario) -> None:
    clock = scenario.make_clock(
        120, PlaybackMode.DETERMINISTIC, epoch=scenario.reference_epoch
    )
    assert clock.absolute_time == scenario.reference_epoch


def test_deterministic_replay_is_exact(scenario: Scenario) -> None:
    # Validation metric V-02: equality must be exact, not approximate.
    first = scenario.make_clock(120, PlaybackMode.DETERMINISTIC)
    second = scenario.make_clock(120, PlaybackMode.DETERMINISTIC)
    for index in range(600):
        first.consume_frame(1.0 / 60.0)
        second.consume_frame(0.05 if index % 5 else 0.0)
    assert first.step_index == second.step_index
    assert first.playback_time_s == second.playback_time_s
    assert first.absolute_time == second.absolute_time


def test_unsourced_collections_are_empty_and_graded_unverified(
    scenario: Scenario,
) -> None:
    # No dated source for barge coordinates or a firing timeline has been
    # located. Populating either with an estimate would convert an unknown into
    # an apparent measurement.
    assert scenario.launch_sites == ()
    assert scenario.events == ()
    assert scenario.missing_datasets() == ["launch_sites", "events"]
    assert scenario.provenance.grade_for("launch_sites") is ConfidenceGrade.UNVERIFIED
    assert scenario.provenance.grade_for("events") is ConfidenceGrade.UNVERIFIED


def test_sourced_fields_carry_evidence_grades(scenario: Scenario) -> None:
    assert scenario.provenance.grade_for("show") is ConfidenceGrade.MEASURED
    assert (
        scenario.provenance.grade_for("environment.weather")
        is ConfidenceGrade.MEASURED
    )
    assert scenario.provenance.grade_for("scene.osm") is ConfidenceGrade.MEASURED
    assert scenario.provenance.grade_for("astronomy") is ConfidenceGrade.MEASURED
    # The observer and the origin are project conventions, not observations.
    assert scenario.provenance.grade_for("observers") is ConfidenceGrade.MODELLED
    assert scenario.provenance.grade_for("origin") is ConfidenceGrade.MODELLED


def test_vertical_datum_names_the_measured_event_water_reference(
    scenario: Scenario,
) -> None:
    assert (
        scenario.render_vertical_datum
        == "wamis_hangang_bridge_2024-10-05_19:20"
    )
    assert (
        scenario.provenance.grade_for("render_vertical_datum")
        is ConfidenceGrade.MEASURED
    )


def test_seed_registry_comes_from_the_scenario(scenario: Scenario) -> None:
    assert scenario.seeds.master_seed == 20241005
    assert scenario.seeds.derive("shell_burst") != scenario.seeds.derive("smoke")


def test_schema_version_mismatch_is_rejected(scenario_dict: dict) -> None:
    scenario_dict["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="unsupported scenario schema_version"):
        Scenario.from_dict(scenario_dict)


def test_naive_epoch_is_rejected(scenario_dict: dict) -> None:
    scenario_dict["reference_epoch"] = "2024-10-05T19:30:00"
    with pytest.raises(ValueError, match="no UTC offset"):
        Scenario.from_dict(scenario_dict)


def test_scenario_requires_an_observer(scenario_dict: dict) -> None:
    scenario_dict["observers"] = []
    with pytest.raises(ValueError, match="at least one observer"):
        Scenario.from_dict(scenario_dict)


def test_duplicate_observer_ids_are_rejected(scenario_dict: dict) -> None:
    scenario_dict["observers"] = scenario_dict["observers"] * 2
    with pytest.raises(ValueError, match="duplicate observer_id"):
        Scenario.from_dict(scenario_dict)


def test_show_end_before_start_is_rejected(scenario_dict: dict) -> None:
    scenario_dict["show"]["end"] = "2024-10-05T19:00:00+09:00"
    with pytest.raises(ValueError, match="precedes start"):
        Scenario.from_dict(scenario_dict)


def test_event_referencing_an_unknown_launch_site_is_rejected(
    scenario_dict: dict,
) -> None:
    scenario_dict["events"] = [
        {
            "event_id": "e1",
            "launch_time_utc": "2024-10-05T19:21:00+09:00",
            "launch_site_id": "barge_that_does_not_exist",
            "shell_profile_id": "peony_100mm_gold",
        }
    ]
    with pytest.raises(ValueError, match="unknown launch site"):
        Scenario.from_dict(scenario_dict)


def test_events_and_launch_sites_parse_when_a_source_supplies_them(
    scenario_dict: dict,
) -> None:
    # The schema shape is fixed now so that arriving data does not force a
    # migration. This exercises it with an explicitly synthetic record.
    populated = copy.deepcopy(scenario_dict)
    populated["launch_sites"] = [
        {
            "site_id": "test_barge",
            "name": "Synthetic test barge",
            "position": {
                "latitude_deg": 37.5285,
                "longitude_deg": 126.9375,
                "altitude_m": 0.0,
            },
            "notes": "Synthetic fixture, not a historical position.",
        }
    ]
    populated["events"] = [
        {
            "event_id": "e1",
            "launch_time_utc": "2024-10-05T19:21:00+09:00",
            "launch_site_id": "test_barge",
            "shell_profile_id": "peony_100mm_gold",
            "confidence_grade": "D",
            "tube_elevation_deg": 88.0,
            "calibre_mm": 100.0,
            "muzzle_velocity_mps": 78.0,
            "fuse_delay_s": 3.05,
            "seed_name": "shell_burst",
        }
    ]
    loaded = Scenario.from_dict(populated)
    assert loaded.missing_datasets() == []
    event = loaded.events[0]
    assert event.confidence_grade is ConfidenceGrade.ARTISTIC
    assert event.calibre_mm == 100.0
    # Unsupplied optional fields stay None so "unknown" is distinguishable
    # from a defaulted zero.
    assert event.tube_azimuth_deg is None
    position = loaded.launch_site_position_eus_m("test_barge")
    assert position.shape == (3,)
    assert np.isfinite(position).all()
    assert ShowEvent.from_dict(event.to_dict()) == event


def test_unknown_top_level_keys_survive_a_round_trip(scenario_dict: dict) -> None:
    scenario_dict["future_field"] = {"written_by": "a newer tool"}
    loaded = Scenario.from_dict(scenario_dict)
    assert loaded.extra["future_field"] == {"written_by": "a newer tool"}


def test_position_validation_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValueError, match="latitude_deg out of range"):
        GeodeticPosition.from_dict({"latitude_deg": 137.5, "longitude_deg": 126.9})
    with pytest.raises(ValueError, match="longitude_deg out of range"):
        GeodeticPosition.from_dict({"latitude_deg": 37.5, "longitude_deg": 226.9})
    with pytest.raises(ValueError, match="requires numeric"):
        GeodeticPosition.from_dict({"latitude_deg": 37.5})


def test_missing_scenario_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="scenario file not found"):
        Scenario.load(tmp_path / "absent.json")


def test_event_referencing_an_unknown_shell_profile_is_rejected(
    scenario_dict: dict,
) -> None:
    # Resolving profile ids at load time turns a typo into a load error rather
    # than a shell that silently never fires.
    scenario_dict["launch_sites"] = [
        {
            "site_id": "s1",
            "position": {"latitude_deg": 37.5301, "longitude_deg": 126.935},
        }
    ]
    scenario_dict["events"] = [
        {
            "event_id": "e1",
            "launch_time_utc": "2024-10-05T19:21:00+09:00",
            "launch_site_id": "s1",
            "shell_profile_id": "no_such_shell",
        }
    ]
    with pytest.raises(ValueError, match="unknown shell profile"):
        Scenario.from_dict(scenario_dict)


def test_duplicate_event_ids_are_rejected(scenario_dict: dict) -> None:
    scenario_dict["launch_sites"] = [
        {
            "site_id": "s1",
            "position": {"latitude_deg": 37.5301, "longitude_deg": 126.935},
        }
    ]
    event = {
        "event_id": "e1",
        "launch_time_utc": "2024-10-05T19:21:00+09:00",
        "launch_site_id": "s1",
        "shell_profile_id": "peony_100mm_gold",
    }
    scenario_dict["events"] = [event, dict(event)]
    with pytest.raises(ValueError, match="duplicate event_id"):
        Scenario.from_dict(scenario_dict)
