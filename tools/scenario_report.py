"""Emit a machine-checkable state report for a scenario at a chosen instant.

This is the verifiable output of the scenario/time/provenance foundation: it
loads a scenario file, builds a deterministic clock at the reference epoch,
resolves observers from WGS84 into local East-Up-South metres, samples the
astronomical and atmospheric state, and prints the result as JSON with the
provenance grade of every input.

Run headlessly, without OpenGL:

    python -m tools.scenario_report
    python -m tools.scenario_report --at 2024-10-05T20:00:00+09:00
    python -m tools.scenario_report --steps 120 --check-determinism
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulator.astronomy import AstronomyModel
from simulator.environment import EnvironmentTimeline
from simulator.provenance import ConfidenceGrade
from simulator.scenario import DEFAULT_SCENARIO_PATH, Scenario
from simulator.timebase import PlaybackMode, format_kst, parse_event_time

DEFAULT_ENVIRONMENT_PATH = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "yeouido_2024-10-05_environment.json"
)


def observer_report(scenario: Scenario) -> list[dict[str, Any]]:
    """Resolve every observer to local metres and back to WGS84.

    The round trip is reported rather than assumed: a non-zero residual here
    would silently corrupt every position-error metric downstream.
    """

    plane = scenario.tangent_plane
    entries: list[dict[str, Any]] = []
    for observer in scenario.observers:
        local = scenario.observer_position_eus_m(observer.observer_id)
        latitude, longitude, altitude = plane.to_geodetic(local)
        residual_m = float(
            np.linalg.norm(
                local
                - plane.to_local(latitude, longitude, altitude)
            )
        )
        entries.append(
            {
                "observer_id": observer.observer_id,
                "name": observer.name,
                "wgs84": observer.position.to_dict(),
                "local_eus_m": {
                    "east": float(local[0]),
                    "up": float(local[1]),
                    "south": float(local[2]),
                },
                "geodetic_round_trip": {
                    "latitude_deg": latitude,
                    "longitude_deg": longitude,
                    "altitude_m": altitude,
                    "residual_m": residual_m,
                },
                "confidence_grade": scenario.provenance.grade_for(
                    "observers"
                ).value,
                "notes": observer.notes,
            }
        )
    return entries


def celestial_report(scenario: Scenario, timestamp: float) -> dict[str, Any]:
    observer = scenario.default_observer
    model = AstronomyModel(
        observer.position.latitude_deg,
        observer.position.longitude_deg,
        observer.position.altitude_m,
    )
    state = model.sample(timestamp)
    return {
        "observer_id": observer.observer_id,
        "sun": {
            "azimuth_deg": state.sun_azimuth_deg,
            "altitude_deg": state.sun_altitude_deg,
            "direction_eus": [float(value) for value in state.sun_direction_eus],
        },
        "moon": {
            "azimuth_deg": state.moon_azimuth_deg,
            "altitude_deg": state.moon_altitude_deg,
            "phase_fraction": state.moon_phase_fraction,
            "illuminance_lux": state.moon_illuminance_lux,
            "direction_eus": [float(value) for value in state.moon_direction_eus],
        },
        "twilight_illuminance_lux": state.twilight_illuminance_lux,
        "confidence_grade": scenario.provenance.grade_for("astronomy").value,
    }


def environment_report(
    scenario: Scenario, timestamp: float, path: Path
) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "MISSING",
            "path": str(path),
            "confidence_grade": ConfidenceGrade.UNVERIFIED.value,
        }
    timeline = EnvironmentTimeline.load(path)
    atmosphere = timeline.sample(timestamp)
    wind = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float64)
    return {
        "status": "LOADED",
        "path": str(path),
        "source": timeline.source,
        "temperature_k": atmosphere.temperature_k,
        "pressure_pa": atmosphere.pressure_pa,
        "relative_humidity": atmosphere.relative_humidity,
        "air_density_kg_m3": atmosphere.air_density_kg_m3,
        "cloud_cover_fraction": atmosphere.cloud_cover_fraction,
        "wind_velocity_eus_mps": [float(value) for value in wind],
        "horizontal_wind_speed_mps": float(np.hypot(wind[0], wind[2])),
        "confidence_grade": scenario.provenance.grade_for(
            "environment.weather"
        ).value,
    }


def determinism_report(scenario: Scenario, steps: int, physics_hz: int) -> dict[str, Any]:
    """Advance two independent deterministic clocks and compare exactly.

    Validation metric V-02. Equality must be exact: playback position is an
    integer step count, so any drift indicates a real defect rather than
    floating-point accumulation.
    """

    first = scenario.make_clock(physics_hz, PlaybackMode.DETERMINISTIC)
    second = scenario.make_clock(physics_hz, PlaybackMode.DETERMINISTIC)
    for _ in range(steps):
        first.consume_frame(1.0 / 60.0)
    # Wall-clock frame times differ deliberately; deterministic mode must ignore
    # them entirely.
    for index in range(steps):
        second.consume_frame(0.001 * (index % 7))
    return {
        "steps": steps,
        "identical_step_index": first.step_index == second.step_index,
        "identical_playback_time": first.playback_time_s == second.playback_time_s,
        "identical_absolute_time": first.absolute_time == second.absolute_time,
        "playback_time_s": first.playback_time_s,
        "absolute_time_kst": format_kst(first.absolute_time),
    }


def build_report(
    scenario: Scenario,
    at: str | None,
    steps: int,
    physics_hz: int,
    environment_path: Path,
    check_determinism: bool,
) -> dict[str, Any]:
    clock = scenario.make_clock(physics_hz, PlaybackMode.DETERMINISTIC)
    # Playback is anchored at the show start; the default report instant is the
    # scenario's reference epoch, and --steps advances from there.
    clock.seek_to_absolute(
        parse_event_time(at) if at is not None else scenario.reference_epoch
    )
    if steps:
        clock.advance_steps(steps)
    timestamp = clock.posix_timestamp

    report: dict[str, Any] = {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "schema_version": scenario.schema_version,
            "source_path": str(scenario.source_path or ""),
            "reference_epoch_utc": scenario.reference_epoch.isoformat(),
            "reference_epoch_kst": format_kst(scenario.reference_epoch),
            "show_start_kst": format_kst(scenario.show_start),
            "show_end_kst": format_kst(scenario.show_end),
            "show_duration_s": scenario.show_duration_s(),
            "origin": scenario.origin.to_dict(),
            "render_vertical_datum": scenario.render_vertical_datum,
            "master_seed": scenario.seeds.master_seed,
        },
        "clock": clock.state(),
        "seeds": {
            name: scenario.seeds.derive(name)
            for name in ("shell_burst", "smoke", "cloud_field", "star_catalogue",
                         "acoustics")
        },
        "observers": observer_report(scenario),
        "celestial": celestial_report(scenario, timestamp),
        "environment": environment_report(scenario, timestamp, environment_path),
        "provenance_summary": scenario.provenance.summary(),
        "unverified_paths": scenario.provenance.paths_with_grade(
            ConfidenceGrade.UNVERIFIED
        ),
        "missing_datasets": scenario.missing_datasets(),
    }
    if check_determinism:
        report["determinism"] = determinism_report(
            scenario, steps or 120, physics_hz
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report the deterministic scenario, clock, observer, celestial, "
            "and atmospheric state for a historical scenario file."
        )
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument(
        "--environment", type=Path, default=DEFAULT_ENVIRONMENT_PATH
    )
    parser.add_argument(
        "--at",
        type=str,
        default=None,
        help=(
            "ISO-8601 instant WITH offset, e.g. 2024-10-05T20:00:00+09:00. "
            "Defaults to the scenario reference epoch."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Fixed physics steps to advance past the epoch before reporting.",
    )
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Run the V-02 deterministic replay comparison.",
    )
    args = parser.parse_args()

    scenario = Scenario.load(args.scenario)
    report = build_report(
        scenario,
        args.at,
        args.steps,
        args.physics_hz,
        args.environment,
        args.check_determinism,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
