"""Implementations of the validation metrics that run without OpenGL.

Every metric here computes CPU-side state only, so a pass or fail cannot be
flipped by GPU floating-point ordering. Each returns a
:class:`~simulator.validation.report.MetricResult` carrying its residuals, so
the report shows how much margin a pass had rather than only that it passed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import astronomy
import numpy as np

from .. import atmosphere
from ..acoustics import FireworkAcoustics, speed_of_sound_mps
from ..astronomy import horizontal_direction_eus
from ..starcatalogue import StarCatalogue
from ..config import (
    AcousticConfig,
    AtmosphereConfig,
    ShellConfig,
    SimulationConfig,
    SmokeConfig,
)
from ..fluid import SmokeFluid2D
from ..lighting import burn_profile, combustion_peak_radiant_power_w
from ..physics import FireworkWorld
from ..scenario import Scenario
from ..shells import SHELL_LIBRARY, BreakPattern, ShellLibrary, ShellProfile
from ..timebase import PlaybackMode
from . import catalogue
from .report import MetricResult, MetricStatus


def _astronomy_time(moment: datetime) -> astronomy.Time:
    return astronomy.Time(
        moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    )


def _angle_difference_deg(first: float, second: float) -> float:
    """Smallest absolute separation between two bearings, in degrees."""

    return abs((first - second + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------
# V-03 geodetic round trip
# ---------------------------------------------------------------------------


def geodetic_round_trip(scenario: Scenario) -> MetricResult:
    """Convert local East-Up-South positions to WGS84 and back."""

    plane = scenario.tangent_plane
    # The sample grid spans the shipped 5 x 4 km scene extent and the terrain
    # height range recorded for it, plus every scenario observer.
    easts = np.linspace(-2_500.0, 2_500.0, 9)
    ups = np.array([-10.0, 0.0, 63.0, 252.0, 400.0])
    souths = np.linspace(-2_000.0, 2_000.0, 9)
    samples: list[np.ndarray] = [
        np.array([east, up, south], dtype=np.float64)
        for east in easts
        for up in ups
        for south in souths
    ]
    samples.extend(
        scenario.observer_position_eus_m(observer.observer_id)
        for observer in scenario.observers
    )

    worst_residual_m = 0.0
    worst_sample: list[float] = []
    for sample in samples:
        latitude, longitude, altitude = plane.to_geodetic(sample)
        residual = float(
            np.linalg.norm(plane.to_local(latitude, longitude, altitude) - sample)
        )
        if residual > worst_residual_m:
            worst_residual_m = residual
            worst_sample = [float(value) for value in sample]

    tolerance_m = 1.0e-3
    passed = worst_residual_m < tolerance_m
    return MetricResult(
        spec=catalogue.V03,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"worst round-trip residual {worst_residual_m:.3e} m over "
            f"{len(samples)} samples"
        ),
        residuals={
            "worst_residual_m": worst_residual_m,
            "tolerance_m": tolerance_m,
        },
        detail={
            "sample_count": len(samples),
            "worst_sample_eus_m": worst_sample,
            "origin": scenario.origin.to_dict(),
        },
    )


# ---------------------------------------------------------------------------
# V-02 deterministic replay
# ---------------------------------------------------------------------------


def _replay_world(scenario: Scenario, steps: int) -> dict[str, np.ndarray]:
    world = FireworkWorld(
        AtmosphereConfig(),
        ShellConfig(burst_star_count=2_048),
        max_particles=4_096,
        seed=scenario.seeds.derive("shell_burst"),
    )
    world.launch()
    burst_positions: list[np.ndarray] = []
    for _ in range(steps):
        world.update(1.0 / 120.0)
        for burst in world.consume_burst_events():
            burst_positions.append(np.asarray(burst.position_m, dtype=np.float64))
    count = world.stars.count
    return {
        "position_m": world.stars.position_m[:count].copy(),
        "velocity_mps": world.stars.velocity_mps[:count].copy(),
        "age_s": world.stars.age_s[:count].copy(),
        "burst_positions_m": (
            np.stack(burst_positions) if burst_positions else np.zeros((0, 3))
        ),
    }


def deterministic_replay(scenario: Scenario, steps: int = 400) -> MetricResult:
    """Run the clock, ballistics, and acoustics twice and compare exactly."""

    # 1. Clock: deterministic playback must ignore wall-clock frame time.
    first_clock = scenario.make_clock(120, PlaybackMode.DETERMINISTIC)
    second_clock = scenario.make_clock(120, PlaybackMode.DETERMINISTIC)
    for index in range(steps):
        first_clock.consume_frame(1.0 / 60.0)
        second_clock.consume_frame(0.05 if index % 5 else 0.0)
    clock_identical = (
        first_clock.step_index == second_clock.step_index
        and first_clock.playback_time_s == second_clock.playback_time_s
        and first_clock.absolute_time == second_clock.absolute_time
    )

    # 2. Ballistics: the same scenario seed must reproduce every star.
    first_world = _replay_world(scenario, steps)
    second_world = _replay_world(scenario, steps)
    world_deltas = {
        f"world_{name}_max_abs_delta": float(
            np.abs(first_world[name] - second_world[name]).max()
        )
        if first_world[name].size
        else 0.0
        for name in first_world
    }
    shapes_match = all(
        first_world[name].shape == second_world[name].shape for name in first_world
    )

    # 3. Acoustics: arrival prediction and synthesised PCM must be identical.
    atmosphere = AtmosphereConfig()
    listener = np.array([0.0, 24.0, 235.0], dtype=np.float32)
    listener_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    source = np.array([0.0, 159.0, 0.0], dtype=np.float32)
    arrivals = []
    waveforms = []
    for _ in range(2):
        acoustics = FireworkAcoustics(
            AcousticConfig(), scenario.seeds.derive("acoustics")
        )
        arrival = acoustics.predict_arrival(
            source, 255_000.0, listener, listener_right, atmosphere
        )
        arrivals.append(arrival)
        waveforms.append(acoustics.synthesize_pcm(arrival))
    acoustic_delta = abs(
        arrivals[0].propagation_delay_s - arrivals[1].propagation_delay_s
    )
    level_delta = abs(
        arrivals[0].sound_pressure_level_db - arrivals[1].sound_pressure_level_db
    )
    pcm_identical = np.array_equal(waveforms[0], waveforms[1])

    # Two empty star fields compare equal, so a run that never reached the
    # fuse delay would report determinism without having tested anything.
    star_count = int(first_world["position_m"].shape[0])
    burst_count = int(first_world["burst_positions_m"].shape[0])
    exercised = star_count > 0 and burst_count > 0

    residuals = {
        **world_deltas,
        "acoustic_delay_delta_s": acoustic_delta,
        "acoustic_level_delta_db": level_delta,
        "stars_compared": float(star_count),
        "bursts_compared": float(burst_count),
    }
    passed = (
        exercised
        and clock_identical
        and shapes_match
        and all(value == 0.0 for value in world_deltas.values())
        and acoustic_delta == 0.0
        and level_delta == 0.0
        and pcm_identical
    )
    message = (
        f"{steps} fixed steps replayed; {star_count} stars and "
        f"{burst_count} bursts compared"
    )
    if not exercised:
        minimum_steps = math.ceil(ShellConfig().fuse_delay_s * 120.0) + 1
        message = (
            f"{steps} steps produced no burst, so nothing was compared; "
            f"at least {minimum_steps} steps are needed to reach the "
            f"{ShellConfig().fuse_delay_s} s fuse delay at 120 Hz"
        )
    return MetricResult(
        spec=catalogue.V02,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=message,
        residuals=residuals,
        detail={
            "steps": steps,
            "clock_identical": clock_identical,
            "star_count": star_count,
            "burst_count": burst_count,
            "comparison_exercised": exercised,
            "star_array_shapes_match": shapes_match,
            "pcm_identical": bool(pcm_identical),
            "shell_burst_seed": scenario.seeds.derive("shell_burst"),
            "acoustics_seed": scenario.seeds.derive("acoustics"),
        },
    )


# ---------------------------------------------------------------------------
# V-15 / V-16 astronomical transform checks
# ---------------------------------------------------------------------------


def _independent_horizontal(
    right_ascension_h: float,
    declination_deg: float,
    latitude_deg: float,
    longitude_deg: float,
    time: astronomy.Time,
) -> tuple[float, float]:
    """Independent spherical-trig transform from apparent RA/Dec to az/alt.

    Deliberately reimplemented from the hour-angle relations rather than reusing
    the library horizon routine, so that a sign, hemisphere, or hour-angle
    convention error in this project's usage is detectable.
    """

    greenwich_sidereal_deg = astronomy.SiderealTime(time) * 15.0
    local_sidereal_deg = (greenwich_sidereal_deg + longitude_deg) % 360.0
    hour_angle = math.radians(
        (local_sidereal_deg - right_ascension_h * 15.0) % 360.0
    )
    declination = math.radians(declination_deg)
    latitude = math.radians(latitude_deg)
    sine_altitude = math.sin(declination) * math.sin(latitude) + math.cos(
        declination
    ) * math.cos(latitude) * math.cos(hour_angle)
    altitude_deg = math.degrees(math.asin(min(max(sine_altitude, -1.0), 1.0)))
    azimuth_deg = (
        math.degrees(
            math.atan2(
                -math.sin(hour_angle) * math.cos(declination),
                math.cos(latitude) * math.sin(declination)
                - math.sin(latitude) * math.cos(declination) * math.cos(hour_angle),
            )
        )
        % 360.0
    )
    return azimuth_deg, altitude_deg


def _sample_instants(scenario: Scenario) -> list[datetime]:
    span_s = scenario.show_duration_s()
    return [
        scenario.show_start,
        scenario.reference_epoch,
        scenario.show_end,
        scenario.show_start.fromtimestamp(
            scenario.show_start.timestamp() + span_s * 0.5, timezone.utc
        ),
    ]


def horizontal_transform_cross_check(scenario: Scenario) -> MetricResult:
    """Compare the library horizon transform against an independent derivation."""

    observer_record = scenario.default_observer
    latitude = observer_record.position.latitude_deg
    longitude = observer_record.position.longitude_deg
    observer = astronomy.Observer(
        latitude, longitude, observer_record.position.altitude_m
    )

    worst_azimuth_deg = 0.0
    worst_altitude_deg = 0.0
    comparisons = 0
    for moment in _sample_instants(scenario):
        time = _astronomy_time(moment)
        for body in (astronomy.Body.Sun, astronomy.Body.Moon):
            equatorial = astronomy.Equator(
                body, time, observer, ofdate=True, aberration=True
            )
            # Refraction is disabled on the library side because the reference
            # derivation is purely geometric; comparing against the refracted
            # horizon would measure the refraction model, not the transform.
            reference = astronomy.Horizon(
                time,
                observer,
                equatorial.ra,
                equatorial.dec,
                astronomy.Refraction.Airless,
            )
            azimuth, altitude = _independent_horizontal(
                equatorial.ra, equatorial.dec, latitude, longitude, time
            )
            worst_azimuth_deg = max(
                worst_azimuth_deg,
                _angle_difference_deg(azimuth, reference.azimuth),
            )
            worst_altitude_deg = max(
                worst_altitude_deg, abs(altitude - reference.altitude)
            )
            comparisons += 1

    tolerance_deg = 1.0e-9
    passed = (
        worst_azimuth_deg < tolerance_deg and worst_altitude_deg < tolerance_deg
    )
    return MetricResult(
        spec=catalogue.V15,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"{comparisons} body/instant comparisons; worst azimuth "
            f"{worst_azimuth_deg:.3e} deg, worst altitude "
            f"{worst_altitude_deg:.3e} deg"
        ),
        residuals={
            "worst_azimuth_deg": worst_azimuth_deg,
            "worst_altitude_deg": worst_altitude_deg,
            "tolerance_deg": tolerance_deg,
        },
        detail={
            "comparisons": comparisons,
            "observer_id": observer_record.observer_id,
            "refraction": "Airless on both sides",
        },
    )


def direction_vector_consistency(scenario: Scenario) -> MetricResult:
    """Verify the East-Up-South direction vectors the renderer consumes."""

    worst_azimuth_deg = 0.0
    worst_altitude_deg = 0.0
    worst_length_error = 0.0
    samples = 0
    for azimuth_deg in np.arange(0.0, 360.0, 15.0):
        for altitude_deg in (-45.0, -14.28, 0.0, 12.5, 47.0, 83.0):
            direction = horizontal_direction_eus(
                float(azimuth_deg), float(altitude_deg)
            )
            length = float(np.linalg.norm(direction))
            worst_length_error = max(worst_length_error, abs(length - 1.0))
            # +X east, +Y up, +Z south, with azimuth measured north-clockwise.
            recovered_altitude = math.degrees(
                math.asin(min(max(float(direction[1]), -1.0), 1.0))
            )
            recovered_azimuth = (
                math.degrees(math.atan2(float(direction[0]), -float(direction[2])))
                % 360.0
            )
            worst_altitude_deg = max(
                worst_altitude_deg, abs(recovered_altitude - altitude_deg)
            )
            if abs(altitude_deg) < 89.9:
                worst_azimuth_deg = max(
                    worst_azimuth_deg,
                    _angle_difference_deg(recovered_azimuth, float(azimuth_deg)),
                )
            samples += 1

    tolerance_deg = 1.0e-5
    tolerance_length = 1.0e-6
    passed = (
        worst_azimuth_deg < tolerance_deg
        and worst_altitude_deg < tolerance_deg
        and worst_length_error < tolerance_length
    )
    return MetricResult(
        spec=catalogue.V16,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"{samples} direction samples; worst azimuth {worst_azimuth_deg:.3e} "
            f"deg, worst altitude {worst_altitude_deg:.3e} deg"
        ),
        residuals={
            "worst_azimuth_deg": worst_azimuth_deg,
            "worst_altitude_deg": worst_altitude_deg,
            "worst_unit_length_error": worst_length_error,
            "tolerance_deg": tolerance_deg,
            "tolerance_length": tolerance_length,
        },
        detail={"samples": samples, "storage_dtype": "float32"},
    )


def external_ephemeris_comparison(
    scenario: Scenario, reference_path: Path | None
) -> MetricResult:
    """Compare computed Sun/Moon horizontal coordinates against a reference file.

    No published ephemeris extract ships with the repository, so this reports
    ``NO_REFERENCE`` until one is supplied. The expected file format is included
    in the result so a supplier can deliver it without further negotiation.
    """

    required_format = {
        "source_id": "e.g. jpl-horizons-2024-10-05",
        "source_url": "retrieval URL",
        "observer": {
            "latitude_deg": 37.529,
            "longitude_deg": 126.935,
            "altitude_m": 5.0,
        },
        "refraction": "normal | airless",
        "samples": [
            {
                "time": "2024-10-05T19:30:00+09:00",
                "body": "sun",
                "azimuth_deg": 0.0,
                "altitude_deg": 0.0,
            }
        ],
    }
    if reference_path is None or not Path(reference_path).exists():
        return MetricResult(
            spec=catalogue.V01,
            status=MetricStatus.NO_REFERENCE,
            message=(
                "no external ephemeris extract present; supply one to enable "
                "this metric"
            ),
            detail={
                "expected_path": str(reference_path) if reference_path else "",
                "required_dataset": catalogue.DATASET_EXTERNAL_EPHEMERIS,
                "required_format": required_format,
            },
        )

    data = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    observer_data = data.get("observer", {})
    observer = astronomy.Observer(
        float(observer_data.get("latitude_deg", scenario.default_observer.position.latitude_deg)),
        float(observer_data.get("longitude_deg", scenario.default_observer.position.longitude_deg)),
        float(observer_data.get("altitude_m", scenario.default_observer.position.altitude_m)),
    )
    refraction = (
        astronomy.Refraction.Airless
        if str(data.get("refraction", "normal")).lower() == "airless"
        else astronomy.Refraction.Normal
    )
    bodies = {"sun": astronomy.Body.Sun, "moon": astronomy.Body.Moon}

    worst_azimuth_deg = 0.0
    worst_altitude_deg = 0.0
    compared = 0
    for sample in data.get("samples", []):
        body = bodies.get(str(sample["body"]).lower())
        if body is None:
            continue
        moment = datetime.fromisoformat(str(sample["time"]))
        if moment.tzinfo is None:
            raise ValueError(
                f"ephemeris sample time {sample['time']!r} has no UTC offset"
            )
        time = _astronomy_time(moment)
        equatorial = astronomy.Equator(
            body, time, observer, ofdate=True, aberration=True
        )
        horizon = astronomy.Horizon(
            time, observer, equatorial.ra, equatorial.dec, refraction
        )
        worst_azimuth_deg = max(
            worst_azimuth_deg,
            _angle_difference_deg(
                horizon.azimuth, float(sample["azimuth_deg"])
            ),
        )
        worst_altitude_deg = max(
            worst_altitude_deg,
            abs(horizon.altitude - float(sample["altitude_deg"])),
        )
        compared += 1

    if compared == 0:
        return MetricResult(
            spec=catalogue.V01,
            status=MetricStatus.NO_REFERENCE,
            message="reference file present but contained no usable samples",
            detail={"required_format": required_format},
        )

    tolerance_deg = 0.05
    passed = (
        worst_azimuth_deg < tolerance_deg and worst_altitude_deg < tolerance_deg
    )
    return MetricResult(
        spec=catalogue.V01,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"{compared} samples compared against "
            f"{data.get('source_id', 'unnamed source')}"
        ),
        residuals={
            "worst_azimuth_deg": worst_azimuth_deg,
            "worst_altitude_deg": worst_altitude_deg,
            "tolerance_deg": tolerance_deg,
        },
        detail={
            "samples_compared": compared,
            "source_id": data.get("source_id", ""),
            "source_url": data.get("source_url", ""),
            "refraction": str(refraction),
        },
    )


# ---------------------------------------------------------------------------
# V-14 physical conservation
# ---------------------------------------------------------------------------


def physical_conservation(seed: int = 9) -> MetricResult:
    """Check combustion yields, radiated energy, and pressure projection."""

    # -- combustion mass and thermal yields ---------------------------------
    shell = ShellConfig(
        fuse_delay_s=0.0,
        burst_star_count=64,
        star_lifetime_mean_s=0.35,
        star_lifetime_std_s=0.0,
        star_composition_mass_kg=0.64,
        star_smoke_yield_fraction=0.25,
        star_specific_energy_j_kg=4_000_000.0,
        star_post_combustion_thermal_fraction=0.05,
    )
    world = FireworkWorld(AtmosphereConfig(), shell, 64, 11)
    world.launch((0.0, 120.0, 0.0))
    emitted_smoke_kg = 0.0
    emitted_thermal_j = 0.0
    for step in range(120):
        world.update(1.0 / 120.0)
        if (step + 1) % 4 == 0:
            for emission in world.consume_combustion_emissions():
                emitted_smoke_kg += float(emission.smoke_mass_kg.sum())
                emitted_thermal_j += float(emission.thermal_energy_j.sum())
    expected_smoke_kg = (
        shell.star_composition_mass_kg * shell.star_smoke_yield_fraction
    )
    expected_thermal_j = (
        shell.star_composition_mass_kg
        * shell.star_specific_energy_j_kg
        * shell.star_post_combustion_thermal_fraction
    )
    smoke_error = abs(emitted_smoke_kg - expected_smoke_kg) / expected_smoke_kg
    thermal_error = abs(emitted_thermal_j - expected_thermal_j) / expected_thermal_j

    # -- radiated energy against the chemical budget ------------------------
    lifetime_s = 2.25
    chemical_energy_j = float(
        ShellConfig().star_composition_mass_kg
        / ShellConfig().burst_star_count
        * ShellConfig().star_specific_energy_j_kg
    )
    radiative_fraction = ShellConfig().star_radiative_energy_fraction
    peak_w = float(
        combustion_peak_radiant_power_w(
            np.array([chemical_energy_j]),
            np.array([lifetime_s]),
            radiative_fraction,
        )[0]
    )
    ages_s = np.linspace(0.0, lifetime_s, 200_001)
    worst_energy_error = 0.0
    for star_index in (0, 1, 2, 4_000, 7_999):
        # Same deterministic modulation the renderer applies in StarField.intensity.
        modulation = 0.97 + 0.03 * np.sin(ages_s * 53.0 + star_index * 1.618)
        power_w = peak_w * burn_profile(ages_s / lifetime_s) * modulation
        radiated_j = float(np.trapezoid(power_w, ages_s))
        expected_j = chemical_energy_j * radiative_fraction
        worst_energy_error = max(
            worst_energy_error, abs(radiated_j - expected_j) / expected_j
        )

    # -- pressure projection and plume mass ---------------------------------
    smoke_config = SmokeConfig()
    atmosphere = AtmosphereConfig()
    fluid = SmokeFluid2D(smoke_config, atmosphere)
    injected_mass_kg = 0.0102
    fluid.inject_burst(
        np.array([0.0, 160.0, 0.0], dtype=np.float32), injected_mass_kg, 42_000.0
    )
    cell_volume_m3 = fluid.dx * fluid.dy * smoke_config.plume_depth_m
    resident_mass_kg = float(fluid.density_kg_m3.sum(dtype=np.float64)) * cell_volume_m3
    mass_error = abs(resident_mass_kg - injected_mass_kg) / injected_mass_kg

    # Seed the velocity field with divergent noise and require the projection
    # to remove it. Only public state is touched, so this exercises the shipped
    # solver configuration rather than an internal stage.
    generator = np.random.default_rng(seed)
    fluid.u_mps[:, 1:-1] = generator.normal(0.0, 1.0, fluid.u_mps[:, 1:-1].shape)
    fluid.v_mps[1:-1, :] = generator.normal(0.0, 1.0, fluid.v_mps[1:-1, :].shape)
    divergence_before = float(np.sqrt(np.mean(fluid.divergence() ** 2)))
    fluid.project(1.0 / smoke_config.update_hz)
    divergence_after = float(np.sqrt(np.mean(fluid.divergence() ** 2)))
    divergence_ratio = divergence_after / max(divergence_before, 1e-30)

    yield_tolerance = 2.0e-5
    energy_tolerance = 1.0e-3
    mass_tolerance = 1.0e-3
    # The shipped configuration runs 24 Jacobi iterations per step. Jacobi is a
    # smoother, not an exact solver, so the residual is bounded rather than
    # zero; this threshold is set from measurement of the default grid.
    divergence_ratio_tolerance = 0.60
    passed = (
        smoke_error < yield_tolerance
        and thermal_error < yield_tolerance
        and worst_energy_error < energy_tolerance
        and mass_error < mass_tolerance
        and divergence_ratio < divergence_ratio_tolerance
    )
    return MetricResult(
        spec=catalogue.V14,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"yields within {max(smoke_error, thermal_error):.2e}, radiated "
            f"energy within {worst_energy_error:.2e}, projection reduced "
            f"divergence by {1.0 - divergence_ratio:.1%}"
        ),
        residuals={
            "smoke_yield_relative_error": smoke_error,
            "thermal_yield_relative_error": thermal_error,
            "radiated_energy_relative_error": worst_energy_error,
            "plume_mass_relative_error": mass_error,
            "divergence_before_per_s": divergence_before,
            "divergence_after_per_s": divergence_after,
            "divergence_ratio": divergence_ratio,
            "yield_tolerance": yield_tolerance,
            "energy_tolerance": energy_tolerance,
            "divergence_ratio_tolerance": divergence_ratio_tolerance,
        },
        detail={
            "emitted_smoke_kg": emitted_smoke_kg,
            "expected_smoke_kg": expected_smoke_kg,
            "emitted_thermal_j": emitted_thermal_j,
            "expected_thermal_j": expected_thermal_j,
            "peak_radiant_power_w": peak_w,
            "fluid_backend": SmokeFluid2D.backend_name,
        },
    )


# ---------------------------------------------------------------------------
# V-11 blast propagation model
# ---------------------------------------------------------------------------


def blast_propagation_model(scenario: Scenario) -> MetricResult:
    """Check the Sedov-Taylor phase and the acoustic hand-off analytically."""

    config = AcousticConfig()
    acoustics = FireworkAcoustics(config, scenario.seeds.derive("acoustics"))
    atmosphere = AtmosphereConfig()
    chemical_energy_j = (
        ShellConfig().burst_charge_mass_kg
        * ShellConfig().burst_specific_energy_j_kg
    )
    blast_energy_j = chemical_energy_j * config.blast_wave_energy_fraction

    # Sedov-Taylor gives radius proportional to t^(2/5) exactly.
    times_s = np.geomspace(1e-4, 1e-2, 24)
    radii_m = np.array(
        [
            acoustics.shock_state(float(t), blast_energy_j, atmosphere).radius_m
            for t in times_s
        ]
    )
    slope, _ = np.polyfit(np.log(times_s), np.log(radii_m), 1)
    exponent_error = abs(float(slope) - 0.4)

    # The strong-shock phase must end exactly where shock velocity equals the
    # local sound speed.
    transition_time_s, transition_radius_m = acoustics.strong_shock_transition(
        blast_energy_j, atmosphere
    )
    sound_speed_mps = speed_of_sound_mps(atmosphere)
    velocity_at_transition = acoustics.shock_state(
        transition_time_s, blast_energy_j, atmosphere
    ).velocity_mps
    crossing_error = (
        abs(velocity_at_transition - sound_speed_mps) / sound_speed_mps
    )
    supersonic_before = acoustics.shock_state(
        transition_time_s * 0.5, blast_energy_j, atmosphere
    ).strong_shock
    subsonic_after = not acoustics.shock_state(
        transition_time_s * 2.0, blast_energy_j, atmosphere
    ).strong_shock

    # The predicted arrival must equal the analytic two-phase path.
    listener = np.array([0.0, 24.0, 235.0], dtype=np.float32)
    source = np.array([0.0, 159.0, 0.0], dtype=np.float32)
    listener_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    arrival = acoustics.predict_arrival(
        source, chemical_energy_j, listener, listener_right, atmosphere
    )
    offset = listener.astype(np.float64) - source.astype(np.float64)
    distance_m = float(np.linalg.norm(offset))
    average_height_m = max(0.5 * (float(listener[1]) + float(source[1])), 10.0)
    wind = np.asarray(
        atmosphere.wind_at_height_m(average_height_m), dtype=np.float64
    )
    effective_speed_mps = max(
        sound_speed_mps + float(np.dot(wind, offset / distance_m)), 250.0
    )
    analytic_delay_s = (
        transition_time_s
        + max(distance_m - transition_radius_m, 0.0) / effective_speed_mps
    )
    delay_error_s = abs(arrival.propagation_delay_s - analytic_delay_s)

    exponent_tolerance = 5.0e-3
    crossing_tolerance = 1.0e-2
    delay_tolerance_s = 1.0e-3
    passed = (
        exponent_error < exponent_tolerance
        and crossing_error < crossing_tolerance
        and supersonic_before
        and subsonic_after
        and delay_error_s < delay_tolerance_s
    )
    return MetricResult(
        spec=catalogue.V11,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"Sedov exponent {float(slope):.5f}; flash-to-boom "
            f"{arrival.propagation_delay_s:.4f} s at {distance_m:.1f} m, "
            f"{arrival.sound_pressure_level_db:.1f} dB"
        ),
        residuals={
            "sedov_exponent": float(slope),
            "sedov_exponent_error": exponent_error,
            "sound_speed_crossing_relative_error": crossing_error,
            "arrival_delay_error_s": delay_error_s,
            "exponent_tolerance": exponent_tolerance,
            "delay_tolerance_s": delay_tolerance_s,
        },
        detail={
            "flash_to_boom_s": arrival.propagation_delay_s,
            "sound_pressure_level_db": arrival.sound_pressure_level_db,
            "distance_m": distance_m,
            "sound_speed_mps": sound_speed_mps,
            "effective_speed_mps": effective_speed_mps,
            "transition_time_s": transition_time_s,
            "transition_radius_m": transition_radius_m,
            "supersonic_before_transition": supersonic_before,
            "subsonic_after_transition": subsonic_after,
            "note": (
                "Model self-consistency only. Comparing the delay against a "
                "real recording requires a timestamped reference video."
            ),
        },
    )


# ---------------------------------------------------------------------------
# V-19 Rayleigh optical depth
# ---------------------------------------------------------------------------


# Tabulated Rayleigh optical depths at standard sea-level pressure, from the
# Bodhaine et al. (1999) formulation these values are computed to reproduce.
PUBLISHED_RAYLEIGH_OPTICAL_DEPTH: dict[float, float] = {
    400.0: 0.3602,
    450.0: 0.2211,
    500.0: 0.1434,
    550.0: 0.0971,
    600.0: 0.0681,
    650.0: 0.0492,
    700.0: 0.0364,
}


def rayleigh_optical_depth_check() -> MetricResult:
    """Check molecular scattering against published optical depths."""

    worst_relative_error = 0.0
    worst_wavelength = 0.0
    computed: dict[str, float] = {}
    for wavelength_nm, published in PUBLISHED_RAYLEIGH_OPTICAL_DEPTH.items():
        value = float(atmosphere.rayleigh_optical_depth(wavelength_nm))
        computed[f"{wavelength_nm:.0f}nm"] = value
        error = abs(value - published) / published
        if error > worst_relative_error:
            worst_relative_error = error
            worst_wavelength = wavelength_nm

    # Rayleigh scattering follows an inverse fourth power of wavelength, with a
    # small excess from the dispersion of air's refractive index.
    ratio = float(
        atmosphere.rayleigh_optical_depth(400.0)
        / atmosphere.rayleigh_optical_depth(800.0)
    )
    power_law_ok = 16.0 <= ratio <= 17.5

    # The optical depth is proportional to the molecular column, so halving the
    # station pressure must halve it exactly.
    half_pressure = float(
        atmosphere.rayleigh_optical_depth(
            550.0, atmosphere.STANDARD_PRESSURE_PA * 0.5
        )
    )
    pressure_error = abs(
        half_pressure / float(atmosphere.rayleigh_optical_depth(550.0)) - 0.5
    )

    tolerance = 1.0e-2
    passed = (
        worst_relative_error < tolerance
        and power_law_ok
        and pressure_error < 1e-12
    )
    return MetricResult(
        spec=catalogue.V19,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"worst relative error {worst_relative_error:.2e} at "
            f"{worst_wavelength:.0f} nm; 400/800 nm ratio {ratio:.3f}"
        ),
        residuals={
            "worst_relative_error": worst_relative_error,
            "worst_wavelength_nm": worst_wavelength,
            "inverse_fourth_power_ratio": ratio,
            "pressure_scaling_error": pressure_error,
            "tolerance": tolerance,
        },
        detail={
            "computed_optical_depth": computed,
            "published_optical_depth": {
                f"{key:.0f}nm": value
                for key, value in PUBLISHED_RAYLEIGH_OPTICAL_DEPTH.items()
            },
            "reference": (
                "Bodhaine, Wood, Dutton and Slusser, J. Atmos. Oceanic "
                "Technol. 16(11), 1999"
            ),
            "aerosol_and_ozone": (
                "Aerosol turbidity is a grade C urban estimate and ozone is "
                "grade U; neither is checked here because neither is measured "
                "for the event."
            ),
        },
    )


# ---------------------------------------------------------------------------
# V-20 star catalogue astrometry
# ---------------------------------------------------------------------------


def star_catalogue_astrometry(
    scenario: Scenario, catalogue_path: Path | None = None
) -> MetricResult:
    """Compare the vectorised catalogue transform against the library path."""

    star_catalogue = (
        StarCatalogue.load_if_present(catalogue_path)
        if catalogue_path is not None
        else StarCatalogue.load_if_present()
    )
    if star_catalogue is None or len(star_catalogue) == 0:
        return MetricResult(
            spec=catalogue.V20,
            status=MetricStatus.NO_REFERENCE,
            message=(
                "no star catalogue imported; run "
                "'python -m tools.import_star_catalogue' to enable this metric"
            ),
            detail={
                "required_dataset": catalogue.DATASET_STAR_CATALOGUE,
                "importer": "tools/import_star_catalogue.py",
                "fallback": (
                    "The renderer uses the procedural grade-D field until a "
                    "catalogue is present."
                ),
            },
        )

    observer_record = scenario.default_observer
    latitude = observer_record.position.latitude_deg
    longitude = observer_record.position.longitude_deg
    elevation = observer_record.position.altitude_m
    moment = scenario.reference_epoch
    time = _astronomy_time(moment)
    observer = astronomy.Observer(latitude, longitude, elevation)

    azimuth, altitude = star_catalogue.apparent_horizontal(
        moment, latitude, longitude, elevation, astronomy.Refraction.Airless
    )
    right_ascension, declination = star_catalogue.positions_at(moment)

    # Astronomy Engine supports eight user-defined stars, so the comparison
    # uses a fixed deterministic sample rather than the whole catalogue.
    generator = np.random.default_rng(scenario.seeds.derive("star_catalogue"))
    sample = generator.choice(len(star_catalogue), 8, replace=False)
    worst_arcsec = 0.0
    without_aberration_arcsec = 0.0
    azimuth_airless, altitude_airless = star_catalogue.apparent_horizontal(
        moment,
        latitude,
        longitude,
        elevation,
        astronomy.Refraction.Airless,
        aberration=False,
    )
    for slot, index in enumerate(sample):
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
        cosine = math.cos(math.radians(horizon.altitude))
        worst_arcsec = max(
            worst_arcsec,
            abs(horizon.altitude - altitude[index]) * 3_600.0,
            _angle_difference_deg(horizon.azimuth, azimuth[index])
            * cosine
            * 3_600.0,
        )
        without_aberration_arcsec = max(
            without_aberration_arcsec,
            abs(horizon.altitude - altitude_airless[index]) * 3_600.0,
            _angle_difference_deg(horizon.azimuth, azimuth_airless[index])
            * cosine
            * 3_600.0,
        )

    above_horizon = int(np.count_nonzero(altitude > 0.0))
    tolerance_arcsec = 0.5
    passed = (
        worst_arcsec < tolerance_arcsec and star_catalogue.is_measured
    )
    return MetricResult(
        spec=catalogue.V20,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"{len(star_catalogue)} stars from {star_catalogue.source_id}; "
            f"worst residual {worst_arcsec:.4f} arcsec over 8 sampled stars"
        ),
        residuals={
            "worst_residual_arcsec": worst_arcsec,
            "residual_without_aberration_arcsec": without_aberration_arcsec,
            "tolerance_arcsec": tolerance_arcsec,
            "stars": float(len(star_catalogue)),
            "stars_above_horizon": float(above_horizon),
        },
        detail={
            **star_catalogue.summary(),
            "epoch_utc": moment.isoformat(),
            "observer_id": observer_record.observer_id,
            "aberration_note": (
                "Annual aberration is applied from the Earth's barycentric "
                "velocity. Omitting it moves the comparison to about 18 "
                "arcseconds, which is the classical aberration constant."
            ),
        },
    )


# ---------------------------------------------------------------------------
# V-18 shell library integrity
# ---------------------------------------------------------------------------


REQUIRED_BREAK_PATTERNS: frozenset[BreakPattern] = frozenset(BreakPattern)
"""Every pattern the product requirements list must be represented."""


def _simulate_profile(
    profile: ShellProfile, seed: int
) -> tuple[float, int, bool]:
    """Fly one shell to completion and total the smoke mass it emitted."""

    world = FireworkWorld(
        AtmosphereConfig(
            wind_velocity_mps=(0.0, 0.0, 0.0),
            wind_velocity_100m_mps=(0.0, 0.0, 0.0),
        ),
        profile,
        120_000,
        seed,
    )
    world.launch()
    emitted_kg = 0.0
    peak_stars = 0
    finite = True
    horizon_s = (
        profile.fuse_delay_s + profile.star_lifetime_mean_s * 3.0 + 2.0
    )
    for step in range(int(horizon_s * 120)):
        world.update(1.0 / 120.0)
        peak_stars = max(peak_stars, world.stars.count)
        if (step + 1) % 4 == 0:
            for emission in world.consume_combustion_emissions():
                emitted_kg += float(emission.smoke_mass_kg.sum(dtype=np.float64))
    count = world.stars.count
    if count:
        finite = bool(
            np.isfinite(world.stars.position_m[:count]).all()
            and np.isfinite(world.stars.velocity_mps[:count]).all()
        )
    return emitted_kg, peak_stars, finite


def shell_library_integrity(
    library: ShellLibrary = SHELL_LIBRARY, seed: int = 13
) -> MetricResult:
    """Fly every profile in the library and check its declared budget."""

    missing_patterns = sorted(
        pattern.value
        for pattern in REQUIRED_BREAK_PATTERNS - library.patterns()
    )
    worst_mass_overrun = 0.0
    worst_profile = ""
    non_finite: list[str] = []
    silent: list[str] = []
    ungraded: list[str] = []
    per_profile: dict[str, dict[str, float]] = {}

    for profile in library:
        emitted_kg, peak_stars, finite = _simulate_profile(profile, seed)
        budget_kg = (
            profile.total_composition_mass_kg()
            * profile.star_smoke_yield_fraction
        )
        overrun = (
            (emitted_kg - budget_kg) / budget_kg if budget_kg > 0.0 else 0.0
        )
        if overrun > worst_mass_overrun:
            worst_mass_overrun = overrun
            worst_profile = profile.profile_id
        if not finite:
            non_finite.append(profile.profile_id)
        if peak_stars == 0:
            silent.append(profile.profile_id)
        if profile.confidence_grade.is_evidence:
            # No measured shell record exists, so nothing in the library may
            # claim to be evidence.
            ungraded.append(profile.profile_id)
        per_profile[profile.profile_id] = {
            "emitted_smoke_kg": emitted_kg,
            "declared_budget_kg": budget_kg,
            "relative_overrun": overrun,
            "peak_stars": float(peak_stars),
        }

    # Determinism: one profile with a secondary break exercises the deepest
    # spawn path, so it is the strongest single determinism probe available.
    probe = next(
        (profile for profile in library if profile.secondary is not None),
        next(iter(library)),
    )
    first = _simulate_profile(probe, seed)
    second = _simulate_profile(probe, seed)
    deterministic = first == second

    tolerance = 1.0e-3
    passed = (
        not missing_patterns
        and not non_finite
        and not silent
        and not ungraded
        and deterministic
        and worst_mass_overrun <= tolerance
    )
    return MetricResult(
        spec=catalogue.V18,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"{len(library)} profiles flown across "
            f"{len(library.patterns())} break patterns; worst mass overrun "
            f"{worst_mass_overrun:.2e}"
            + (f" ({worst_profile})" if worst_profile else "")
        ),
        residuals={
            "profiles": float(len(library)),
            "patterns": float(len(library.patterns())),
            "worst_mass_overrun": worst_mass_overrun,
            "tolerance": tolerance,
        },
        detail={
            "missing_patterns": missing_patterns,
            "non_finite_profiles": non_finite,
            "profiles_that_produced_no_stars": silent,
            "profiles_claiming_evidence_grade": ungraded,
            "deterministic": deterministic,
            "determinism_probe": probe.profile_id,
            "per_profile": per_profile,
            "note": (
                "Every shipped profile is confidence grade D. These are "
                "archetypes reproducing each named effect's documented visual "
                "behaviour, not measured records of any shell that was fired."
            ),
        },
    )


# ---------------------------------------------------------------------------
# V-13 memory footprint
# ---------------------------------------------------------------------------


def _array_bytes(container: Any) -> int:
    """Sum the NumPy allocations an object holds.

    Handles both ordinary objects and ``slots=True`` dataclasses such as
    :class:`~simulator.scene.StaticScene`, which have no ``__dict__``.
    """

    names = getattr(container, "__slots__", None)
    if names is None:
        values = vars(container).values()
    else:
        values = (
            getattr(container, name)
            for name in names
            if hasattr(container, name)
        )
    return sum(
        int(value.nbytes) for value in values if isinstance(value, np.ndarray)
    )


def cpu_state_footprint(
    scenario: Scenario, config: SimulationConfig | None = None
) -> MetricResult:
    """Sum the simulator's CPU-side state allocations exactly."""

    config = config or SimulationConfig()
    world = FireworkWorld(
        AtmosphereConfig(),
        config.shell,
        config.render.max_particles,
        scenario.seeds.derive("shell_burst"),
    )
    star_bytes = _array_bytes(world.stars)
    fluid = SmokeFluid2D(config.smoke, AtmosphereConfig())
    fluid_bytes = _array_bytes(fluid)

    scene_bytes = 0
    scene_detail: dict[str, Any] = {"status": "not loaded"}
    scene_path = (
        Path(__file__).resolve().parent.parent.parent
        / "assets"
        / "yeouido_scene.npz"
    )
    if scene_path.exists():
        from ..scene import load_scene

        scene = load_scene(scene_path)
        scene_bytes = _array_bytes(scene)
        scene_detail = {
            "status": "loaded",
            "building_vertices": int(scene.building_vertices.shape[0]),
            "detail_vertices": int(scene.detail_vertices.shape[0]),
            "road_vertices": int(scene.road_vertices.shape[0]),
        }

    total_bytes = star_bytes + fluid_bytes + scene_bytes
    to_mib = 1.0 / (1024.0 * 1024.0)
    return MetricResult(
        spec=catalogue.V13,
        status=MetricStatus.REPORTED,
        message=(
            f"{total_bytes * to_mib:.1f} MiB of CPU simulation state "
            f"({config.render.max_particles:,} particle capacity)"
        ),
        residuals={
            "star_field_mib": star_bytes * to_mib,
            "plume_solver_mib": fluid_bytes * to_mib,
            "static_scene_mib": scene_bytes * to_mib,
            "total_mib": total_bytes * to_mib,
        },
        detail={
            "max_particles": config.render.max_particles,
            "smoke_grid": list(config.smoke.grid_size),
            "static_scene": scene_detail,
            "note": (
                "Exact array allocations, not peak process RSS. Driver-side "
                "GPU allocations are not counted here."
            ),
        },
    )
