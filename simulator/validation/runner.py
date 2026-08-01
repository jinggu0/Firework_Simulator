"""Assemble a complete validation report.

Every metric in the catalogue appears in the report exactly once. Metrics with
no implementation, or whose reference dataset is absent, are reported as such
rather than omitted: a shorter report would make missing coverage look like
passing coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..scenario import Scenario, load_default_scenario
from . import catalogue, metrics
from .report import MetricResult, MetricStatus, ValidationReport

DEFAULT_EPHEMERIS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "assets"
    / "reference_ephemeris_2024-10-05.json"
)


def _declared_result(spec) -> MetricResult:
    """Report a catalogue entry that has no runnable implementation yet."""

    if spec.required_datasets:
        return MetricResult(
            spec=spec,
            status=MetricStatus.NO_REFERENCE,
            message=(
                "required dataset(s) not present: "
                + ", ".join(spec.required_datasets)
            ),
            detail={"required_datasets": list(spec.required_datasets)},
        )
    return MetricResult(
        spec=spec,
        status=MetricStatus.NOT_IMPLEMENTED,
        message="declared in the catalogue; no implementation yet",
    )


def _guarded(
    spec, run: Callable[[], MetricResult]
) -> MetricResult:
    """Run a metric, converting an exception into ERROR rather than a crash.

    A metric that raises is a failure of the validation suite itself, so it must
    turn the run red rather than quietly disappear from the report.
    """

    try:
        return run()
    except Exception as error:  # noqa: BLE001 - deliberately broad
        return MetricResult(
            spec=spec,
            status=MetricStatus.ERROR,
            message=f"{type(error).__name__}: {error}",
        )


def run_validation(
    scenario: Scenario | None = None,
    *,
    ephemeris_path: Path | None = None,
    star_catalogue_path: Path | None = None,
    include_performance: bool = False,
    include_rendering: bool = False,
    performance_frames: int = 240,
    fluid_backend: str = "3d",
    replay_steps: int = 400,
) -> ValidationReport:
    """Run every runnable metric and declare the rest."""

    scenario = scenario or load_default_scenario()
    ephemeris = (
        ephemeris_path if ephemeris_path is not None else DEFAULT_EPHEMERIS_PATH
    )

    runnable: dict[str, Callable[[], MetricResult]] = {
        "V-01": lambda: metrics.external_ephemeris_comparison(scenario, ephemeris),
        "V-02": lambda: metrics.deterministic_replay(scenario, replay_steps),
        "V-03": lambda: metrics.geodetic_round_trip(scenario),
        "V-11": lambda: metrics.blast_propagation_model(scenario),
        "V-13": lambda: metrics.cpu_state_footprint(scenario),
        "V-14": lambda: metrics.physical_conservation(),
        "V-15": lambda: metrics.horizontal_transform_cross_check(scenario),
        "V-16": lambda: metrics.direction_vector_consistency(scenario),
        "V-18": lambda: metrics.shell_library_integrity(),
        "V-19": lambda: metrics.rayleigh_optical_depth_check(),
        "V-20": lambda: metrics.star_catalogue_astrometry(
            scenario, star_catalogue_path
        ),
        "V-21": lambda: metrics.asset_checksum_integrity(scenario),
    }
    if include_performance:
        from .performance import frame_budget

        runnable["V-12"] = lambda: frame_budget(
            performance_frames, fluid_backend
        )
    if include_rendering:
        # Separate from performance: V-12 is a machine-specific timing that
        # must not be compared across hardware, while V-22 compares a rendered
        # frame against a closed form and is portable.
        from .aerial_perspective import aerial_perspective
        from .display_transform import display_transform
        from .observer_transform import observer_transform

        runnable["V-22"] = aerial_perspective
        runnable["V-23"] = display_transform
        runnable["V-24"] = observer_transform

    results: list[MetricResult] = []
    for spec in catalogue.CATALOGUE:
        run = runnable.get(spec.metric_id)
        results.append(
            _guarded(spec, run) if run is not None else _declared_result(spec)
        )

    return ValidationReport(
        results=tuple(results),
        scenario_id=scenario.scenario_id,
        scenario_path=str(scenario.source_path or ""),
    )
