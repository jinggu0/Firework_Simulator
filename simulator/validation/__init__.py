"""Validation harness for the historical reconstruction.

The governing rule: a metric with no reference data reports ``NO_REFERENCE``,
never ``PASS``. Absence of evidence is not a green check, and only ``FAIL`` or
``ERROR`` may fail a run.

Metrics are declared in :mod:`~simulator.validation.catalogue` and implemented
in :mod:`~simulator.validation.metrics`. Everything except V-08 and V-12 runs
without an OpenGL context, so the harness is usable on a headless agent.
"""

from .catalogue import BY_ID, CATALOGUE
from .report import MetricResult, MetricSpec, MetricStatus, ValidationReport
from .runner import DEFAULT_EPHEMERIS_PATH, run_validation

__all__ = [
    "BY_ID",
    "CATALOGUE",
    "DEFAULT_EPHEMERIS_PATH",
    "MetricResult",
    "MetricSpec",
    "MetricStatus",
    "ValidationReport",
    "run_validation",
]
