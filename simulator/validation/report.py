"""Data model for validation results.

The governing rule from the validation spec is encoded here: a metric with no
reference data reports ``NO_REFERENCE`` and never ``PASS``. Absence of evidence
is not a green check, and only ``FAIL`` or ``ERROR`` may fail a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MetricStatus(Enum):
    PASS = "PASS"
    """Reference exists and the residual is inside tolerance."""

    FAIL = "FAIL"
    """Reference exists and the residual is outside tolerance."""

    REPORTED = "REPORTED"
    """The metric ran and produced a value, but no gate is defined yet.

    Used where gating would measure the wrong thing — a pixel metric on a
    tone-mapped image, or a resource ceiling that has never been measured.
    """

    NO_REFERENCE = "NO_REFERENCE"
    """Implemented, but the dataset it compares against is not present."""

    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    """Declared in the catalogue; no implementation yet."""

    ERROR = "ERROR"
    """The metric raised. A broken check is a failure, not a skip."""

    @property
    def is_failure(self) -> bool:
        return self in (MetricStatus.FAIL, MetricStatus.ERROR)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A metric's declared identity, gate, and justification.

    ``physical_basis`` is mandatory. A tolerance without a recorded reason is a
    guess that will later be loosened to make a test pass.
    """

    metric_id: str
    title: str
    tolerance: str
    physical_basis: str
    required_datasets: tuple[str, ...] = ()
    requires_opengl: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "title": self.title,
            "tolerance": self.tolerance,
            "physical_basis": self.physical_basis,
            "required_datasets": list(self.required_datasets),
            "requires_opengl": self.requires_opengl,
        }


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One metric's outcome, with residuals reported rather than a bare boolean.

    A pass at 0.049 deg and a pass at 0.004 deg are different engineering
    situations, so the numbers travel with the verdict.
    """

    spec: MetricSpec
    status: MetricStatus
    message: str = ""
    residuals: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def metric_id(self) -> str:
        return self.spec.metric_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "status": self.status.value,
            "message": self.message,
            "residuals": dict(self.residuals),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    results: tuple[MetricResult, ...]
    scenario_id: str = ""
    scenario_path: str = ""
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def result(self, metric_id: str) -> MetricResult:
        for candidate in self.results:
            if candidate.metric_id == metric_id:
                return candidate
        raise KeyError(
            f"unknown metric {metric_id!r}; have "
            f"{[candidate.metric_id for candidate in self.results]}"
        )

    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in MetricStatus}
        for candidate in self.results:
            counts[candidate.status.value] += 1
        return counts

    @property
    def failures(self) -> tuple[MetricResult, ...]:
        return tuple(
            candidate for candidate in self.results if candidate.status.is_failure
        )

    @property
    def exit_code(self) -> int:
        """Non-zero only for FAIL or ERROR.

        NO_REFERENCE and NOT_IMPLEMENTED are honest states of the project, not
        regressions, so they must not turn a build red.
        """

        return 1 if self.failures else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "scenario_id": self.scenario_id,
            "scenario_path": self.scenario_path,
            "counts": self.counts(),
            "exit_code": self.exit_code,
            "failed_metrics": [result.metric_id for result in self.failures],
            "metrics": [result.to_dict() for result in self.results],
        }
