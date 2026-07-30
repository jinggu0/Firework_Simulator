"""Machine-readable source grading for every historical or environmental datum.

The repository previously recorded data confidence only as prose in
``assets/ATTRIBUTION.md`` and ``docs/ARCHITECTURE.md``. Prose cannot be queried
at runtime, so a renderer could not distinguish a surveyed building height from
a 12 m fallback. This module makes the grade a value that travels with the datum.

Grades follow the project fidelity contract:

======  =====================================================  ================
Grade   Meaning                                                 Permitted claim
======  =====================================================  ================
``A``   Official observation or original data                   "measured"
``B``   Reconstructed from video, photogrammetry, or geometry   "reconstructed"
``C``   Estimated with an accepted physical/statistical model    "modelled"
``D``   Artistic completion for visual continuity                not evidence
``U``   Unverified / not yet obtained                            none
======  =====================================================  ================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping


class ConfidenceGrade(Enum):
    """Ordered source-confidence grade. Lower ``rank`` is stronger evidence."""

    MEASURED = "A"
    RECONSTRUCTED = "B"
    MODELLED = "C"
    ARTISTIC = "D"
    UNVERIFIED = "U"

    @property
    def rank(self) -> int:
        return _GRADE_RANK[self.value]

    @property
    def is_evidence(self) -> bool:
        """True only for grades that may be described as evidence."""

        return self in (ConfidenceGrade.MEASURED, ConfidenceGrade.RECONSTRUCTED)


_GRADE_RANK: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3, "U": 4}

_RECORD_FIELDS: tuple[str, ...] = (
    "source_id",
    "source_url",
    "license",
    "captured_at",
    "valid_from",
    "valid_to",
    "coordinate_reference_system",
    "units",
    "confidence_grade",
    "uncertainty",
    "checksum",
    "notes",
)

_TIMESTAMP_FIELDS: tuple[str, ...] = ("captured_at", "valid_from", "valid_to")


def require_aware_timestamp(text: str, field_name: str) -> datetime:
    """Parse an ISO-8601 timestamp, rejecting offset-less input.

    ``datetime.fromisoformat`` accepts naive strings and downstream
    ``.timestamp()`` then silently applies the *host machine's* local timezone.
    On a UTC build agent that shifts an ``Asia/Seoul`` observation by nine
    hours while still producing a plausible number, so the failure is not
    detectable by inspecting the result. Rejecting the input is the only safe
    behaviour.
    """

    try:
        parsed = datetime.fromisoformat(text)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name}: not an ISO-8601 timestamp: {text!r}"
        ) from error
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise ValueError(
            f"{field_name}: timestamp {text!r} has no UTC offset. "
            "Naive timestamps are rejected because they would be interpreted "
            "in the host machine's local timezone."
        )
    return parsed


def file_checksum(path: Path) -> str:
    """Return the ``sha256:`` checksum string used by provenance records."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class DataRecord:
    """Provenance for one datum or one file.

    Every field is optional except ``confidence_grade`` so that partial
    knowledge is representable, but grades ``A`` and ``B`` additionally require
    an identified source: claiming measurement without naming what measured it
    is precisely the failure this class exists to prevent.
    """

    confidence_grade: ConfidenceGrade
    source_id: str = ""
    source_url: str = ""
    license: str = ""
    captured_at: str = ""
    valid_from: str = ""
    valid_to: str = ""
    coordinate_reference_system: str = ""
    units: str = ""
    uncertainty: str = ""
    checksum: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.confidence_grade.is_evidence and not (
            self.source_id or self.source_url
        ):
            raise ValueError(
                f"grade {self.confidence_grade.value} requires source_id or "
                "source_url; an evidence claim must name its source"
            )
        for name in _TIMESTAMP_FIELDS:
            value = getattr(self, name)
            if value:
                require_aware_timestamp(value, name)

    @property
    def grade(self) -> ConfidenceGrade:
        return self.confidence_grade

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataRecord":
        unknown = set(data) - set(_RECORD_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown provenance fields: {sorted(unknown)}"
            )
        raw_grade = data.get("confidence_grade", "U")
        try:
            grade = ConfidenceGrade(raw_grade)
        except ValueError as error:
            raise ValueError(
                f"unknown confidence_grade {raw_grade!r}; "
                f"expected one of {sorted(_GRADE_RANK)}"
            ) from error
        payload = {
            name: str(data.get(name, ""))
            for name in _RECORD_FIELDS
            if name != "confidence_grade"
        }
        return cls(confidence_grade=grade, **payload)

    def to_dict(self) -> dict[str, str]:
        result = {
            name: getattr(self, name)
            for name in _RECORD_FIELDS
            if name != "confidence_grade"
        }
        result["confidence_grade"] = self.confidence_grade.value
        return result


@dataclass(frozen=True, slots=True)
class Provenance:
    """Lookup from a dotted field path to the record that justifies it."""

    records: dict[str, DataRecord] = field(default_factory=dict)

    def record_for(self, path: str) -> DataRecord | None:
        """Return the record for ``path``, falling back to dotted ancestors.

        ``observers.origin_reference.position`` resolves against
        ``observers.origin_reference`` and then ``observers`` so a single
        record can cover a whole subtree.
        """

        parts = path.split(".")
        for depth in range(len(parts), 0, -1):
            candidate = ".".join(parts[:depth])
            if candidate in self.records:
                return self.records[candidate]
        return None

    def grade_for(self, path: str) -> ConfidenceGrade:
        record = self.record_for(path)
        return record.grade if record else ConfidenceGrade.UNVERIFIED

    def worst_grade(self, paths: Iterable[str]) -> ConfidenceGrade:
        """Return the weakest grade among ``paths``.

        A derived quantity can never be stronger than its weakest input, so
        this is how a reconstructed value is prevented from inheriting the
        confidence of the one measured term it happens to contain.
        """

        grades = [self.grade_for(path) for path in paths]
        if not grades:
            return ConfidenceGrade.UNVERIFIED
        return max(grades, key=lambda grade: grade.rank)

    def paths_with_grade(self, grade: ConfidenceGrade) -> list[str]:
        return sorted(
            path
            for path, record in self.records.items()
            if record.grade is grade
        )

    def summary(self) -> dict[str, int]:
        """Count of records per grade, for reporting and regression checks."""

        counts = {grade.value: 0 for grade in ConfidenceGrade}
        for record in self.records.values():
            counts[record.grade.value] += 1
        return counts

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Provenance":
        return cls(
            {
                str(path): DataRecord.from_dict(record)
                for path, record in data.items()
            }
        )

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {path: record.to_dict() for path, record in self.records.items()}
