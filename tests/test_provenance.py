import json
from pathlib import Path

import pytest

from simulator.provenance import (
    ConfidenceGrade,
    DataRecord,
    Provenance,
    require_aware_timestamp,
)

ROOT = Path(__file__).resolve().parent.parent


def test_grade_ordering_places_measurement_above_art() -> None:
    assert ConfidenceGrade.MEASURED.rank < ConfidenceGrade.RECONSTRUCTED.rank
    assert ConfidenceGrade.RECONSTRUCTED.rank < ConfidenceGrade.MODELLED.rank
    assert ConfidenceGrade.MODELLED.rank < ConfidenceGrade.ARTISTIC.rank
    assert ConfidenceGrade.ARTISTIC.rank < ConfidenceGrade.UNVERIFIED.rank


def test_only_measured_and_reconstructed_count_as_evidence() -> None:
    assert ConfidenceGrade.MEASURED.is_evidence
    assert ConfidenceGrade.RECONSTRUCTED.is_evidence
    assert not ConfidenceGrade.MODELLED.is_evidence
    assert not ConfidenceGrade.ARTISTIC.is_evidence
    assert not ConfidenceGrade.UNVERIFIED.is_evidence


def test_evidence_grade_requires_a_named_source() -> None:
    with pytest.raises(ValueError, match="requires source_id or source_url"):
        DataRecord(confidence_grade=ConfidenceGrade.MEASURED)
    # A modelled or artistic value may legitimately have no external source.
    DataRecord(confidence_grade=ConfidenceGrade.MODELLED)


def test_naive_timestamps_are_rejected() -> None:
    # An offset-less string would be reinterpreted in the host machine's local
    # timezone, shifting an Asia/Seoul observation by up to nine hours while
    # still producing a plausible number.
    with pytest.raises(ValueError, match="no UTC offset"):
        require_aware_timestamp("2024-10-05T19:20:00", "captured_at")
    with pytest.raises(ValueError, match="no UTC offset"):
        DataRecord(
            confidence_grade=ConfidenceGrade.MODELLED,
            captured_at="2024-10-05T19:20:00",
        )
    aware = require_aware_timestamp("2024-10-05T19:20:00+09:00", "captured_at")
    assert aware.utcoffset() is not None


def test_record_round_trips_through_a_dict() -> None:
    record = DataRecord(
        confidence_grade=ConfidenceGrade.MEASURED,
        source_id="meteostat-47108-2024",
        units="degC, hPa, m/s",
        captured_at="2026-07-29T00:00:00+09:00",
    )
    restored = DataRecord.from_dict(record.to_dict())
    assert restored == record


def test_unknown_fields_and_grades_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown provenance fields"):
        DataRecord.from_dict({"confidence_grade": "C", "confidance": "typo"})
    with pytest.raises(ValueError, match="unknown confidence_grade"):
        DataRecord.from_dict({"confidence_grade": "AA"})


def test_lookup_falls_back_to_dotted_ancestors() -> None:
    provenance = Provenance(
        {"observers": DataRecord(confidence_grade=ConfidenceGrade.MODELLED)}
    )
    assert (
        provenance.grade_for("observers.origin_reference.position")
        is ConfidenceGrade.MODELLED
    )
    assert provenance.grade_for("events") is ConfidenceGrade.UNVERIFIED


def test_worst_grade_prevents_a_derived_value_outranking_its_inputs() -> None:
    provenance = Provenance(
        {
            "weather": DataRecord(
                confidence_grade=ConfidenceGrade.MEASURED, source_id="isd_lite"
            ),
            "wind_profile": DataRecord(confidence_grade=ConfidenceGrade.MODELLED),
            "launch_sites": DataRecord(confidence_grade=ConfidenceGrade.UNVERIFIED),
        }
    )
    assert (
        provenance.worst_grade(["weather", "wind_profile"])
        is ConfidenceGrade.MODELLED
    )
    assert (
        provenance.worst_grade(["weather", "wind_profile", "launch_sites"])
        is ConfidenceGrade.UNVERIFIED
    )
    assert provenance.worst_grade([]) is ConfidenceGrade.UNVERIFIED


def test_summary_and_grade_listing() -> None:
    provenance = Provenance(
        {
            "a": DataRecord(
                confidence_grade=ConfidenceGrade.MEASURED, source_id="x"
            ),
            "b": DataRecord(confidence_grade=ConfidenceGrade.MODELLED),
            "c": DataRecord(confidence_grade=ConfidenceGrade.UNVERIFIED),
        }
    )
    assert provenance.summary() == {"A": 1, "B": 0, "C": 1, "D": 0, "U": 1}
    assert provenance.paths_with_grade(ConfidenceGrade.UNVERIFIED) == ["c"]


def test_refined_terrain_provenance_matches_the_shipped_asset() -> None:
    metadata = json.loads(
        (ROOT / "assets" / "yeouido_terrain_2023_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    from simulator.provenance import file_checksum

    assert metadata["derived_asset_sha256"] == file_checksum(
        ROOT / "assets" / "yeouido_scene.npz"
    ).removeprefix("sha256:")
    assert metadata["official_support_fraction"] == 1.0
    assert metadata["constraint_fit"]["rmse_m"] < 2.5


def test_event_water_datum_is_gauge_zero_plus_observed_stage() -> None:
    water = json.loads(
        (ROOT / "assets" / "yeouido_2024-10-05_water_level.json").read_text(
            encoding="utf-8"
        )
    )
    assert water["reference_stage_m"] == water["hourly_stage_m"]["19"]
    assert water["reference_stage_m"] == water["hourly_stage_m"]["20"]
    assert water["reference_surface_elevation_el_m"] == pytest.approx(
        water["station"]["zero_elevation_el_m"] + water["reference_stage_m"]
    )
