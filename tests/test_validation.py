import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.scenario import load_default_scenario
from simulator.validation import (
    BY_ID,
    CATALOGUE,
    MetricStatus,
    ValidationReport,
    run_validation,
)
from simulator.validation import capture, catalogue, metrics, performance
from simulator.validation.report import MetricResult
from simulator.validation.runner import _guarded


@pytest.fixture(scope="module")
def scenario():
    return load_default_scenario()


@pytest.fixture(scope="module")
def report(scenario) -> ValidationReport:
    # 400 steps at 120 Hz clears the 3.05 s fuse delay, so the replay actually
    # compares a populated star field rather than two empty arrays.
    return run_validation(scenario, replay_steps=400)


# --- report structure ------------------------------------------------------


def test_every_catalogue_metric_appears_exactly_once(report) -> None:
    # A shorter report would make missing coverage look like passing coverage.
    reported = [result.metric_id for result in report]
    assert reported == [spec.metric_id for spec in CATALOGUE]
    assert len(set(reported)) == len(reported)


def test_every_metric_declares_a_physical_basis() -> None:
    # A tolerance without a recorded reason is a guess that will later be
    # loosened to make a test pass.
    for spec in CATALOGUE:
        assert spec.tolerance.strip()
        assert len(spec.physical_basis.strip()) > 40


def test_dataset_dependent_metrics_either_pass_or_declare_absence(report) -> None:
    # A metric that needs external data must resolve to exactly one of two
    # states: it read the data and reached a verdict, or it reported the data
    # missing. Anything else is a silent skip.
    allowed = {
        MetricStatus.PASS,
        MetricStatus.FAIL,
        MetricStatus.NO_REFERENCE,
    }
    for result in report:
        if result.spec.required_datasets:
            assert result.status in allowed, result.metric_id


def test_a_passing_dataset_metric_must_name_the_data_it_read(report) -> None:
    # Guards the case this suite previously assumed away: once a required
    # dataset is actually present, a pass must be backed by an identified
    # source rather than by the metric quietly finding nothing to check.
    for result in report:
        if result.status is MetricStatus.PASS and result.spec.required_datasets:
            assert result.detail.get("source_id") or result.detail.get(
                "source_url"
            ), result.metric_id


def test_metrics_without_data_never_report_pass(report) -> None:
    for result in report:
        if result.status is MetricStatus.NO_REFERENCE:
            assert result.status is not MetricStatus.PASS
            assert result.residuals == {} or "tolerance" in str(
                result.residuals
            ), result.metric_id


def test_unrunnable_metrics_do_not_fail_the_run(report) -> None:
    counts = report.counts()
    assert counts["NO_REFERENCE"] > 0
    assert report.exit_code == 0
    assert report.failures == ()


def test_report_serialises_to_json(report) -> None:
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert payload["scenario_id"] == "yeouido-2024-10-05"
    assert payload["exit_code"] == 0
    assert len(payload["metrics"]) == len(CATALOGUE)
    assert payload["metrics"][0]["metric_id"] == "V-01"


def test_result_lookup_and_unknown_id(report) -> None:
    assert report.result("V-03").metric_id == "V-03"
    with pytest.raises(KeyError, match="unknown metric"):
        report.result("V-99")


def test_a_raising_metric_becomes_error_not_a_skip() -> None:
    def explode() -> MetricResult:
        raise RuntimeError("solver diverged")

    result = _guarded(BY_ID["V-14"], explode)
    assert result.status is MetricStatus.ERROR
    assert "solver diverged" in result.message
    assert result.status.is_failure
    assert ValidationReport(results=(result,)).exit_code == 1


# --- individual metrics ----------------------------------------------------


def test_geodetic_round_trip_passes_well_inside_tolerance(report) -> None:
    result = report.result("V-03")
    assert result.status is MetricStatus.PASS
    # Measured at about 2e-9 m; the 1 mm gate leaves five orders of margin.
    assert result.residuals["worst_residual_m"] < 1e-6
    assert result.detail["sample_count"] > 400


def test_deterministic_replay_is_bit_exact(report) -> None:
    result = report.result("V-02")
    assert result.status is MetricStatus.PASS
    assert result.detail["clock_identical"] is True
    assert result.detail["pcm_identical"] is True
    assert result.detail["comparison_exercised"] is True
    assert result.detail["star_count"] > 0
    assert result.detail["burst_count"] > 0
    for name in (
        "world_position_m_max_abs_delta",
        "world_velocity_mps_max_abs_delta",
        "world_age_s_max_abs_delta",
        "world_burst_positions_m_max_abs_delta",
        "acoustic_delay_delta_s",
        "acoustic_level_delta_db",
    ):
        assert result.residuals[name] == 0.0, f"{name} drifted"


def test_replay_metric_fails_when_no_burst_was_reached(scenario) -> None:
    # Two empty star fields compare equal, so the metric must refuse to claim
    # determinism from a run that never reached the fuse delay.
    result = metrics.deterministic_replay(scenario, steps=100)
    assert result.status is MetricStatus.FAIL
    assert result.residuals["stars_compared"] == 0.0
    assert "no burst" in result.message


def test_replay_actually_advances_state(scenario) -> None:
    # Guards against the comparison being trivially satisfied: one more step
    # must change the star field.
    from simulator.validation.metrics import _replay_world

    baseline = _replay_world(scenario, 400)
    assert baseline["position_m"].size > 0
    assert not np.array_equal(
        baseline["position_m"], _replay_world(scenario, 401)["position_m"]
    )


def test_horizontal_transform_cross_check_agrees_to_double_precision(
    report,
) -> None:
    result = report.result("V-15")
    assert result.status is MetricStatus.PASS
    # Both sides are pure rotations of the same RA/Dec, so agreement is limited
    # only by float64; measurement gives about 1e-13 deg.
    assert result.residuals["worst_azimuth_deg"] < 1e-11
    assert result.residuals["worst_altitude_deg"] < 1e-11
    assert result.detail["comparisons"] >= 8


def test_direction_vectors_round_trip_within_float32_precision(report) -> None:
    result = report.result("V-16")
    assert result.status is MetricStatus.PASS
    assert result.residuals["worst_unit_length_error"] < 1e-6
    assert result.detail["samples"] == 144


def test_conservation_closes_on_yields_energy_and_divergence(report) -> None:
    result = report.result("V-14")
    assert result.status is MetricStatus.PASS
    assert result.residuals["smoke_yield_relative_error"] < 2e-5
    assert result.residuals["thermal_yield_relative_error"] < 2e-5
    # The deterministic 0.97 +/- 0.03 combustion modulation leaves an
    # uncancelled residual of roughly 2.6e-4 over a 2.25 s burn.
    assert result.residuals["radiated_energy_relative_error"] < 1e-3
    assert (
        result.residuals["divergence_after_per_s"]
        < result.residuals["divergence_before_per_s"]
    )


def test_blast_model_reproduces_the_sedov_exponent(report) -> None:
    result = report.result("V-11")
    assert result.status is MetricStatus.PASS
    # Sedov-Taylor gives radius proportional to t^(2/5) exactly.
    assert abs(result.residuals["sedov_exponent"] - 0.4) < 5e-3
    assert result.residuals["arrival_delay_error_s"] < 1e-3
    assert result.detail["supersonic_before_transition"] is True
    assert result.detail["subsonic_after_transition"] is True
    assert result.detail["flash_to_boom_s"] > 0.0


def test_cpu_footprint_is_reported_not_gated(report) -> None:
    result = report.result("V-13")
    assert result.status is MetricStatus.REPORTED
    assert result.status is not MetricStatus.PASS
    assert result.residuals["total_mib"] > 0.0
    assert "not peak process RSS" in result.detail["note"]


def test_shell_library_integrity_passes(report) -> None:
    result = report.result("V-18")
    assert result.status is MetricStatus.PASS
    assert result.residuals["worst_mass_overrun"] < 1e-3
    assert result.detail["missing_patterns"] == []
    assert result.detail["non_finite_profiles"] == []
    assert result.detail["profiles_that_produced_no_stars"] == []
    assert result.detail["deterministic"] is True
    # Nothing in the library may claim to be evidence: no measured shell record
    # for the 2024 performance has been obtained.
    assert result.detail["profiles_claiming_evidence_grade"] == []


def test_shell_library_integrity_fails_on_missing_patterns() -> None:
    from simulator.shells import SHELL_LIBRARY as library
    from simulator.shells import ShellLibrary

    partial = ShellLibrary((library.get("peony_100mm_gold"),))
    result = metrics.shell_library_integrity(partial)
    assert result.status is MetricStatus.FAIL
    assert "ring" in result.detail["missing_patterns"]


def test_shell_library_integrity_fails_on_an_evidence_grade_claim() -> None:
    from dataclasses import replace

    from simulator.provenance import ConfidenceGrade
    from simulator.shells import SHELL_LIBRARY as library
    from simulator.shells import ShellLibrary

    overclaimed = replace(
        library.get("peony_100mm_gold"),
        confidence_grade=ConfidenceGrade.MEASURED,
    )
    result = metrics.shell_library_integrity(library.with_profile(overclaimed))
    assert result.status is MetricStatus.FAIL
    assert "peony_100mm_gold" in result.detail["profiles_claiming_evidence_grade"]
    assert isinstance(library, ShellLibrary)


def test_asset_checksums_match_the_shipped_files(report) -> None:
    result = report.result("V-21")
    assert result.status is MetricStatus.PASS
    assert result.residuals["checksums_verified"] >= 3
    assert result.residuals["checksums_mismatched"] == 0
    assert result.detail["mismatched"] == []
    assert result.detail["missing_files"] == []


def test_a_stale_checksum_fails(scenario) -> None:
    # This is the failure the metric exists for: regenerating a derived asset
    # leaves the declared checksum pointing at the previous file.
    from dataclasses import replace as dataclass_replace

    from simulator.provenance import Provenance

    record = scenario.provenance.record_for("scene.osm")
    stale = dataclass_replace(record, checksum="sha256:" + "0" * 64)
    corrupted = dataclass_replace(
        scenario,
        provenance=Provenance({**scenario.provenance.records, "scene.osm": stale}),
    )
    result = metrics.asset_checksum_integrity(corrupted)
    assert result.status is MetricStatus.FAIL
    assert result.residuals["checksums_mismatched"] == 1
    assert any("scene.osm" in entry for entry in result.detail["mismatched"])


def test_performance_metric_is_absent_unless_requested(report) -> None:
    # V-12 needs OpenGL, so a headless run must declare it rather than fail.
    assert report.result("V-12").status is MetricStatus.NOT_IMPLEMENTED


# --- V-01 external ephemeris ----------------------------------------------


def _write_ephemeris(path, offset_deg: float = 0.0):
    """Build a reference extract from the model, optionally perturbed.

    This verifies the comparison machinery, not astronomical accuracy: a
    self-generated reference cannot validate the ephemeris itself. Real
    validation needs an independent published extract, which is why the shipped
    state of V-01 is NO_REFERENCE.
    """

    import astronomy

    observer = astronomy.Observer(37.529, 126.935, 5.0)
    samples = []
    for iso in ("2024-10-05T19:20:00+09:00", "2024-10-05T20:30:00+09:00"):
        time = metrics._astronomy_time(
            __import__("datetime").datetime.fromisoformat(iso)
        )
        for name, body in (("sun", astronomy.Body.Sun), ("moon", astronomy.Body.Moon)):
            equatorial = astronomy.Equator(
                body, time, observer, ofdate=True, aberration=True
            )
            horizon = astronomy.Horizon(
                time,
                observer,
                equatorial.ra,
                equatorial.dec,
                astronomy.Refraction.Normal,
            )
            samples.append(
                {
                    "time": iso,
                    "body": name,
                    "azimuth_deg": horizon.azimuth + offset_deg,
                    "altitude_deg": horizon.altitude + offset_deg,
                }
            )
    path.write_text(
        json.dumps(
            {
                "source_id": "synthetic-fixture",
                "source_url": "",
                "refraction": "normal",
                "observer": {
                    "latitude_deg": 37.529,
                    "longitude_deg": 126.935,
                    "altitude_m": 5.0,
                },
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ephemeris_metric_reports_no_reference_when_absent(scenario, tmp_path) -> None:
    result = metrics.external_ephemeris_comparison(
        scenario, tmp_path / "absent.json"
    )
    assert result.status is MetricStatus.NO_REFERENCE
    assert result.status is not MetricStatus.PASS
    assert "required_format" in result.detail
    assert (
        result.detail["required_dataset"] == catalogue.DATASET_EXTERNAL_EPHEMERIS
    )


def test_ephemeris_metric_compares_when_a_reference_is_supplied(
    scenario, tmp_path
) -> None:
    path = _write_ephemeris(tmp_path / "ephemeris.json")
    result = metrics.external_ephemeris_comparison(scenario, path)
    assert result.status is MetricStatus.PASS
    assert result.detail["samples_compared"] == 4
    assert result.residuals["worst_azimuth_deg"] < 1e-9


def test_ephemeris_metric_fails_on_a_disagreeing_reference(
    scenario, tmp_path
) -> None:
    # 0.2 deg is four times the 0.05 deg gate.
    path = _write_ephemeris(tmp_path / "wrong.json", offset_deg=0.2)
    result = metrics.external_ephemeris_comparison(scenario, path)
    assert result.status is MetricStatus.FAIL
    assert result.residuals["worst_altitude_deg"] == pytest.approx(0.2, abs=1e-6)


def test_ephemeris_metric_rejects_naive_sample_times(scenario, tmp_path) -> None:
    path = tmp_path / "naive.json"
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "time": "2024-10-05T19:20:00",
                        "body": "sun",
                        "azimuth_deg": 275.13,
                        "altitude_deg": -14.28,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no UTC offset"):
        metrics.external_ephemeris_comparison(scenario, path)


# --- V-12 subprocess handling ---------------------------------------------


def test_frame_budget_gates_on_the_display_period(monkeypatch) -> None:
    payload = json.dumps(
        {"integrated": {"fluid_backend": "gpu_compute_mac_3d",
                        "frame_mean_ms": 9.0, "frame_p95_ms": 13.9,
                        "frame_p99_ms": 15.1, "physics_p95_ms": 5.8,
                        "visual_p95_ms": 9.5}}
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="pygame banner\n" + payload, stderr=""
        ),
    )
    result = performance.frame_budget(frames=8)
    assert result.status is MetricStatus.PASS
    assert result.residuals["margin_ms"] == pytest.approx(16.6667 - 13.9, abs=1e-3)
    # Machine context must travel with the number.
    assert result.detail["machine_specific"] is True
    assert result.detail["node"]


def test_frame_budget_fails_over_budget(monkeypatch) -> None:
    payload = json.dumps({"integrated": {"frame_p95_ms": 24.5}})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=payload, stderr=""),
    )
    assert performance.frame_budget(frames=8).status is MetricStatus.FAIL


def test_frame_budget_treats_a_missing_gl_context_as_no_reference(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="OpenGL 3.3 core context unavailable"
        ),
    )
    result = performance.frame_budget(frames=8)
    assert result.status is MetricStatus.NO_REFERENCE
    assert "OpenGL" in result.detail["stderr_tail"]


def test_frame_budget_reports_unparsable_output_as_error(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="no json here", stderr=""),
    )
    assert performance.frame_budget(frames=8).status is MetricStatus.ERROR


# --- linear HDR capture ----------------------------------------------------


def test_capture_requires_an_hdr_target() -> None:
    with pytest.raises(AttributeError, match="hdr_texture"):
        capture.read_linear_hdr(SimpleNamespace())


def test_capture_rejects_a_non_half_float_target() -> None:
    fake = SimpleNamespace(
        hdr_texture=SimpleNamespace(size=(4, 3), components=4, dtype="f4")
    )
    with pytest.raises(ValueError, match="half-float"):
        capture.read_linear_hdr(fake)


def test_capture_flips_opengl_row_order() -> None:
    height, width = 3, 4
    rows = np.zeros((height, width, 4), dtype=np.float16)
    rows[0] = 1.0  # OpenGL row 0 is the bottom of the image.
    fake = SimpleNamespace(
        hdr_texture=SimpleNamespace(
            size=(width, height),
            components=4,
            dtype="f2",
            read=lambda: rows.tobytes(),
        )
    )
    frame = capture.read_linear_hdr(fake)
    assert frame.shape == (height, width, 4)
    assert frame.dtype == np.float32
    assert np.all(frame[-1] == 1.0)
    assert np.all(frame[0] == 0.0)


def test_linear_hdr_statistics_do_not_apply_a_display_transform() -> None:
    frame = np.zeros((2, 2, 4), dtype=np.float32)
    frame[..., :3] = 8.0  # well above SDR white; must survive unclamped
    stats = capture.linear_hdr_statistics(frame)
    assert stats["maximum"] == 8.0
    assert stats["mean"] == 8.0
    assert stats["non_finite_fraction"] == 0.0


def test_saving_a_linear_frame_round_trips_without_quantisation(tmp_path) -> None:
    frame = np.random.default_rng(3).random((5, 7, 4)).astype(np.float32) * 120.0
    path = capture.save_linear_hdr(frame, tmp_path / "frame.png")
    # A PNG or JPEG would gamma-encode and quantise the values being compared.
    assert path.suffix == ".npy"
    np.testing.assert_array_equal(np.load(path), frame)
