from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest


REPORT = Path(
    "docs/validation/road_markings_v2/metric_road_marking_report.json"
)
CAPTURE = Path(
    "docs/validation/road_markings_v2/runtime_current/metric_lane_close.json"
)


def test_committed_metric_marking_report_locks_current_implementation() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["stage"] == "V2-2c"
    for source in report["implementation_sources"].values():
        path = Path(source["asset"])
        assert sha256(path.read_bytes()).hexdigest() == source["sha256"]
    binding = report["binding"]
    assert sha256(Path(binding["asset"]).read_bytes()).hexdigest() == binding["sha256"]


def test_metric_marking_contract_is_physical_and_evidence_gated() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    basis = report["official_dimension_basis"]
    contract = report["application_contract"]
    stats = contract["stats"]

    assert basis["legal_line_width_range_m"] == [0.15, 0.2]
    assert basis["selected_line_width_m"] == pytest.approx(0.15)
    assert basis["urban_dash_paint_m"] == pytest.approx(3.0)
    assert basis["urban_dash_gap_m"] == pytest.approx(3.0)
    assert not basis["site_surveyed"]
    assert contract["required_tags"] == {
        "oneway": "yes",
        "lanes": "integer >= 2",
    }
    assert stats["explicit_oneway_lane_way_count"] == 66
    assert stats["marked_segment_count"] == 4_387
    assert stats["marked_lane_divider_count"] == 9_933
    assert stats["suppressed_asphalt_segment_count"] == 21_229
    assert stats["marked_centreline_length_m"] == pytest.approx(48_567.0265)


def test_unsupported_marking_types_and_inferred_cycle_bands_stay_blocked() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    gates = report["gates"]

    assert gates["metric_line_width"]
    assert gates["generic_all_asphalt_paint_removed"]
    assert gates["unsupported_cycleway_transverse_pattern_removed"]
    assert not gates["bidirectional_centre_line_application_allowed"]
    assert not gates["edge_line_application_allowed"]
    assert not gates["turn_arrow_application_allowed"]
    assert not gates["bicycle_symbol_application_allowed"]


def test_runtime_close_capture_matches_its_committed_image() -> None:
    capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
    image_path = CAPTURE.parent / capture["sdr"]["asset"]

    assert capture["view_id"] == "metric_lane_close"
    assert not capture["synthetic_firework_present"]
    assert sha256(image_path.read_bytes()).hexdigest() == capture["sdr"]["sha256"]
    assert capture["sdr"]["statistics"]["clipped_white_fraction"] == 0.0
