from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from tools.acquire_smap_ortho import coordinate_tile, tile_bounds, tile_url


REPORT = Path(
    "docs/validation/road_marking_registration_v2/"
    "smap_2024_registration_report.json"
)


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_smap_epsg5186_tile_math_matches_browser_observed_requests() -> None:
    assert tile_bounds(636, 520) == pytest.approx(
        (194_172.0, 547_224.0, 194_236.0, 547_288.0)
    )
    assert coordinate_tile(194_172.0, 547_224.0) == (636, 520)
    assert coordinate_tile(194_235.999, 547_287.999) == (636, 520)
    assert tile_url(636, 520) == (
        "https://smap1.eseoul.go.kr:5432/tile.sqlite/"
        "ortho_drone_25cm_2024/10/636/520.jpg"
    )
    assert "smap2.eseoul.go.kr" in tile_url(634, 520)
    assert "smap3.eseoul.go.kr" in tile_url(635, 520)


def test_committed_registration_report_locks_repository_inputs() -> None:
    report = _report()

    assert report["stage"] == "V2-2e"
    for source in report["inputs"].values():
        path = Path(source["asset"])
        assert sha256(path.read_bytes()).hexdigest() == source["sha256"]
    diagnostic = report["diagnostic"]
    assert sha256(Path(diagnostic["asset"]).read_bytes()).hexdigest() == diagnostic[
        "sha256"
    ]


def test_local_only_crop_is_native_grid_registered_without_redistributing_pixels() -> None:
    report = _report()
    source = report["source"]
    layer = report["layer_contract"]
    acquisition = report["local_acquisition"]
    registration = report["registration"]

    assert source["provider_attribution"] == "© 서울특별시 모든 권리 보유"
    assert not source["raw_tile_redistribution_authorised"]
    assert not source["raw_pixels_committed_to_repository"]
    assert layer["crs"] == "EPSG:5186"
    assert layer["native_resolution_m_per_pixel"] == pytest.approx(0.25)
    assert layer["tile_size_px"] == [256, 256]
    assert acquisition["tile_range"] == [634, 639, 520, 525]
    assert acquisition["tile_count"] == 36
    assert acquisition["ground_coverage_m"] == [384.0, 384.0]
    assert acquisition["mosaic_pixel_size"] == [1536, 1536]
    assert registration["grid_registration_residual_m"] == pytest.approx(0.0)
    assert registration["provider_grid_registration_passes"]
    assert not registration["independent_spatial_check_passes"]


def test_focus_crop_covers_only_verified_cycleway_candidates() -> None:
    coverage = _report()["candidate_coverage"]

    for name, group in coverage.items():
        if name == "cycle_lane_tag_candidates":
            assert group == {
                "way_count": 3,
                "rendered_segment_count": 34,
                "tile_count": 7,
                "osm_way_ids": [384826214, 474507374, 474507381],
            }
        else:
            assert group["way_count"] == 0
            assert group["rendered_segment_count"] == 0


def test_event_date_and_runtime_gates_remain_closed() -> None:
    report = _report()
    layer = report["layer_contract"]
    gates = report["application_gates"]

    assert layer["code_comment_period_hint"] == "2024-10"
    assert layer["exact_imagery_acquisition_date"] is None
    assert not layer["event_date_applicability_confirmed"]
    assert gates["provider_pixels_acquired_locally"]
    assert gates["provider_crs_and_grid_explicit"]
    assert gates["native_25cm_resolution_confirmed"]
    assert not gates["exact_imagery_acquisition_date_confirmed"]
    assert not gates["independent_spatial_check_passes"]
    assert not gates["event_date_marking_classification_allowed"]
    assert not gates["runtime_geometry_changed_by_this_stage"]
