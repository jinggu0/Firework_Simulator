from __future__ import annotations

import json
from pathlib import Path

from simulator import shaders
from simulator.material_textures import SCANNED_MATERIAL_ANISOTROPY
from tools.audit_material_filtering import (
    MAX_LOCAL_SCALE_DEVIATION,
    WARP_AMPLITUDE_M,
    WARP_FREQUENCY_PER_M,
)


REPORT = Path(
    "docs/validation/material_filtering_v2/material_filtering_report.json"
)


def test_antitile_warp_has_a_bounded_metric_jacobian() -> None:
    assert WARP_AMPLITUDE_M == 0.16
    assert WARP_FREQUENCY_PER_M == (0.071, 0.057)
    assert MAX_LOCAL_SCALE_DEVIATION < 0.012


def test_shader_uses_the_audited_warp_constants() -> None:
    source = shaders.source("scene.frag")

    assert "metric_uv.y * .071" in source
    assert "metric_uv.x * .057" in source
    assert ") * .16;" in source


def test_committed_filtering_report_is_a_matched_engineering_ab() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["stage"] == "V2-1b"
    assert report["capture_contract"]["matches_v2_1a"]
    assert report["implementation"]["anisotropy_requested"] == (
        SCANNED_MATERIAL_ANISOTROPY
    )
    assert report["implementation"]["anisotropy_applied"] >= 1.0
    assert report["implementation"][
        "maximum_local_scale_deviation_fraction"
    ] < 0.012
    assert [
        comparison["view_id"]
        for comparison in report["motion_comparisons"]
    ] == ["grass_close", "road_ground"]
    assert report["gates"]["shader_rendered_in_both_views"]
    assert report["gates"]["anisotropy_request_applied"]
    assert not report["gates"]["site_colour_tuning_allowed"]
    assert not report["gates"]["temporal_shimmer_gate_defined"]
