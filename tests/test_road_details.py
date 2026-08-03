from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator import shaders
from simulator.materials import MATERIAL_LIBRARY, PatternKind
from simulator.passes.scene import (
    KERB_REVEAL_HEIGHT_M,
    KERB_TOP_WIDTH_M,
)
from simulator.validation.road_details import road_quad_measurements
from simulator.scene import SURFACE_CYCLEWAY


REPORT = Path("docs/validation/road_detail_v1/road_detail_audit.json")


def test_road_quad_measurement_recovers_metric_width_and_length() -> None:
    vertices = np.zeros((6, 10), dtype=np.float32)
    vertices[:, :3] = np.array(
        [
            [0.0, 0.06, -3.2],
            [0.0, 0.06, 3.2],
            [12.0, 0.06, 3.2],
            [0.0, 0.06, -3.2],
            [12.0, 0.06, 3.2],
            [12.0, 0.06, -3.2],
        ]
    )
    vertices[:, 6] = 3.0

    result = road_quad_measurements(vertices)

    assert result["length_m"].tolist() == pytest.approx([12.0])
    assert result["width_m"].tolist() == pytest.approx([6.4])
    assert result["surface_code"].tolist() == [3]


def test_road_shader_exposes_metric_semantic_paint_contract() -> None:
    source = shaders.source("scene.frag")

    for name in (
        "ROAD_MARKING_ENCODING_BASE",
        "ROAD_MARKING_LANE_STRIDE",
        "ROAD_LANE_LINE_WIDTH_M",
        "ROAD_LANE_DASH_PAINT_M",
        "ROAD_LANE_DASH_GAP_M",
    ):
        assert f"const float {name}" in source
    assert "ROAD_EDGE_LINE_V_INNER" not in source


def test_kerb_contract_remains_explicitly_generic() -> None:
    assert KERB_REVEAL_HEIGHT_M == 0.14
    assert KERB_TOP_WIDTH_M == 0.18


def test_cycleway_surface_has_no_inferred_transverse_paint() -> None:
    cycleway = MATERIAL_LIBRARY.get(SURFACE_CYCLEWAY)
    assert cycleway.pattern == PatternKind.UNIFORM
    assert cycleway.pattern_mix == 0.0


def test_committed_road_detail_audit_blocks_unsupported_geometry() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["stage"] == "V2-2a"
    # This is an archived before-state. Its implementation hashes intentionally
    # identify the pre-V2-2c files rather than the current working tree.
    for source in report["implementation_sources"].values():
        assert len(source["sha256"]) == 64
    inventory = report["runtime_road_inventory"]
    assert inventory["visible_segment_count"] == 35_215
    assert [surface["surface"] for surface in inventory["surfaces"]] == [
        "asphalt_road",
        "concrete_footway",
        "cycleway",
        "compacted_trail",
    ]
    assert report["kerb_contract"]["confidence_grade"] == "D"
    assert not report["kerb_contract"]["surveyed_at_yeouido"]
    paint = report["asphalt_marking_contract"]
    assert paint["edge_line_support_width_m"]["maximum"] > 1.0
    assert not paint["road_specific_lane_semantics_available"]
    missing = report["missing_detail_inventory"]
    assert not missing["manhole_geometry_present"]
    assert missing["detail_snapshot"]["highway_element_count"] == 0
    gates = report["gates"]
    assert gates["current_scale_audit_complete"]
    assert not gates["marking_width_is_metric"]
    assert not gates["detail_geometry_application_allowed"]
