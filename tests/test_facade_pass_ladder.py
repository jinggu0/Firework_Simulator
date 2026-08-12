from __future__ import annotations

import numpy as np
import pytest

from tools.probe_facade_pass_ladder import (
    REGION_HALF_OPEN,
    STAGES,
    _state_differences,
    ladder_fragment_source,
    locate_residual_quad,
)


def test_ladder_stage_order_is_cumulative() -> None:
    assert STAGES == (
        "viewport_constants",
        "triangle_interpolation",
        "triangle_depth",
        "production_program",
        "production_program_depth",
    )


def test_ladder_shader_uses_original_fragment_coordinates() -> None:
    source = ladder_fragment_source(interpolated=False)

    assert "gl_FragCoord.x - 598.5" in source
    assert "gl_FragCoord.y - 380.5" in source
    assert "ladder_world_position" not in source


def test_interpolation_shader_replaces_only_captured_geometry_inputs() -> None:
    source = ladder_fragment_source(interpolated=True)

    assert "in vec3 ladder_world_position;" in source
    assert "in vec3 ladder_world_normal;" in source
    assert "return ladder_world_position;" in source
    assert "return ladder_world_normal;" in source
    assert "captured_reflectance" in source
    assert "captured_facade" in source


def test_residual_quad_locator_finds_the_frontmost_source_pair() -> None:
    quad = locate_residual_quad()

    assert quad.triangle_indices == (34588, 34589)
    assert quad.vertex_range == (103764, 103770)
    assert quad.source_vertices.shape == (6, 10)
    assert quad.world_vertices.shape == (6, 3)
    assert np.allclose(
        quad.source_vertices[:, 3:6],
        [-0.981241, 0.0, -0.19278513],
        atol=2e-6,
    )
    left, bottom, right, top = REGION_HALF_OPEN
    assert quad.screen_xy[:, :, 0].min() < right
    assert quad.screen_xy[:, :, 0].max() > left
    assert quad.screen_xy[:, :, 1].min() < top
    assert quad.screen_xy[:, :, 1].max() > bottom


def test_ladder_source_transform_fails_closed_on_v0_11_drift(monkeypatch) -> None:
    monkeypatch.setattr(
        "tools.probe_facade_pass_ladder.fragment_source",
        lambda stage: "#version 330\nvoid main() {}",
    )

    with pytest.raises(RuntimeError, match="source contract changed"):
        ladder_fragment_source(interpolated=False)


def test_state_differences_reports_pixel_extent_and_fp16_delta() -> None:
    reference = np.zeros((4, 2, 4), dtype=np.float16)
    candidate = reference.copy()
    candidate[:, :, 0] = np.float16(0.001204)

    differences = _state_differences(
        [reference.tobytes(), reference.tobytes(), candidate.tobytes()]
    )

    assert len(differences) == 1
    assert differences[0]["count"] == 1
    assert differences[0]["differing_pixels"] == 8
    assert differences[0]["max_abs_rgb"] == pytest.approx(0.001204, abs=1e-6)
    assert differences[0]["local_rows"] == [0, 3]
    assert differences[0]["local_columns"] == [0, 1]
