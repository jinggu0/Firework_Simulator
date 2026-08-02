from __future__ import annotations

import json

import numpy as np
from PIL import Image

from simulator.passes.scene import linear_feature_uv, road_edge_detail_vertices
from simulator.scene import LINEAR_STYLE_STEPS, load_scene
from simulator.terrain_detail import (
    adaptive_terrain_tessellate,
    load_priority_area_bounds,
)
from simulator.validation.terrain_causes import load_priority_area
from tools.compose_visual_ab import compose
from tools.measure_adaptive_road_mesh import _performance_comparison


BOUNDS = np.array((0.0, 0.0, 2.0, 2.0), dtype=np.float32)


def _triangle() -> np.ndarray:
    vertices = np.zeros((3, 10), dtype=np.float32)
    vertices[:, [0, 2]] = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    vertices[:, 1] = 0.06
    vertices[:, 3:6] = (0.0, 1.0, 0.0)
    vertices[:, 6] = 3.0
    vertices[:, 7:9] = ((0.0, -1.0), (2.0, 1.0), (2.0, -1.0))
    return vertices


def _tessellate(
    vertices: np.ndarray,
    height: np.ndarray,
    *,
    water: np.ndarray | None = None,
    area: tuple[float, float, float, float] = (0.0, 0.0, 2.0, 2.0),
):
    mask = (
        np.asarray(water, dtype=np.uint8) * 255
        if water is not None
        else np.zeros_like(height, dtype=np.uint8)
    )
    return adaptive_terrain_tessellate(
        vertices,
        np.asarray(height, dtype=np.float32),
        BOUNDS,
        mask,
        BOUNDS,
        area,
    )


def test_flat_terrain_keeps_the_original_triangle_bit_exact() -> None:
    source = _triangle()
    output, stats = _tessellate(source, np.zeros((3, 3)))
    np.testing.assert_array_equal(output, source)
    assert stats.input_triangles == 1
    assert stats.output_triangles == 1
    assert stats.refined_triangles_by_level == (0,)


def test_error_driven_refinement_interpolates_existing_attributes() -> None:
    height = np.zeros((3, 3), dtype=np.float32)
    height[1, 1] = 2.0
    source = _triangle()
    output, stats = _tessellate(source, height)
    assert stats.output_triangles > stats.input_triangles
    assert stats.output_triangles <= stats.input_triangles * 16
    assert set(np.unique(output[:, 6])) == {3.0}
    assert output[:, 7].min() >= source[:, 7].min()
    assert output[:, 7].max() <= source[:, 7].max()
    assert output[:, 8].min() >= source[:, 8].min()
    assert output[:, 8].max() <= source[:, 8].max()


def test_area_and_water_gates_prevent_unsupported_refinement() -> None:
    height = np.zeros((3, 3), dtype=np.float32)
    height[1, 1] = 2.0
    source = _triangle()
    outside, outside_stats = _tessellate(
        source, height, area=(5.0, 5.0, 6.0, 6.0)
    )
    water, water_stats = _tessellate(
        source, height, water=np.ones((3, 3), dtype=bool)
    )
    np.testing.assert_array_equal(outside, source)
    np.testing.assert_array_equal(water, source)
    assert outside_stats.output_triangles == 1
    assert water_stats.output_triangles == 1


def test_runtime_priority_bounds_match_the_audited_area() -> None:
    assert load_priority_area_bounds() == load_priority_area().bounds_xz_m


def test_shipped_scene_refines_only_a_small_fraction_of_road_triangles() -> None:
    scene = load_scene("assets/yeouido_scene.npz")
    road = linear_feature_uv(scene.road_vertices)
    ordinary = road[~np.isclose(road[:, 9], LINEAR_STYLE_STEPS)]
    output, stats = adaptive_terrain_tessellate(
        ordinary,
        scene.terrain_height_m,
        scene.terrain_bounds,
        scene.water_mask,
        scene.water_mask_bounds,
        load_priority_area_bounds(),
    )
    assert stats.input_triangles == 70_464
    assert stats.output_triangles == 76_236
    assert stats.refined_triangles_by_level == (1_001, 923)
    assert stats.triangle_multiplier < 1.1
    assert len(output) == stats.output_triangles * 3

    edges = road_edge_detail_vertices(ordinary)
    edge_output, edge_stats = adaptive_terrain_tessellate(
        edges,
        scene.terrain_height_m,
        scene.terrain_bounds,
        scene.water_mask,
        scene.water_mask_bounds,
        load_priority_area_bounds(),
    )
    assert edge_stats.input_triangles == 81_024
    assert edge_stats.output_triangles == 86_553
    assert edge_stats.refined_triangles_by_level == (858, 985)
    assert edge_stats.triangle_multiplier < 1.1
    assert len(edge_output) == edge_stats.output_triangles * 3


def test_runtime_profile_comparison_keeps_all_paired_runs(tmp_path) -> None:
    def write_profile(name: str, frame_p95_ms: float):
        payload = {
            "integrated": {
                "frame_mean_ms": 13.0,
                "frame_p95_ms": frame_p95_ms,
                "frame_p99_ms": 18.0,
                "visual_p95_ms": 12.0,
                "physics_p95_ms": 5.5,
            }
        }
        path = tmp_path / name
        path.write_text(
            "pygame support prompt\n" + json.dumps(payload), encoding="utf-8"
        )
        return path

    baseline = [write_profile("baseline.json", 17.2)]
    candidate = [write_profile("candidate.json", 17.0)]
    comparison = _performance_comparison(baseline, candidate)
    assert comparison["paired_runs"][0]["frame_p95_delta_ms"] < 0.0
    assert not comparison["candidate_all_runs_pass_60fps_p95"]


def test_visual_ab_composer_preserves_three_labelled_panels(tmp_path) -> None:
    before = np.zeros((9, 16, 3), dtype=np.uint8)
    after = before.copy()
    after[:, 8:] = 120
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    Image.fromarray(before).save(before_path)
    Image.fromarray(after).save(after_path)
    output = compose(before_path, after_path, tmp_path / "ab.jpg")
    with Image.open(output) as image:
        assert image.size == (48, 43)
