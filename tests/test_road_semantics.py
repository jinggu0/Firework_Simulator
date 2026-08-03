from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from simulator.road_semantics import (
    RoadSemanticCorridor,
    RoadStructureSemantics,
    filter_occluded_road_segments,
    load_road_structure_semantics,
)
from simulator.scene import load_scene


def _quad(start: tuple[float, float], end: tuple[float, float], width: float = 4.0):
    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    direction = b - a
    direction /= np.linalg.norm(direction)
    side = np.array((-direction[1], direction[0]), dtype=np.float32) * width * 0.5
    points = (a - side, a + side, b + side, a - side, b + side, b - side)
    vertices = np.zeros((6, 10), dtype=np.float32)
    vertices[:, 0] = [point[0] for point in points]
    vertices[:, 2] = [point[1] for point in points]
    vertices[:, 1] = 0.06
    vertices[:, 3:6] = (0.0, 1.0, 0.0)
    vertices[:, 6] = 3.0
    return vertices


def _semantics() -> RoadStructureSemantics:
    return RoadStructureSemantics(
        scene_asset="scene.npz",
        scene_asset_sha256="0" * 64,
        snapshot_utc="2024-10-05T10:20:00Z",
        centreline_tolerance_m=0.25,
        heading_cosine_minimum=0.98,
        corridors=(
            RoadSemanticCorridor(
                osm_way_id=42,
                kind="tunnel",
                render_policy="occluded_below_terrain",
                polyline_xz_m=np.asarray(((-10.0, 0.0), (10.0, 0.0))),
                tags={"tunnel": "yes"},
            ),
        ),
    )


def test_heading_match_removes_tunnel_but_preserves_crossing_surface_road() -> None:
    tunnel = _quad((-8.0, 0.0), (8.0, 0.0))
    crossing = _quad((0.0, -8.0), (0.0, 8.0))
    output, stats = filter_occluded_road_segments(
        np.concatenate((tunnel, crossing)), _semantics()
    )
    np.testing.assert_array_equal(output, crossing)
    assert stats.input_segments == 2
    assert stats.output_segments == 1
    assert stats.excluded_segments == 1
    assert stats.excluded_osm_way_ids == (42,)


def test_nearby_parallel_surface_road_is_outside_exact_centreline_gate() -> None:
    nearby = _quad((-8.0, 0.5), (8.0, 0.5))
    output, stats = filter_occluded_road_segments(nearby, _semantics())
    np.testing.assert_array_equal(output, nearby)
    assert stats.excluded_segments == 0
    assert stats.excluded_osm_way_ids == ()


def test_dated_runtime_semantics_match_all_eight_shipped_tunnel_ways() -> None:
    scene = load_scene("assets/yeouido_scene.npz")
    semantics = load_road_structure_semantics()
    output, stats = filter_occluded_road_segments(
        scene.road_vertices, semantics
    )
    assert semantics.snapshot_utc == scene.snapshot_utc
    scene_path = Path("assets/yeouido_scene.npz")
    assert semantics.scene_asset == scene_path.as_posix()
    assert semantics.scene_asset_sha256 == sha256(scene_path.read_bytes()).hexdigest()
    assert len(semantics.corridors) == 8
    assert stats.input_segments == 35_340
    assert stats.output_segments == 35_215
    assert stats.excluded_segments == 125
    assert stats.excluded_osm_way_ids == tuple(
        corridor.osm_way_id for corridor in semantics.corridors
    )
    assert len(output) == len(scene.road_vertices) - 125 * 6
