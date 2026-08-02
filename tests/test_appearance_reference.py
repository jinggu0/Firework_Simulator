import json
from pathlib import Path

import numpy as np
import pytest

from simulator.passes.scene import (
    bridge_lighting_uv,
    bridge_structure_vertices,
    grass_detail_chunks,
    linear_feature_uv,
    road_edge_detail_vertices,
    rooftop_detail_vertices,
    stair_structure_vertices,
)
from simulator.scene import (
    LINEAR_STYLE_STEPS,
    SURFACE_GRASS_BLADE,
    load_scene,
)
from simulator.site_details import GRASS_VERTICES_PER_TUFT
from tools.analyze_appearance_reference import crop_statistics


ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "assets" / "yeouido_2024-10-05_appearance_reference.json"


def test_event_appearance_reference_records_sources_and_limits() -> None:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert data["event_date"] == "2024-10-05"
    assert len(data["sources"]) >= 4
    assert all(source["page_url"] for source in data["sources"].values())
    assert data["limitations"]
    assert data["implemented_calibration"]["water_reflection_taps"] == 3


def test_appearance_reference_matches_authored_grass_count() -> None:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    scene = load_scene(ROOT / "assets" / "yeouido_scene.npz")
    grass_vertices = scene.detail_vertices[
        np.isclose(scene.detail_vertices[:, 6], SURFACE_GRASS_BLADE)
    ]
    # Each mapped sample expands to five narrow double-sided blades.
    assert len(grass_vertices) % GRASS_VERTICES_PER_TUFT == 0
    assert (
        len(grass_vertices) // GRASS_VERTICES_PER_TUFT
        == data["implemented_calibration"][
            "grass_blades_authored"
        ]
    )
    chunks = grass_detail_chunks(scene.detail_vertices)
    assert len(chunks) > 10
    assert sum(len(vertices) for vertices, _, _ in chunks) == len(
        grass_vertices
    )
    assert all(radius <= 46.0 for _, _, radius in chunks)


def test_bridge_uv_tracks_distance_and_edges_across_connected_segments() -> None:
    vertices = np.zeros((12, 10), dtype=np.float32)
    vertices[:6, :3] = np.array(
        [[0, 7, -2], [0, 7, 2], [10, 7, 2],
         [0, 7, -2], [10, 7, 2], [10, 7, -2]], dtype=np.float32
    )
    vertices[6:, :3] = np.array(
        [[10, 7, -2], [10, 7, 2], [25, 7, 2],
         [10, 7, -2], [25, 7, 2], [25, 7, -2]], dtype=np.float32
    )
    converted = bridge_lighting_uv(vertices)
    assert converted is not vertices
    assert converted[:, 8].tolist() == [-1, 1, 1, -1, 1, -1] * 2
    assert converted[:6, 7].tolist() == [0, 0, 10, 0, 10, 10]
    assert converted[6:, 7].tolist() == [10, 10, 25, 10, 25, 25]


def test_bridge_deck_derives_visible_fascia_and_underside() -> None:
    vertices = np.zeros((6, 10), dtype=np.float32)
    vertices[:, :3] = np.array(
        [[0, 7, -5], [0, 7, 5], [100, 7, 5],
         [0, 7, -5], [100, 7, 5], [100, 7, -5]], dtype=np.float32
    )
    mapped = bridge_lighting_uv(vertices)
    structure = bridge_structure_vertices(mapped)
    assert structure.shape[1] == 10
    assert len(structure) >= 18
    assert np.isfinite(structure).all()
    assert structure[:, 1].min() < vertices[:, 1].min()
    assert {2.0, 12.0}.issubset(set(structure[:, 6]))


def test_asphalt_edges_derive_a_raised_concrete_kerb() -> None:
    vertices = np.zeros((6, 10), dtype=np.float32)
    vertices[:, :3] = np.array(
        [[0, .06, -3], [0, .06, 3], [12, .06, 3],
         [0, .06, -3], [12, .06, 3], [12, .06, -3]], dtype=np.float32
    )
    vertices[:, 6] = 3.0
    kerbs = road_edge_detail_vertices(vertices)
    assert kerbs.shape == (24, 10)
    assert np.all(kerbs[:, 6] == 12.0)
    assert kerbs[:, 1].max() - vertices[:, 1].max() == pytest.approx(0.14)


def test_resolved_stair_rise_becomes_level_treads_and_bounded_risers() -> None:
    vertices = np.zeros((6, 10), dtype=np.float32)
    vertices[:, :3] = np.array(
        [[0, .06, -1], [0, .06, 1], [1.2, .06, 1],
         [0, .06, -1], [1.2, .06, 1], [1.2, .06, -1]], dtype=np.float32
    )
    vertices[:, 6] = 5.0
    vertices[:, 9] = 1.0
    terrain = np.array([[0.0, 0.6], [0.0, 0.6]], dtype=np.float32)
    stairs = stair_structure_vertices(
        vertices, terrain, np.array([0.0, -1.0, 1.2, 1.0])
    )
    # Four 0.30 m treads and four <= 0.18 m risers, two triangles each.
    assert stairs.shape == (48, 10)
    assert np.all(stairs[:, 9] == 1.0)
    world_y = stairs[:, 1] + np.interp(stairs[:, 0], [0.0, 1.2], [0.0, 0.6])
    for offset in range(0, len(stairs), 12):
        assert np.ptp(world_y[offset : offset + 6]) < 1e-6
        assert np.ptp(world_y[offset + 6 : offset + 12]) <= 0.15 + 1e-6


def test_unresolved_stair_rise_keeps_the_draped_source_deck() -> None:
    vertices = np.zeros((6, 10), dtype=np.float32)
    vertices[:, :3] = np.array(
        [[0, .06, -1], [0, .06, 1], [2, .06, 1],
         [0, .06, -1], [2, .06, 1], [2, .06, -1]], dtype=np.float32
    )
    vertices[:, 6] = 5.0
    vertices[:, 9] = 1.0
    stairs = stair_structure_vertices(
        vertices, np.zeros((2, 2), dtype=np.float32),
        np.array([0.0, -1.0, 2.0, 1.0]),
    )
    assert np.array_equal(stairs[:, :9], vertices[:, :9])
    assert np.all(stairs[:, 9] == 0.0)


def test_shipped_historical_steps_expand_against_official_terrain() -> None:
    scene = load_scene(ROOT / "assets" / "yeouido_scene.npz")
    roads = linear_feature_uv(scene.road_vertices)
    source = roads[np.isclose(roads[:, 9], LINEAR_STYLE_STEPS)]
    stairs = stair_structure_vertices(
        source, scene.terrain_height_m, scene.terrain_bounds
    )
    assert source.shape == (648, 10)
    assert stairs.shape == (13_092, 10)
    assert np.isfinite(stairs).all()


def test_large_flat_roof_gets_a_bounded_mechanical_penthouse() -> None:
    vertices = np.zeros((3, 10), dtype=np.float32)
    vertices[:, :3] = np.array(
        [[0, 40, 0], [50, 40, 0], [0, 40, 50]], dtype=np.float32
    )
    vertices[:, 6] = 1.0
    detail = rooftop_detail_vertices(vertices)
    assert detail.shape == (30, 10)
    assert detail[:, 1].min() >= 40.0
    assert detail[:, 1].max() > 41.0
    assert np.all(detail[:, 6] == 12.0)


def test_reference_crop_statistics_are_display_referred_and_repeatable() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :10] = (10, 20, 30)
    stats = crop_statistics(image, [0.0, 0.0, 0.5, 1.0])
    assert stats["pixel_bounds"] == [0, 0, 10, 10]
    assert stats["median_srgb_8bit"] == [10.0, 20.0, 30.0]
