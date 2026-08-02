import json
from pathlib import Path

import numpy as np

from simulator.passes.scene import (
    bridge_lighting_uv,
    bridge_structure_vertices,
    rooftop_detail_vertices,
)
from simulator.scene import SURFACE_GRASS_BLADE, load_scene
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
    # Each tuft is two crossed, double-sided triangles: twelve vertices.
    assert len(grass_vertices) % 12 == 0
    assert len(grass_vertices) // 12 == data["implemented_calibration"][
        "grass_blades_authored"
    ]


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
