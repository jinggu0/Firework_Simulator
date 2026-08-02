from pathlib import Path
import json

import numpy as np

from simulator.scene import build_scene, build_water_mask, load_scene
from simulator.scene import (
    SURFACE_CYCLEWAY,
    SURFACE_FOOTWAY,
    SURFACE_TRAIL,
)
from simulator.site_details import _lamp, classify_path_surfaces


def test_park_lamp_has_a_separate_base_arm_housing_and_lens() -> None:
    vertices = np.asarray(_lamp(np.array([2.0, 3.0]), 0.0))
    assert vertices.shape[1] == 10
    assert len(vertices) > 180
    assert np.count_nonzero(np.isclose(vertices[:, 6], 10.0)) == 36
    assert vertices[:, 1].max() > 7.0
    assert vertices[:, 0].max() - vertices[:, 0].min() > 1.2


def test_building_footprint_becomes_walls_and_roof() -> None:
    osm = {
        "elements": [
            {
                "tags": {"building": "yes", "building:levels": "5"},
                "geometry": [
                    {"lat": 37.5, "lon": 126.9},
                    {"lat": 37.5, "lon": 126.9001},
                    {"lat": 37.5001, "lon": 126.9001},
                    {"lat": 37.5001, "lon": 126.9},
                    {"lat": 37.5, "lon": 126.9},
                ],
            }
        ]
    }
    scene = build_scene(osm, 37.5, 126.9)
    assert scene.building_vertices.shape == (30, 10)
    assert np.isclose(scene.building_vertices[:, 1].max(), 16.0)
    assert np.isfinite(scene.building_vertices).all()
    assert scene.snapshot_utc == ""


def test_bridge_way_becomes_a_deck() -> None:
    osm = {
        "elements": [
            {
                "tags": {"bridge": "yes", "width": "20"},
                "geometry": [
                    {"lat": 37.5, "lon": 126.9},
                    {"lat": 37.501, "lon": 126.9},
                ],
            }
        ]
    }
    scene = build_scene(osm, 37.5, 126.9)
    assert scene.bridge_vertices.shape == (6, 10)
    assert np.all(scene.bridge_vertices[:, 1] == 7.0)


def test_road_width_and_green_surface_are_meshed() -> None:
    osm = {
        "elements": [
            {
                "tags": {"highway": "primary", "lanes": "4"},
                "geometry": [
                    {"lat": 37.5, "lon": 126.9},
                    {"lat": 37.501, "lon": 126.9},
                ],
            },
            {
                "tags": {"leisure": "park"},
                "geometry": [
                    {"lat": 37.5, "lon": 126.9},
                    {"lat": 37.5, "lon": 126.9001},
                    {"lat": 37.5001, "lon": 126.9001},
                    {"lat": 37.5001, "lon": 126.9},
                    {"lat": 37.5, "lon": 126.9},
                ],
            },
        ]
    }
    scene = build_scene(osm, 37.5, 126.9, "2024-10-05T10:20:00Z")
    assert scene.road_vertices.shape == (6, 10)
    assert np.all(scene.road_vertices[:, 6] == 3.0)
    assert scene.vegetation_vertices.shape == (6, 10)
    assert np.all(scene.vegetation_vertices[:, 6] == 4.0)
    assert scene.snapshot_utc == "2024-10-05T10:20:00Z"


def test_shipped_scene_uses_official_contour_grid_and_event_water_datum() -> None:
    scene = load_scene(
        Path(__file__).resolve().parent.parent / "assets" / "yeouido_scene.npz"
    )
    assert scene.terrain_height_m.shape == (1024, 1024)
    assert np.isclose(scene.elevation_datum_m, 2.79, atol=1e-6)


def test_named_landmarks_receive_distinct_facade_styles() -> None:
    def building(name: str, longitude: float, **tags: str) -> dict:
        return {
            "tags": {"building": "office", "name": name, **tags},
            "geometry": [
                {"lat": 37.5, "lon": longitude},
                {"lat": 37.5, "lon": longitude + 0.00005},
                {"lat": 37.50005, "lon": longitude + 0.00005},
                {"lat": 37.50005, "lon": longitude},
                {"lat": 37.5, "lon": longitude},
            ],
        }

    scene = build_scene(
        {
            "elements": [
                building("63시티", 126.9000, height="252"),
                building("Three IFC", 126.9001, height="283"),
                building("파크원 타워1", 126.9002, height="322"),
                building("전경련회관", 126.9003, height="246"),
                building("국회의사당", 126.9004, building="government"),
                building(
                    "101동", 126.9005, height="80", building="apartments"
                ),
            ]
        },
        37.5,
        126.9,
    )
    assert {1.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 10.0}.issubset(
        set(np.unique(scene.building_vertices[:, 9]))
    )


def test_architect_published_landmark_heights_override_stale_osm_values() -> None:
    def maximum_height(name: str, tagged_height: str) -> float:
        scene = build_scene(
            {
                "elements": [
                    {
                        "tags": {
                            "building": "office",
                            "name": name,
                            "height": tagged_height,
                        },
                        "geometry": [
                            {"lat": 37.5, "lon": 126.9},
                            {"lat": 37.5, "lon": 126.9001},
                            {"lat": 37.5001, "lon": 126.9001},
                            {"lat": 37.5001, "lon": 126.9},
                            {"lat": 37.5, "lon": 126.9},
                        ],
                    }
                ]
            },
            37.5,
            126.9,
        )
        return float(scene.building_vertices[:, 1].max())

    assert maximum_height("파크원 타워1", "322") == 318.0
    assert maximum_height("NH금융타워(타워2)", "256") == 246.0
    assert maximum_height("전경련회관", "246") > 240.0  # 10° PV canopy


def test_landmark_parent_is_not_drawn_over_its_building_parts() -> None:
    def polygon(offset: float, size: float) -> list[dict[str, float]]:
        return [
            {"lat": 37.5 + offset, "lon": 126.9 + offset},
            {"lat": 37.5 + offset, "lon": 126.9 + offset + size},
            {"lat": 37.5 + offset + size, "lon": 126.9 + offset + size},
            {"lat": 37.5 + offset + size, "lon": 126.9 + offset},
            {"lat": 37.5 + offset, "lon": 126.9 + offset},
        ]

    scene = build_scene(
        {
            "elements": [
                {
                    "tags": {"building": "office", "name": "One IFC", "height": "185"},
                    "geometry": polygon(0.0, 0.0002),
                },
                {
                    "tags": {"building:part": "yes", "height": "120"},
                    "geometry": polygon(0.00004, 0.0001),
                },
            ]
        },
        37.5,
        126.9,
    )
    assert scene.building_vertices[:, 1].max() == 120.0
    assert set(np.unique(scene.building_vertices[:, 9])) == {1.0}


def test_skillion_roof_is_a_sloped_surface_not_a_flat_extrusion() -> None:
    scene = build_scene(
        {
            "elements": [
                {
                    "tags": {
                        "building:part": "yes",
                        "height": "60",
                        "roof:height": "30",
                        "roof:shape": "skillion",
                    },
                    "geometry": [
                        {"lat": 37.5, "lon": 126.9},
                        {"lat": 37.5, "lon": 126.9002},
                        {"lat": 37.5001, "lon": 126.9002},
                        {"lat": 37.5001, "lon": 126.9},
                        {"lat": 37.5, "lon": 126.9},
                    ],
                }
            ]
        },
        37.5,
        126.9,
    )
    roof = scene.building_vertices[scene.building_vertices[:, 6] == 1.0]
    walls = scene.building_vertices[scene.building_vertices[:, 6] == 0.0]
    assert np.ptp(roof[:, 1]) > 29.0
    assert np.any(np.abs(roof[:, 3]) > 0.01) or np.any(np.abs(roof[:, 5]) > 0.01)
    assert np.ptp(walls[:, 7]) > 10.0


def test_minimum_height_preserves_elevated_building_part() -> None:
    scene = build_scene(
        {
            "elements": [
                {
                    "tags": {
                        "building": "roof",
                        "height": "24",
                        "min_height": "20",
                    },
                    "geometry": [
                        {"lat": 37.5, "lon": 126.9},
                        {"lat": 37.5, "lon": 126.9001},
                        {"lat": 37.5001, "lon": 126.9001},
                        {"lat": 37.5001, "lon": 126.9},
                        {"lat": 37.5, "lon": 126.9},
                    ],
                }
            ]
        },
        37.5,
        126.9,
    )
    wall_vertices = scene.building_vertices[
        scene.building_vertices[:, 6] == 0.0
    ]
    assert np.isclose(wall_vertices[:, 1].min(), 20.0)


def test_dome_building_part_has_curved_3d_roof() -> None:
    scene = build_scene(
        {
            "elements": [
                {
                    "tags": {
                        "building:part": "yes",
                        "height": "30",
                        "min_height": "20",
                        "roof:shape": "dome",
                    },
                    "geometry": [
                        {"lat": 37.5, "lon": 126.9},
                        {"lat": 37.5, "lon": 126.9001},
                        {"lat": 37.5001, "lon": 126.9001},
                        {"lat": 37.5001, "lon": 126.9},
                        {"lat": 37.5, "lon": 126.9},
                    ],
                }
            ]
        },
        37.5,
        126.9,
    )
    heights = np.unique(scene.building_vertices[:, 1])
    assert len(heights) > 5
    assert np.isclose(heights.min(), 20.0)
    assert np.isclose(heights.max(), 30.0)
    assert np.all(scene.building_vertices[:, 6] == 1.0)


def test_water_relation_preserves_inner_land_ring() -> None:
    nodes = [
        {"type": "node", "id": 1, "lat": 37.499, "lon": 126.899},
        {"type": "node", "id": 2, "lat": 37.499, "lon": 126.901},
        {"type": "node", "id": 3, "lat": 37.501, "lon": 126.901},
        {"type": "node", "id": 4, "lat": 37.501, "lon": 126.899},
        {"type": "node", "id": 5, "lat": 37.4998, "lon": 126.8998},
        {"type": "node", "id": 6, "lat": 37.4998, "lon": 126.9002},
        {"type": "node", "id": 7, "lat": 37.5002, "lon": 126.9002},
        {"type": "node", "id": 8, "lat": 37.5002, "lon": 126.8998},
    ]
    relation_data = {
        "elements": nodes
        + [
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1]},
            {"type": "way", "id": 11, "nodes": [5, 6, 7, 8, 5]},
            {
                "type": "relation",
                "id": 152336,
                "members": [
                    {"type": "way", "ref": 10, "role": "outer"},
                    {"type": "way", "ref": 11, "role": "inner"},
                ],
            },
        ]
    }
    mask, _ = build_water_mask(
        relation_data,
        37.5,
        126.9,
        bounds=(-120.0, -140.0, 120.0, 140.0),
        resolution=(128, 128),
    )
    assert mask[64, 64] == 0
    assert mask[30, 30] == 255


def test_shipped_scene_is_the_event_time_snapshot() -> None:
    scene = load_scene(
        Path(__file__).resolve().parent.parent
        / "assets"
        / "yeouido_scene.npz"
    )
    assert scene.snapshot_utc == "2024-10-05T10:20:00Z"
    assert scene.building_vertices.shape[1] == 10
    assert len(scene.building_vertices) > 99_000
    assert len(scene.road_vertices) > 60_000
    assert len(scene.vegetation_vertices) > 5_000
    assert len(scene.detail_vertices) > 100_000
    assert {
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 16.0
    }.issubset(set(np.unique(scene.detail_vertices[:, 6])))
    assert {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}.issubset(
        set(np.unique(scene.building_vertices[:, 9]))
    )
    assembly_columns = scene.building_vertices[
        np.isclose(scene.building_vertices[:, 9], 10.0)
    ]
    assert len(assembly_columns) == 24 * 8 * 6
    assert np.count_nonzero(
        np.isclose(scene.building_vertices[:, 9], 9.0)
    ) > 2_000
    assembly_dome = scene.building_vertices[
        np.isclose(scene.building_vertices[:, 9], 12.0)
    ]
    assert np.isclose(assembly_dome[:, 1].min(), 43.0)
    assert np.isclose(assembly_dome[:, 1].max(), 61.0)
    assert np.isclose(scene.building_vertices[:, 1].max(), 318.0)


def test_imported_path_widths_select_distinct_surface_materials() -> None:
    def segment(width: float, x: float) -> np.ndarray:
        vertices = np.zeros((6, 10), dtype=np.float32)
        vertices[:, :3] = np.array(
            [
                [x, 0.06, -width * 0.5],
                [x, 0.06, width * 0.5],
                [x + 10.0, 0.06, width * 0.5],
                [x, 0.06, -width * 0.5],
                [x + 10.0, 0.06, width * 0.5],
                [x + 10.0, 0.06, -width * 0.5],
            ]
        )
        vertices[:, 6] = 3.0
        return vertices

    roads = np.concatenate(
        [segment(1.5, 0.0), segment(2.2, 20.0), segment(2.5, 40.0)]
    )
    classified = classify_path_surfaces(roads)
    assert np.all(classified[:6, 6] == SURFACE_TRAIL)
    assert np.all(classified[6:12, 6] == SURFACE_FOOTWAY)
    assert np.all(classified[12:18, 6] == SURFACE_CYCLEWAY)


def test_detail_source_snapshots_retain_provenance_and_counts() -> None:
    assets = Path(__file__).resolve().parent.parent / "assets"
    historical = json.loads(
        (assets / "yeouido_detail_osm_2024-10-05.json").read_text(
            encoding="utf-8"
        )
    )
    facilities = json.loads(
        (assets / "yeouido_official_facilities.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(historical["elements"]) == 131
    assert "2024-10-05T10:20:00Z" in str(historical["osm3s"])
    assert facilities["park_code"] == "Hzone007"
    assert facilities["retrieved_date"] == "2026-07-29"
    assert len(facilities["facilities"]) == 121
