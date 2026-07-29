from pathlib import Path

import numpy as np

from simulator.scene import build_scene, build_water_mask, load_scene


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
                building(
                    "101동", 126.9003, height="80", building="apartments"
                ),
            ]
        },
        37.5,
        126.9,
    )
    assert {1.0, 2.0, 3.0, 5.0}.issubset(
        set(np.unique(scene.building_vertices[:, 9]))
    )


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
    assert {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}.issubset(
        set(np.unique(scene.building_vertices[:, 9]))
    )
    assert np.isclose(scene.building_vertices[:, 1].max(), 322.0)
