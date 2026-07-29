import numpy as np

from simulator.scene import build_scene


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
    assert scene.building_vertices.shape == (30, 7)
    assert np.isclose(scene.building_vertices[:, 1].max(), 16.0)
    assert np.isfinite(scene.building_vertices).all()


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
    assert scene.bridge_vertices.shape == (6, 7)
    assert np.all(scene.bridge_vertices[:, 1] == 7.0)

