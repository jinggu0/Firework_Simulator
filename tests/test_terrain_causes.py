from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from simulator.geodesy import LocalTangentPlane
from simulator.scene import StaticScene
from simulator.validation.terrain_causes import (
    TerrainPriorityArea,
    analyze_terrain_causes,
    iter_tessellated_road_samples,
    load_priority_area,
    render_terrain_cause_map,
)
from simulator.validation.terrain_contacts import audit_terrain_contacts


def _scene(height: np.ndarray, water: np.ndarray, roads: np.ndarray) -> StaticScene:
    empty = np.empty((0, 10), dtype=np.float32)
    bounds = np.array((0.0, 0.0, 2.0, 2.0), dtype=np.float32)
    return StaticScene(
        building_vertices=empty,
        bridge_vertices=empty,
        road_vertices=roads,
        vegetation_vertices=empty,
        detail_vertices=empty,
        water_mask=np.asarray(water, dtype=np.uint8) * 255,
        water_mask_bounds=bounds,
        terrain_height_m=np.asarray(height, dtype=np.float32),
        terrain_bounds=bounds,
        elevation_datum_m=0.0,
        origin_latitude_deg=37.0,
        origin_longitude_deg=126.0,
        snapshot_utc="2024-10-05T00:00:00Z",
    )


def _triangle() -> np.ndarray:
    vertices = np.zeros((3, 10), dtype=np.float32)
    vertices[:, [0, 2]] = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    vertices[:, 1] = 0.06
    vertices[:, 6] = 3.0
    return vertices


def _area() -> TerrainPriorityArea:
    return TerrainPriorityArea(
        area_id="test",
        bounds_xz_m=(0.0, 0.0, 2.0, 2.0),
        confidence_grade="C",
        purpose="test area",
        derivation={},
        scene_asset="",
        scene_asset_sha256="0" * 64,
    )


def test_shipped_priority_area_is_tied_to_the_scene_and_camera_scope() -> None:
    area = load_priority_area()
    assert area.verify_scene_asset().name == "yeouido_scene.npz"
    assert area.area_id == "event_park_north_bank"
    assert area.confidence_grade == "C"
    camera_positions = np.array(
        [
            (-120.0, 330.0),
            (180.0, 800.0),
            (-500.0, 650.0),
            (-650.0, 900.0),
            (-900.0, 250.0),
            (-100.0, 260.0),
        ]
    )
    assert area.contains(camera_positions).all()


def test_priority_area_encloses_its_declared_official_facilities() -> None:
    area = load_priority_area()
    facilities = json.loads(
        Path("assets/yeouido_official_facilities.json").read_text(encoding="utf-8")
    )["facilities"]
    declared = set(area.derivation["facility_codes_enclosed"])
    tangent = LocalTangentPlane(37.529, 126.935, 0.0)
    positions = []
    for facility in facilities:
        if facility["code"] in declared:
            local = tangent.to_local(
                facility["latitude"], facility["longitude"], 0.0
            )
            positions.append((local[0], local[2]))
    assert len(positions) == len(declared)
    assert area.contains(np.asarray(positions)).all()


def test_midpoint_subdivision_increases_triangle_count_without_moving_extent() -> None:
    scene = _scene(np.zeros((3, 3)), np.zeros((3, 3)), _triangle())
    level_zero = list(iter_tessellated_road_samples(scene, 0))
    level_two = list(iter_tessellated_road_samples(scene, 2))
    assert sum(len(batch[0]) for batch in level_zero) == 4
    assert sum(len(batch[0]) for batch in level_two) == 64
    positions = np.concatenate([batch[0] for batch in level_two])
    assert positions.min() >= 0.0
    assert positions.max() <= 2.0


def test_refinement_reduces_representation_error_over_an_unchanged_peak() -> None:
    height = np.zeros((3, 3), dtype=np.float32)
    height[1, 1] = 2.0
    scene = _scene(height, np.zeros((3, 3)), _triangle())
    audit = audit_terrain_contacts(scene)
    analysis = analyze_terrain_causes(scene, audit, _area())
    levels = analysis.report["road_tessellation_counterfactual"]["levels"]
    assert levels[0]["priority_area"]["priority_fraction"] > 0.0
    assert (
        levels[2]["priority_area"]["priority_fraction"]
        < levels[0]["priority_area"]["priority_fraction"]
    )
    assert (
        analysis.report["road_tessellation_counterfactual"][
            "priority_fraction_reduction_at_final_level"
        ]
        > 0.0
    )


def test_north_shoreline_requires_water_in_the_north_neighbor() -> None:
    height = np.full((3, 3), 5.0, dtype=np.float32)
    height[0] = 0.0
    water = np.zeros((3, 3), dtype=bool)
    water[0] = True
    scene = _scene(height, water, np.empty((0, 10), dtype=np.float32))
    audit = audit_terrain_contacts(scene)
    analysis = analyze_terrain_causes(scene, audit, _area())
    assert analysis.north_shoreline[1].all()
    assert analysis.report["north_shoreline"]["sample_count"] == 3
    assert analysis.report["north_shoreline"]["warning_height_sample_count"] == 3


def test_cause_map_aligns_to_the_terrain_grid(tmp_path) -> None:
    scene = _scene(np.zeros((3, 3)), np.zeros((3, 3)), _triangle())
    audit = audit_terrain_contacts(scene)
    analysis = analyze_terrain_causes(scene, audit, _area())
    path = render_terrain_cause_map(
        scene, audit, _area(), analysis, tmp_path / "causes.jpg"
    )
    with Image.open(path) as image:
        assert image.size == (3, 97)
