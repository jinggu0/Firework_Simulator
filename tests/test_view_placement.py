from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulator.camera import CameraConfig
from simulator.scene import load_scene
from simulator.terrain import TerrainSurface
from simulator.validation.view_placement import (
    advance_out_of_buildings,
    building_faces_overhead_m,
    nearest_building_face_m,
    validate_view_placement,
)
from simulator.validation.views import VisualViewError, load_visual_regression_suite
from tools.capture_visual_baselines import validate_placements


LENS_CLEARANCE_M = CameraConfig().camera_collision_radius_m

#: The camera `facade_landmark` shipped with from 2026-08-03 until V2-3c. It
#: stood eighteen metres above the terrain inside a 49 m building, so every
#: capture and metric taken from it described the inside of a closed shell.
ENCLOSED_FACADE_CAMERA = (-650.0, 28.8274, 900.0)


def _quad(corners: list[tuple[float, float, float]]) -> list[list[float]]:
    a, b, c, d = corners
    return [
        [*vertex, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for vertex in (a, b, c, a, c, d)
    ]


def _box_building(
    x0: float = 0.0, x1: float = 10.0, z0: float = 0.0, z1: float = 10.0,
    height: float = 20.0,
) -> np.ndarray:
    """Walls and a flat roof, with no floor, like an extruded footprint."""

    vertices: list[list[float]] = []
    vertices += _quad([(x0, 0, z0), (x1, 0, z0), (x1, height, z0), (x0, height, z0)])
    vertices += _quad([(x1, 0, z0), (x1, 0, z1), (x1, height, z1), (x1, height, z0)])
    vertices += _quad([(x1, 0, z1), (x0, 0, z1), (x0, height, z1), (x1, height, z1)])
    vertices += _quad([(x0, 0, z1), (x0, 0, z0), (x0, height, z0), (x0, height, z1)])
    vertices += _quad(
        [(x0, height, z0), (x1, height, z0), (x1, height, z1), (x0, height, z1)]
    )
    return np.asarray(vertices, dtype=np.float32)


def test_a_camera_inside_a_building_has_its_roof_overhead() -> None:
    overhead = building_faces_overhead_m((3.0, 8.0, 6.0), _box_building())

    assert overhead == pytest.approx([12.0])


def test_a_ray_through_a_shared_triangle_edge_crosses_the_roof_once() -> None:
    # (5, 5) lies on the diagonal that splits the roof quad into two
    # triangles, so both report a hit at the same distance. That is one
    # crossing of one surface, not two roofs.
    overhead = building_faces_overhead_m((5.0, 8.0, 5.0), _box_building())

    assert overhead == pytest.approx([12.0])


def test_a_camera_beside_or_above_a_building_has_nothing_overhead() -> None:
    building = _box_building()

    assert len(building_faces_overhead_m((15.0, 8.0, 5.0), building)) == 0
    assert len(building_faces_overhead_m((5.0, 25.0, 5.0), building)) == 0


def test_nearest_face_is_measured_to_walls_and_roofs() -> None:
    building = _box_building()

    assert nearest_building_face_m((5.0, 8.0, 9.5), building) == pytest.approx(0.5)
    assert nearest_building_face_m((12.0, 8.0, 5.0), building) == pytest.approx(2.0)
    assert nearest_building_face_m((5.0, 23.0, 5.0), building) == pytest.approx(3.0)


def test_nearest_face_is_measured_to_an_edge_beyond_every_plane() -> None:
    # Off the roof corner, the closest point is the corner itself. A distance
    # to the nearest face *plane* would report 2 m here and pass a camera that
    # is really almost 5 m away — or, mirrored, pass one that is not.
    distance = nearest_building_face_m((12.0, 24.0, 12.0), _box_building())

    assert distance == pytest.approx(np.sqrt(4.0 + 16.0 + 4.0))


def test_placement_rejects_a_camera_inside_a_building() -> None:
    suite = load_visual_regression_suite()
    view = replace(suite.view("facade_landmark"), position_eus_m=(5.0, 8.0, 5.0))

    with pytest.raises(VisualViewError, match="inside a building"):
        validate_view_placement(view, _box_building(), LENS_CLEARANCE_M)


def test_placement_rejects_a_lens_touching_a_wall() -> None:
    suite = load_visual_regression_suite()
    view = replace(
        suite.view("facade_landmark"), position_eus_m=(10.05, 8.0, 5.0)
    )

    with pytest.raises(VisualViewError, match="lens clearance"):
        validate_view_placement(view, _box_building(), LENS_CLEARANCE_M)


def test_placement_reports_what_it_measured_for_a_clear_camera() -> None:
    suite = load_visual_regression_suite()
    view = replace(suite.view("facade_landmark"), position_eus_m=(14.0, 8.0, 5.0))

    placement = validate_view_placement(view, _box_building(), LENS_CLEARANCE_M)

    assert placement["building_faces_overhead_m"] == []
    assert placement["nearest_building_face_m"] == pytest.approx(4.0)
    assert placement["lens_clearance_m"] == LENS_CLEARANCE_M


def test_the_shipped_enclosed_facade_camera_is_rejected() -> None:
    suite = load_visual_regression_suite()
    scene = load_scene(suite.verify_scene_asset())
    view = replace(
        suite.view("facade_landmark"), position_eus_m=ENCLOSED_FACADE_CAMERA
    )

    with pytest.raises(VisualViewError, match="inside a building"):
        validate_view_placement(view, scene.building_vertices, LENS_CLEARANCE_M)


def test_every_canonical_camera_stands_clear_of_buildings() -> None:
    suite = load_visual_regression_suite()
    scene = load_scene(suite.verify_scene_asset())

    for view in suite.views:
        placement = validate_view_placement(
            view, scene.building_vertices, LENS_CLEARANCE_M
        )
        assert placement["building_faces_overhead_m"] == [], view.view_id


class _SlopedTerrain:
    """Ground that rises one metre per ten along x, so held clearance shows."""

    def height_at(self, x_m: float, z_m: float) -> float:
        return 0.1 * x_m


def test_advancing_leaves_the_building_toward_the_target() -> None:
    suite = load_visual_regression_suite()
    inside = replace(
        suite.view("facade_landmark"),
        position_eus_m=(4.0, 0.4 + 8.0, 5.0),
        target_eus_m=(100.0, 8.0, 5.0),
    )

    moved, advance_m = advance_out_of_buildings(
        inside, _box_building(), _SlopedTerrain(), LENS_CLEARANCE_M
    )

    # The wall is 6 m ahead; 6 m would put the lens on it, 7 m clears it.
    assert advance_m == 7
    assert moved.position_eus_m == pytest.approx((11.0, 1.1 + 8.0, 5.0))
    assert moved.target_eus_m == inside.target_eus_m


def test_advancing_holds_the_bearing_and_the_terrain_clearance() -> None:
    suite = load_visual_regression_suite()
    inside = replace(
        suite.view("facade_landmark"),
        position_eus_m=(5.0, 0.5 + 8.0, 4.0),
        target_eus_m=(65.0, 8.0, 84.0),
    )

    moved, advance_m = advance_out_of_buildings(
        inside, _box_building(), _SlopedTerrain(), LENS_CLEARANCE_M
    )

    # Heading (0.6, 0.8), the z = 10 wall is crossed at 7.5 m; 8 m clears it.
    assert advance_m == 8
    x_m, y_m, z_m = moved.position_eus_m
    assert (x_m - 5.0) * 80.0 == pytest.approx((z_m - 4.0) * 60.0, abs=1e-3)
    assert y_m - _SlopedTerrain().height_at(x_m, z_m) == pytest.approx(8.0)
    assert moved.yaw_deg == pytest.approx(inside.yaw_deg, abs=1e-3)


def test_advancing_a_camera_that_is_already_clear_moves_nothing() -> None:
    suite = load_visual_regression_suite()
    clear = replace(
        suite.view("facade_landmark"),
        position_eus_m=(20.0, 2.0 + 8.0, 5.0),
        target_eus_m=(100.0, 8.0, 5.0),
    )

    moved, advance_m = advance_out_of_buildings(
        clear, _box_building(), _SlopedTerrain(), LENS_CLEARANCE_M
    )

    assert advance_m == 0
    assert moved == clear


def test_advancing_gives_up_rather_than_walking_forever() -> None:
    suite = load_visual_regression_suite()
    buried = replace(
        suite.view("facade_landmark"),
        position_eus_m=(4.0, 0.4 + 8.0, 5.0),
        target_eus_m=(100.0, 8.0, 5.0),
    )

    with pytest.raises(VisualViewError, match="no clear position"):
        advance_out_of_buildings(
            buried,
            _box_building(x1=1000.0),
            _SlopedTerrain(),
            LENS_CLEARANCE_M,
            maximum_advance_m=50,
        )


def test_the_shipped_facade_camera_is_the_rule_applied_to_the_enclosed_one() -> None:
    suite = load_visual_regression_suite()
    scene = load_scene(suite.verify_scene_asset())
    terrain = TerrainSurface(
        scene.terrain_height_m,
        scene.terrain_bounds,
        scene.water_mask,
        scene.water_mask_bounds,
    )
    shipped = suite.view("facade_landmark")
    enclosed = replace(shipped, position_eus_m=ENCLOSED_FACADE_CAMERA)

    moved, advance_m = advance_out_of_buildings(
        enclosed, scene.building_vertices, terrain, LENS_CLEARANCE_M
    )

    assert advance_m == 36
    assert moved.position_eus_m == shipped.position_eus_m


def test_capture_refuses_a_suite_with_an_enclosed_camera() -> None:
    suite = load_visual_regression_suite()
    enclosed = tuple(
        replace(view, position_eus_m=ENCLOSED_FACADE_CAMERA)
        if view.view_id == "facade_landmark"
        else view
        for view in suite.views
    )

    with pytest.raises(VisualViewError, match="facade_landmark"):
        validate_placements(replace(suite, views=enclosed), enclosed)


def test_capture_records_placement_for_each_selected_view() -> None:
    suite = load_visual_regression_suite()
    selected = (suite.view("water_reflection"),)

    placements = validate_placements(suite, selected)

    assert list(placements) == ["water_reflection"]
    assert placements["water_reflection"]["nearest_building_face_m"] > 1.0
