from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from simulator.scene import StaticScene
from simulator.validation.terrain_contacts import (
    TerrainContactThresholds,
    audit_terrain_contacts,
    render_terrain_contact_map,
)


def _vertices(*positions: tuple[float, float, float]) -> np.ndarray:
    result = np.zeros((len(positions), 10), dtype=np.float32)
    result[:, :3] = positions
    result[:, 1] = 0.06
    result[:, 6] = 3.0
    return result


def _scene(
    height: np.ndarray,
    *,
    water: np.ndarray | None = None,
    roads: np.ndarray | None = None,
) -> StaticScene:
    height = np.asarray(height, dtype=np.float32)
    mask = (
        np.asarray(water, dtype=np.uint8) * 255
        if water is not None
        else np.zeros_like(height, dtype=np.uint8)
    )
    empty = np.empty((0, 10), dtype=np.float32)
    bounds = np.array((0.0, 0.0, 2.0, 2.0), dtype=np.float32)
    return StaticScene(
        building_vertices=empty,
        bridge_vertices=empty,
        road_vertices=roads if roads is not None else empty,
        vegetation_vertices=empty,
        detail_vertices=empty,
        water_mask=mask,
        water_mask_bounds=bounds,
        terrain_height_m=height,
        terrain_bounds=bounds,
        elevation_datum_m=0.0,
        origin_latitude_deg=37.0,
        origin_longitude_deg=126.0,
        snapshot_utc="2024-10-05T00:00:00Z",
    )


def test_a_planar_road_over_flat_terrain_has_no_contact_error() -> None:
    roads = _vertices((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    audit = audit_terrain_contacts(_scene(np.zeros((3, 3)), roads=roads))
    assert not audit.road_penetration.any()
    assert not audit.road_floating.any()
    assert audit.report["roads"]["deviation_p01_p50_p99_min_max_m"] == pytest.approx(
        [0.0] * 5, abs=1e-7
    )


def test_triangle_chord_error_finds_a_road_buried_by_an_interior_peak() -> None:
    terrain = np.zeros((3, 3), dtype=np.float32)
    terrain[1, 1] = 2.0
    roads = _vertices((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    audit = audit_terrain_contacts(_scene(terrain, roads=roads))
    assert audit.road_penetration.any()
    assert audit.road_deviation_m.min() == pytest.approx(-2.0)
    assert audit.report["roads"]["priority_deviation_sample_count"] >= 1
    assert audit.report["zones"]["priority_counts"]["P0"] >= 1


def test_high_land_next_to_the_water_datum_is_a_shoreline_warning() -> None:
    terrain = np.full((3, 3), 5.0, dtype=np.float32)
    terrain[:, 0] = 0.0
    water = np.zeros((3, 3), dtype=bool)
    water[:, 0] = True
    audit = audit_terrain_contacts(_scene(terrain, water=water))
    assert audit.report["shoreline"]["land_boundary_sample_count"] == 3
    assert audit.report["shoreline"]["warning_sample_count"] == 3
    assert not np.any(audit.slope_warning & audit.shoreline_warning)


def test_an_unclassified_road_sample_over_water_is_reported() -> None:
    water = np.ones((3, 3), dtype=bool)
    roads = _vertices((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    audit = audit_terrain_contacts(
        _scene(np.zeros((3, 3)), water=water, roads=roads)
    )
    assert audit.road_over_water.all()
    assert audit.report["roads"]["over_water_sample_count"] == 4
    assert audit.report["roads"]["priority_deviation_sample_count"] == 0


def test_error_map_has_one_pixel_per_terrain_sample_plus_a_legend(tmp_path) -> None:
    terrain = np.zeros((3, 3), dtype=np.float32)
    terrain[1, 1] = 2.0
    roads = _vertices((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    scene = _scene(terrain, roads=roads)
    audit = audit_terrain_contacts(scene)
    path = render_terrain_contact_map(scene, audit, tmp_path / "audit.jpg")
    assert path.suffix == ".png"
    with Image.open(path) as image:
        assert image.size == (3, 119)


def test_invalid_threshold_order_is_rejected() -> None:
    thresholds = TerrainContactThresholds(
        road_warning_deviation_m=0.3,
        road_priority_deviation_m=0.2,
    )
    with pytest.raises(ValueError, match="increasing"):
        audit_terrain_contacts(_scene(np.zeros((3, 3))), thresholds)
