"""Static terrain, shoreline and draped-road visual-contact audit.

This is a geometric defect detector, not a substitute for surveyed truth.  It
finds places where the shipped rendering representation can visibly separate:
steep height-field cells, unsupported water/land steps, and planar road
triangles that depart from the height field between their vertices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..scene import StaticScene
from ..terrain import sample_heightmap_array


@dataclass(frozen=True, slots=True)
class TerrainContactThresholds:
    """Metric thresholds used to triage visible representation errors."""

    slope_warning_deg: float = 38.0
    slope_priority_deg: float = 50.0
    slope_immediate_deg: float = 60.0
    shoreline_warning_height_m: float = 4.0
    shoreline_immediate_height_m: float = 8.0
    road_target_clearance_m: float = 0.06
    road_warning_deviation_m: float = 0.05
    road_priority_deviation_m: float = 0.20
    road_immediate_deviation_m: float = 0.50
    zone_size_m: float = 100.0

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
            raise ValueError("terrain-contact thresholds must be finite and positive")
        if not (
            self.slope_warning_deg
            < self.slope_priority_deg
            < self.slope_immediate_deg
            < 90.0
        ):
            raise ValueError("slope thresholds must be increasing and below 90 degrees")
        if self.shoreline_warning_height_m >= self.shoreline_immediate_height_m:
            raise ValueError("shoreline thresholds must be increasing")
        if not (
            self.road_warning_deviation_m
            < self.road_priority_deviation_m
            < self.road_immediate_deviation_m
        ):
            raise ValueError("road-deviation thresholds must be increasing")


@dataclass(slots=True)
class TerrainContactAudit:
    """Machine-readable summary plus the raster fields used to render a map."""

    report: dict[str, object]
    slope_deg: np.ndarray
    water: np.ndarray
    shoreline_warning: np.ndarray
    slope_warning: np.ndarray
    road_positions_xz_m: np.ndarray
    road_deviation_m: np.ndarray
    road_penetration: np.ndarray
    road_floating: np.ndarray
    road_over_water: np.ndarray


def _water_on_terrain(scene: StaticScene) -> np.ndarray:
    """Nearest-resample the water classification onto terrain texel centres."""

    terrain_rows, terrain_columns = scene.terrain_height_m.shape
    mask_rows, mask_columns = scene.water_mask.shape
    row_indices = np.rint(
        np.linspace(0, mask_rows - 1, terrain_rows, dtype=np.float64)
    ).astype(np.int32)
    column_indices = np.rint(
        np.linspace(0, mask_columns - 1, terrain_columns, dtype=np.float64)
    ).astype(np.int32)
    return scene.water_mask[np.ix_(row_indices, column_indices)] >= 128


def _sample_water(scene: StaticScene, positions_xz_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_xz_m, dtype=np.float64)
    rows, columns = scene.water_mask.shape
    bounds = np.asarray(scene.water_mask_bounds, dtype=np.float64)
    x = np.rint(
        np.clip((positions[:, 0] - bounds[0]) / (bounds[2] - bounds[0]), 0.0, 1.0)
        * (columns - 1)
    ).astype(np.int32)
    z = np.rint(
        np.clip((positions[:, 1] - bounds[1]) / (bounds[3] - bounds[1]), 0.0, 1.0)
        * (rows - 1)
    ).astype(np.int32)
    return scene.water_mask[z, x] >= 128


def _road_contact_samples(
    scene: StaticScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the actual GPU road triangles at centroids and edge midpoints."""

    vertices = np.asarray(scene.road_vertices, dtype=np.float64)
    if not len(vertices):
        empty_positions = np.empty((0, 2), dtype=np.float64)
        empty_values = np.empty(0, dtype=np.float64)
        empty_flags = np.empty(0, dtype=bool)
        return empty_positions, empty_values, empty_flags, empty_flags
    if len(vertices) % 3:
        raise ValueError("road vertex batch must contain complete triangles")
    triangles = vertices.reshape(-1, 3, vertices.shape[1])
    weights = np.asarray(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float64,
    )
    sample_positions = np.einsum(
        "sa,tac->tsc", weights, triangles[:, :, :3]
    ).reshape(-1, 3)
    terrain_at_vertices = sample_heightmap_array(
        scene.terrain_height_m,
        scene.terrain_bounds,
        triangles[:, :, [0, 2]].reshape(-1, 2),
    ).reshape(-1, 3)
    rendered_world_y = triangles[:, :, 1] + terrain_at_vertices
    rendered_sample_y = np.einsum(
        "sa,ta->ts", weights, rendered_world_y
    ).reshape(-1)
    terrain_at_samples = sample_heightmap_array(
        scene.terrain_height_m,
        scene.terrain_bounds,
        sample_positions[:, [0, 2]],
    )
    clearance_m = rendered_sample_y - terrain_at_samples
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    inside = (
        (sample_positions[:, 0] >= bounds[0])
        & (sample_positions[:, 0] <= bounds[2])
        & (sample_positions[:, 2] >= bounds[1])
        & (sample_positions[:, 2] <= bounds[3])
    )
    water = _sample_water(scene, sample_positions[:, [0, 2]])
    return sample_positions[:, [0, 2]], clearance_m, inside, water


def _grid_positions(scene: StaticScene) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = scene.terrain_height_m.shape
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    return (
        np.linspace(bounds[0], bounds[2], columns, dtype=np.float64),
        np.linspace(bounds[1], bounds[3], rows, dtype=np.float64),
    )


def _percentiles(values: np.ndarray, quantiles: tuple[float, ...]) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return []
    return np.percentile(finite, quantiles).tolist()


def _zone_key(
    x_m: float, z_m: float, bounds: np.ndarray, zone_size_m: float
) -> tuple[int, int]:
    count_x = max(1, int(math.ceil(float(bounds[2] - bounds[0]) / zone_size_m)))
    count_z = max(1, int(math.ceil(float(bounds[3] - bounds[1]) / zone_size_m)))
    index_x = int(math.floor((x_m - float(bounds[0])) / zone_size_m))
    index_z = int(math.floor((z_m - float(bounds[1])) / zone_size_m))
    return min(max(index_x, 0), count_x - 1), min(max(index_z, 0), count_z - 1)


def _build_zones(
    scene: StaticScene,
    thresholds: TerrainContactThresholds,
    slope_deg: np.ndarray,
    slope_warning: np.ndarray,
    shoreline_warning: np.ndarray,
    road_positions: np.ndarray,
    road_deviation: np.ndarray,
    road_penetration: np.ndarray,
    road_floating: np.ndarray,
    road_over_water: np.ndarray,
) -> list[dict[str, object]]:
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    x_coordinates, z_coordinates = _grid_positions(scene)
    zones: dict[tuple[int, int], dict[str, object]] = {}

    def zone_for(x_m: float, z_m: float) -> dict[str, object]:
        key = _zone_key(x_m, z_m, bounds, thresholds.zone_size_m)
        if key not in zones:
            minimum_x = float(bounds[0] + key[0] * thresholds.zone_size_m)
            minimum_z = float(bounds[1] + key[1] * thresholds.zone_size_m)
            zones[key] = {
                "zone_id": f"x{key[0]:02d}_z{key[1]:02d}",
                "bounds_eus_m": [
                    minimum_x,
                    minimum_z,
                    minimum_x + thresholds.zone_size_m,
                    minimum_z + thresholds.zone_size_m,
                ],
                "slope_sample_count": 0,
                "maximum_slope_deg": 0.0,
                "shoreline_sample_count": 0,
                "maximum_shoreline_height_m": 0.0,
                "road_penetration_sample_count": 0,
                "road_floating_sample_count": 0,
                "road_over_water_sample_count": 0,
                "maximum_road_deviation_m": 0.0,
            }
        return zones[key]

    for row, column in np.argwhere(slope_warning):
        zone = zone_for(x_coordinates[column], z_coordinates[row])
        zone["slope_sample_count"] += 1
        zone["maximum_slope_deg"] = max(
            zone["maximum_slope_deg"], float(slope_deg[row, column])
        )
    heights = np.asarray(scene.terrain_height_m, dtype=np.float64)
    for row, column in np.argwhere(shoreline_warning):
        zone = zone_for(x_coordinates[column], z_coordinates[row])
        zone["shoreline_sample_count"] += 1
        zone["maximum_shoreline_height_m"] = max(
            zone["maximum_shoreline_height_m"], float(heights[row, column])
        )
    road_issue = road_penetration | road_floating | road_over_water
    for index in np.flatnonzero(road_issue):
        x_m, z_m = road_positions[index]
        zone = zone_for(float(x_m), float(z_m))
        if road_penetration[index]:
            zone["road_penetration_sample_count"] += 1
        if road_floating[index]:
            zone["road_floating_sample_count"] += 1
        if road_over_water[index]:
            zone["road_over_water_sample_count"] += 1
        zone["maximum_road_deviation_m"] = max(
            zone["maximum_road_deviation_m"], abs(float(road_deviation[index]))
        )

    def priority(zone: dict[str, object]) -> str:
        if (
            zone["maximum_road_deviation_m"]
            >= thresholds.road_immediate_deviation_m
            or zone["maximum_shoreline_height_m"]
            >= thresholds.shoreline_immediate_height_m
            or zone["maximum_slope_deg"] >= thresholds.slope_immediate_deg
            or zone["road_over_water_sample_count"] > 0
        ):
            return "P0"
        if (
            zone["maximum_road_deviation_m"]
            >= thresholds.road_priority_deviation_m
            or zone["maximum_shoreline_height_m"]
            >= thresholds.shoreline_warning_height_m
            or zone["maximum_slope_deg"] >= thresholds.slope_priority_deg
        ):
            return "P1"
        return "P2"

    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    output = []
    for zone in zones.values():
        zone["priority"] = priority(zone)
        output.append(zone)
    output.sort(
        key=lambda zone: (
            priority_order[zone["priority"]],
            -zone["maximum_road_deviation_m"],
            -zone["maximum_shoreline_height_m"],
            -zone["maximum_slope_deg"],
            zone["zone_id"],
        )
    )
    return output


def audit_terrain_contacts(
    scene: StaticScene,
    thresholds: TerrainContactThresholds = TerrainContactThresholds(),
) -> TerrainContactAudit:
    """Audit visible terrain contacts in the exact shipped static scene."""

    thresholds.validate()
    height = np.asarray(scene.terrain_height_m, dtype=np.float64)
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    rows, columns = height.shape
    spacing_x_m = float(bounds[2] - bounds[0]) / max(columns - 1, 1)
    spacing_z_m = float(bounds[3] - bounds[1]) / max(rows - 1, 1)
    gradient_z, gradient_x = np.gradient(height, spacing_z_m, spacing_x_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gradient_x, gradient_z)))
    water = _water_on_terrain(scene)
    land = ~water
    adjacent_water = np.zeros_like(water)
    adjacent_water[1:] |= water[:-1]
    adjacent_water[:-1] |= water[1:]
    adjacent_water[:, 1:] |= water[:, :-1]
    adjacent_water[:, :-1] |= water[:, 1:]
    shoreline = land & adjacent_water
    # Shore cells are reported separately; counting their deliberate datum
    # transition as a generic terrain slope would duplicate the same defect.
    slope_warning = (
        land & ~shoreline & (slope_deg >= thresholds.slope_warning_deg)
    )
    shoreline_warning = shoreline & (
        height >= thresholds.shoreline_warning_height_m
    )

    road_positions, road_clearance, road_inside, road_water = _road_contact_samples(
        scene
    )
    road_deviation = road_clearance - thresholds.road_target_clearance_m
    valid_land_road = road_inside & ~road_water
    road_penetration = valid_land_road & (
        road_deviation <= -thresholds.road_warning_deviation_m
    )
    road_floating = valid_land_road & (
        road_deviation >= thresholds.road_warning_deviation_m
    )
    road_over_water = road_inside & road_water

    zones = _build_zones(
        scene,
        thresholds,
        slope_deg,
        slope_warning,
        shoreline_warning,
        road_positions,
        road_deviation,
        road_penetration,
        road_floating,
        road_over_water,
    )
    finite_road = road_deviation[valid_land_road]
    report: dict[str, object] = {
        "schema_version": 1,
        "coordinate_system": "local East-Up-South metres",
        "classification": "engineering visual-defect triage; not surveyed truth",
        "thresholds": asdict(thresholds),
        "terrain": {
            "shape": [int(rows), int(columns)],
            "bounds_eus_m": bounds.tolist(),
            "spacing_xz_m": [spacing_x_m, spacing_z_m],
            "height_min_max_m": [float(height.min()), float(height.max())],
            "land_slope_p50_p95_p99_max_deg": _percentiles(
                slope_deg[land], (50, 95, 99, 100)
            ),
            "slope_warning_sample_count": int(slope_warning.sum()),
        },
        "shoreline": {
            "land_boundary_sample_count": int(shoreline.sum()),
            "warning_sample_count": int(shoreline_warning.sum()),
            "height_p50_p95_p99_max_m": _percentiles(
                height[shoreline], (50, 95, 99, 100)
            ),
        },
        "roads": {
            "sample_count": int(len(road_deviation)),
            "outside_terrain_sample_count": int((~road_inside).sum()),
            "over_water_sample_count": int(road_over_water.sum()),
            "penetration_sample_count": int(road_penetration.sum()),
            "floating_sample_count": int(road_floating.sum()),
            "deviation_p01_p50_p99_min_max_m": (
                [
                    float(np.percentile(finite_road, 1)),
                    float(np.percentile(finite_road, 50)),
                    float(np.percentile(finite_road, 99)),
                    float(finite_road.min()),
                    float(finite_road.max()),
                ]
                if len(finite_road)
                else []
            ),
            "priority_deviation_sample_count": int(
                (
                    valid_land_road
                    & (np.abs(road_deviation) >= thresholds.road_priority_deviation_m)
                ).sum()
            ),
        },
        "zones": {
            "zone_size_m": thresholds.zone_size_m,
            "affected_zone_count": len(zones),
            "priority_counts": {
                key: sum(zone["priority"] == key for zone in zones)
                for key in ("P0", "P1", "P2")
            },
            "ordered": zones,
        },
    }
    return TerrainContactAudit(
        report=report,
        slope_deg=slope_deg,
        water=water,
        shoreline_warning=shoreline_warning,
        slope_warning=slope_warning,
        road_positions_xz_m=road_positions,
        road_deviation_m=road_deviation,
        road_penetration=road_penetration,
        road_floating=road_floating,
        road_over_water=road_over_water,
    )


def render_terrain_contact_map(
    scene: StaticScene, audit: TerrainContactAudit, path: Path
) -> Path:
    """Render an evidence map whose raster aligns with terrain texel centres."""

    height = np.asarray(scene.terrain_height_m, dtype=np.float64)
    land_values = height[~audit.water]
    if not len(land_values):
        land_values = height.reshape(-1)
    low, high = np.percentile(land_values, [2, 98])
    normalized = np.clip((height - low) / max(high - low, 1e-6), 0.0, 1.0)
    grey = np.asarray(35.0 + normalized * 150.0, dtype=np.uint8)
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    rgb[audit.water] = (12, 31, 51)

    def overlay(mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> None:
        base = rgb[mask].astype(np.float32)
        rgb[mask] = np.asarray(
            base * (1.0 - alpha) + np.asarray(colour) * alpha,
            dtype=np.uint8,
        )

    overlay(audit.slope_warning, (255, 145, 35), 0.72)
    overlay(audit.shoreline_warning, (229, 42, 122), 0.90)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    width, height_px = image.size

    def pixel(position: np.ndarray) -> tuple[int, int]:
        x = int(
            round((position[0] - bounds[0]) / (bounds[2] - bounds[0]) * (width - 1))
        )
        z = int(
            round(
                (position[1] - bounds[1])
                / (bounds[3] - bounds[1])
                * (height_px - 1)
            )
        )
        return x, z

    for flags, colour in (
        (audit.road_penetration, (255, 40, 30)),
        (audit.road_floating, (40, 235, 255)),
        (audit.road_over_water, (105, 90, 255)),
    ):
        for position in audit.road_positions_xz_m[flags]:
            x, z = pixel(position)
            draw.rectangle((x - 1, z - 1, x + 1, z + 1), fill=colour)

    legend_height = 116
    canvas = Image.new("RGB", (width, height_px + legend_height), (15, 17, 20))
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    items = (
        ("steep land slope", (255, 145, 35)),
        ("unsupported shoreline step", (229, 42, 122)),
        ("road penetration", (255, 40, 30)),
        ("road floating", (40, 235, 255)),
        ("unclassified road over water", (105, 90, 255)),
    )
    x, y = 18, height_px + 16
    for label, colour in items:
        legend.rectangle((x, y, x + 14, y + 14), fill=colour)
        legend.text((x + 21, y + 1), label, fill=(230, 232, 235))
        x += 194
    zones = audit.report["zones"]
    roads = audit.report["roads"]
    legend.text(
        (18, height_px + 51),
        (
            f"P0/P1/P2 zones: {zones['priority_counts']['P0']}/"
            f"{zones['priority_counts']['P1']}/{zones['priority_counts']['P2']}   "
            f"road penetrate/float/water: {roads['penetration_sample_count']}/"
            f"{roads['floating_sample_count']}/{roads['over_water_sample_count']}"
        ),
        fill=(230, 232, 235),
    )
    legend.text(
        (18, height_px + 76),
        "North is up; coordinates are local East-South. "
        "Triage map, not survey evidence.",
        fill=(165, 172, 180),
    )
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return path
