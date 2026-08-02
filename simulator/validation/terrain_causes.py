"""Separate road tessellation error from steep-source terrain risk."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw

from ..scene import StaticScene
from ..terrain import sample_heightmap_array
from .terrain_contacts import TerrainContactAudit, TerrainContactThresholds


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIORITY_AREAS_PATH = REPOSITORY_ROOT / "assets" / "terrain_priority_areas.json"


@dataclass(frozen=True, slots=True)
class TerrainPriorityArea:
    area_id: str
    bounds_xz_m: tuple[float, float, float, float]
    confidence_grade: str
    purpose: str
    derivation: dict[str, object]
    scene_asset: str
    scene_asset_sha256: str

    def contains(self, positions_xz_m: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_xz_m, dtype=np.float64)
        minimum_x, minimum_z, maximum_x, maximum_z = self.bounds_xz_m
        return (
            (positions[:, 0] >= minimum_x)
            & (positions[:, 0] <= maximum_x)
            & (positions[:, 1] >= minimum_z)
            & (positions[:, 1] <= maximum_z)
        )

    def verify_scene_asset(self, repository_root: Path = REPOSITORY_ROOT) -> Path:
        path = repository_root / self.scene_asset
        return self.verify_scene_path(path)

    def verify_scene_path(self, path: Path) -> Path:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"priority-area scene asset is missing: {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != self.scene_asset_sha256:
            raise ValueError(
                f"priority-area scene checksum {digest} != {self.scene_asset_sha256}"
            )
        return path


@dataclass(slots=True)
class TerrainCauseAnalysis:
    report: dict[str, object]
    current_priority_positions_xz_m: np.ndarray
    residual_priority_positions_xz_m: np.ndarray
    north_shoreline: np.ndarray
    north_shoreline_warning: np.ndarray


def load_priority_area(
    path: Path = DEFAULT_PRIORITY_AREAS_PATH,
    area_id: str = "event_park_north_bank",
) -> TerrainPriorityArea:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("terrain priority-area schema_version must be 1")
    digest = str(payload.get("scene_asset_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("priority-area scene checksum must be lowercase SHA-256")
    for candidate in payload.get("areas", []):
        if candidate.get("area_id") != area_id:
            continue
        bounds = tuple(float(value) for value in candidate.get("bounds_xz_m", []))
        if len(bounds) != 4 or not (
            bounds[2] > bounds[0] and bounds[3] > bounds[1]
        ):
            raise ValueError(
                "priority-area bounds must be [min_x, min_z, max_x, max_z]"
            )
        if candidate.get("confidence_grade") not in {"C", "D"}:
            raise ValueError("project-defined priority areas must remain grade C or D")
        purpose = str(candidate.get("purpose", "")).strip()
        derivation = candidate.get("derivation")
        if not purpose or not isinstance(derivation, dict):
            raise ValueError("priority area requires purpose and derivation metadata")
        return TerrainPriorityArea(
            area_id=area_id,
            bounds_xz_m=bounds,
            confidence_grade=str(candidate["confidence_grade"]),
            purpose=purpose,
            derivation=derivation,
            scene_asset=str(payload.get("scene_asset", "")),
            scene_asset_sha256=digest,
        )
    raise KeyError(area_id)


def _subdivide_triangles(
    positions: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = positions[:, 0], positions[:, 1], positions[:, 2]
    offset_a, offset_b, offset_c = offsets[:, 0], offsets[:, 1], offsets[:, 2]
    ab, bc, ca = (a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5
    offset_ab = (offset_a + offset_b) * 0.5
    offset_bc = (offset_b + offset_c) * 0.5
    offset_ca = (offset_c + offset_a) * 0.5
    children = np.stack(
        (
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3, 2)
    child_offsets = np.stack(
        (
            np.stack((offset_a, offset_ab, offset_ca), axis=1),
            np.stack((offset_ab, offset_b, offset_bc), axis=1),
            np.stack((offset_ca, offset_bc, offset_c), axis=1),
            np.stack((offset_ab, offset_bc, offset_ca), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3)
    return children, child_offsets


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


def iter_tessellated_road_samples(
    scene: StaticScene,
    subdivision_levels: int,
    *,
    triangle_chunk_size: int = 2048,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield road contact samples after counterfactual midpoint subdivision."""

    if subdivision_levels < 0 or subdivision_levels > 3:
        raise ValueError("road subdivision_levels must be between zero and three")
    vertices = np.asarray(scene.road_vertices, dtype=np.float64)
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
    terrain_bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    for start in range(0, len(triangles), triangle_chunk_size):
        batch = triangles[start : start + triangle_chunk_size]
        positions = batch[:, :, [0, 2]].copy()
        offsets = batch[:, :, 1].copy()
        for _ in range(subdivision_levels):
            positions, offsets = _subdivide_triangles(positions, offsets)
        terrain_at_vertices = sample_heightmap_array(
            scene.terrain_height_m,
            scene.terrain_bounds,
            positions.reshape(-1, 2),
        ).reshape(-1, 3)
        world_y = terrain_at_vertices + offsets
        samples = np.einsum("sa,tac->tsc", weights, positions).reshape(-1, 2)
        rendered_y = np.einsum("sa,ta->ts", weights, world_y).reshape(-1)
        terrain_y = sample_heightmap_array(
            scene.terrain_height_m, scene.terrain_bounds, samples
        )
        clearance = rendered_y - terrain_y
        inside = (
            (samples[:, 0] >= terrain_bounds[0])
            & (samples[:, 0] <= terrain_bounds[2])
            & (samples[:, 1] >= terrain_bounds[1])
            & (samples[:, 1] <= terrain_bounds[3])
        )
        water = _sample_water(scene, samples)
        yield samples, clearance, inside, water


def _sample_raster(
    raster: np.ndarray, bounds: np.ndarray, positions_xz_m: np.ndarray
) -> np.ndarray:
    return sample_heightmap_array(raster, bounds, positions_xz_m)


def _level_metrics(
    scene: StaticScene,
    area: TerrainPriorityArea,
    audit: TerrainContactAudit,
    thresholds: TerrainContactThresholds,
    level: int,
) -> tuple[dict[str, object], np.ndarray]:
    all_deviations: list[np.ndarray] = []
    area_deviations: list[np.ndarray] = []
    area_priority_positions: list[np.ndarray] = []
    area_priority_slopes: list[np.ndarray] = []
    for positions, clearance, inside, water in iter_tessellated_road_samples(
        scene, level
    ):
        deviation = clearance - thresholds.road_target_clearance_m
        valid = inside & ~water
        in_area = valid & area.contains(positions)
        all_deviations.append(deviation[valid])
        area_deviations.append(deviation[in_area])
        priority = in_area & (
            np.abs(deviation) >= thresholds.road_priority_deviation_m
        )
        if np.any(priority):
            selected = positions[priority]
            area_priority_positions.append(selected)
            area_priority_slopes.append(
                _sample_raster(audit.slope_deg, scene.terrain_bounds, selected)
            )
    all_values = np.concatenate(all_deviations) if all_deviations else np.empty(0)
    area_values = np.concatenate(area_deviations) if area_deviations else np.empty(0)
    priority_positions = (
        np.concatenate(area_priority_positions)
        if area_priority_positions
        else np.empty((0, 2), dtype=np.float64)
    )
    priority_slopes = (
        np.concatenate(area_priority_slopes)
        if area_priority_slopes
        else np.empty(0, dtype=np.float64)
    )

    def summarize(values: np.ndarray) -> dict[str, object]:
        absolute = np.abs(values)
        if not len(values):
            return {
                "sample_count": 0,
                "warning_fraction": 0.0,
                "priority_fraction": 0.0,
                "immediate_fraction": 0.0,
                "absolute_p95_p99_max_m": [],
            }
        return {
            "sample_count": int(len(values)),
            "warning_fraction": float(
                np.mean(absolute >= thresholds.road_warning_deviation_m)
            ),
            "priority_fraction": float(
                np.mean(absolute >= thresholds.road_priority_deviation_m)
            ),
            "immediate_fraction": float(
                np.mean(absolute >= thresholds.road_immediate_deviation_m)
            ),
            "absolute_p95_p99_max_m": np.percentile(
                absolute, [95, 99, 100]
            ).tolist(),
        }

    metrics = {
        "subdivision_level": level,
        "triangle_multiplier": 4**level,
        "whole_scene": summarize(all_values),
        "priority_area": summarize(area_values),
        "priority_area_road_source_slope": {
            "priority_sample_count": int(len(priority_slopes)),
            "moderate_source_slope_sample_count": int(
                np.sum(priority_slopes < thresholds.slope_warning_deg)
            ),
            "steep_source_slope_sample_count": int(
                np.sum(priority_slopes >= thresholds.slope_warning_deg)
            ),
        },
    }
    return metrics, priority_positions


def analyze_terrain_causes(
    scene: StaticScene,
    audit: TerrainContactAudit,
    area: TerrainPriorityArea,
    thresholds: TerrainContactThresholds = TerrainContactThresholds(),
    maximum_subdivision_level: int = 2,
) -> TerrainCauseAnalysis:
    """Measure how much road error survives geometry-only refinement."""

    if maximum_subdivision_level < 1 or maximum_subdivision_level > 3:
        raise ValueError("maximum_subdivision_level must be between one and three")
    levels = []
    priority_positions = []
    for level in range(maximum_subdivision_level + 1):
        metrics, positions = _level_metrics(scene, area, audit, thresholds, level)
        levels.append(metrics)
        priority_positions.append(positions)

    rows, columns = scene.terrain_height_m.shape
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    x = np.linspace(bounds[0], bounds[2], columns)
    z = np.linspace(bounds[1], bounds[3], rows)
    xx, zz = np.meshgrid(x, z)
    area_mask = (
        (xx >= area.bounds_xz_m[0])
        & (xx <= area.bounds_xz_m[2])
        & (zz >= area.bounds_xz_m[1])
        & (zz <= area.bounds_xz_m[3])
    )
    land = ~audit.water
    water_north = np.zeros_like(audit.water)
    water_north[1:] = audit.water[:-1]
    north_shoreline = area_mask & land & water_north
    north_warning = north_shoreline & (
        scene.terrain_height_m >= thresholds.shoreline_warning_height_m
    )
    first_fraction = levels[0]["priority_area"]["priority_fraction"]
    last_fraction = levels[-1]["priority_area"]["priority_fraction"]
    reduction = (
        1.0 - last_fraction / first_fraction if first_fraction > 0.0 else 0.0
    )
    report = {
        "schema_version": 1,
        "classification": (
            "counterfactual rendering-geometry test; does not alter or validate "
            "surveyed elevations"
        ),
        "priority_area": {
            "area_id": area.area_id,
            "bounds_xz_m": list(area.bounds_xz_m),
            "confidence_grade": area.confidence_grade,
            "purpose": area.purpose,
            "derivation": area.derivation,
        },
        "road_tessellation_counterfactual": {
            "method": (
                "recursively split every road triangle at edge midpoints; each new "
                "vertex samples the unchanged terrain height field"
            ),
            "levels": levels,
            "priority_fraction_reduction_at_final_level": reduction,
            "interpretation": (
                "errors removed by subdivision are representation-limited; residual "
                "errors on steep slopes or sharp grid curvature require terrain/road "
                "evidence review"
            ),
        },
        "north_shoreline": {
            "definition": (
                "land texels inside the priority area with a water texel "
                "immediately north"
            ),
            "sample_count": int(north_shoreline.sum()),
            "warning_height_sample_count": int(north_warning.sum()),
            "height_p50_p95_p99_max_m": (
                np.percentile(
                    scene.terrain_height_m[north_shoreline], [50, 95, 99, 100]
                ).tolist()
                if np.any(north_shoreline)
                else []
            ),
        },
    }
    return TerrainCauseAnalysis(
        report=report,
        current_priority_positions_xz_m=priority_positions[0],
        residual_priority_positions_xz_m=priority_positions[-1],
        north_shoreline=north_shoreline,
        north_shoreline_warning=north_warning,
    )


def render_terrain_cause_map(
    scene: StaticScene,
    audit: TerrainContactAudit,
    area: TerrainPriorityArea,
    analysis: TerrainCauseAnalysis,
    path: Path,
) -> Path:
    """Render the AOI and errors that survive two geometry subdivisions."""

    height = np.asarray(scene.terrain_height_m, dtype=np.float64)
    land_values = height[~audit.water]
    low, high = np.percentile(land_values, [2, 98])
    grey = np.asarray(
        32.0 + np.clip((height - low) / max(high - low, 1e-6), 0.0, 1.0) * 145.0,
        dtype=np.uint8,
    )
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    rgb[audit.water] = (12, 31, 51)
    rgb[analysis.north_shoreline] = (90, 210, 130)
    rgb[analysis.north_shoreline_warning] = (235, 55, 145)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    width, height_px = image.size

    def pixel(position: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        return (
            int(
                round(
                    (position[0] - bounds[0])
                    / (bounds[2] - bounds[0])
                    * (width - 1)
                )
            ),
            int(
                round(
                    (position[1] - bounds[1])
                    / (bounds[3] - bounds[1])
                    * (height_px - 1)
                )
            ),
        )

    minimum = pixel((area.bounds_xz_m[0], area.bounds_xz_m[1]))
    maximum = pixel((area.bounds_xz_m[2], area.bounds_xz_m[3]))
    draw.rectangle((*minimum, *maximum), outline=(245, 245, 245), width=2)
    for positions, colour in (
        (analysis.current_priority_positions_xz_m, (255, 55, 35)),
        (analysis.residual_priority_positions_xz_m, (255, 210, 35)),
    ):
        for position in positions:
            x, z = pixel(position)
            draw.point((x, z), fill=colour)

    legend_height = 94
    canvas = Image.new("RGB", (width, height_px + legend_height), (15, 17, 20))
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    items = (
        ("AOI", (245, 245, 245)),
        ("north shoreline", (90, 210, 130)),
        ("shoreline >= 4 m", (235, 55, 145)),
        ("current road >= 0.2 m", (255, 55, 35)),
        ("residual after L2", (255, 210, 35)),
    )
    x, y = 18, height_px + 16
    for label, colour in items:
        legend.rectangle((x, y, x + 13, y + 13), fill=colour)
        legend.text((x + 19, y), label, fill=(230, 232, 235))
        x += 190
    reduction = analysis.report["road_tessellation_counterfactual"][
        "priority_fraction_reduction_at_final_level"
    ]
    legend.text(
        (18, height_px + 50),
        f"L2 priority-error fraction reduction in AOI: {reduction * 100.0:.2f}%",
        fill=(230, 232, 235),
    )
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return path
