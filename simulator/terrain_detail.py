"""Evidence-scoped terrain-conforming detail for static linear meshes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .terrain import sample_heightmap_array


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIORITY_AREAS_PATH = REPOSITORY_ROOT / "assets" / "terrain_priority_areas.json"


@dataclass(frozen=True, slots=True)
class AdaptiveTessellationStats:
    input_triangles: int
    output_triangles: int
    refined_triangles_by_level: tuple[int, ...]
    maximum_subdivision_level: int
    warning_deviation_m: float
    area_bounds_xz_m: tuple[float, float, float, float]

    @property
    def triangle_multiplier(self) -> float:
        return self.output_triangles / max(self.input_triangles, 1)


def load_priority_area_bounds(
    path: Path = DEFAULT_PRIORITY_AREAS_PATH,
    area_id: str = "event_park_north_bank",
) -> tuple[float, float, float, float]:
    """Load only the small runtime contract needed by the scene pass."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("terrain priority-area schema_version must be 1")
    for area in payload.get("areas", []):
        if area.get("area_id") != area_id:
            continue
        bounds = tuple(float(value) for value in area.get("bounds_xz_m", []))
        if len(bounds) != 4 or not (
            bounds[2] > bounds[0] and bounds[3] > bounds[1]
        ):
            raise ValueError("priority area has invalid bounds_xz_m")
        return bounds
    raise KeyError(area_id)


def _sample_water(
    water_mask: np.ndarray,
    water_bounds: np.ndarray,
    positions_xz_m: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions_xz_m, dtype=np.float64)
    mask = np.asarray(water_mask, dtype=np.uint8)
    bounds = np.asarray(water_bounds, dtype=np.float64)
    rows, columns = mask.shape
    x = np.rint(
        np.clip((positions[:, 0] - bounds[0]) / (bounds[2] - bounds[0]), 0.0, 1.0)
        * (columns - 1)
    ).astype(np.int32)
    z = np.rint(
        np.clip((positions[:, 1] - bounds[1]) / (bounds[3] - bounds[1]), 0.0, 1.0)
        * (rows - 1)
    ).astype(np.int32)
    return mask[z, x] >= 128


def _refinement_mask(
    triangles: np.ndarray,
    terrain_height_m: np.ndarray,
    terrain_bounds: np.ndarray,
    water_mask: np.ndarray,
    water_bounds: np.ndarray,
    area_bounds_xz_m: tuple[float, float, float, float],
    warning_deviation_m: float,
) -> np.ndarray:
    positions = triangles[:, :, [0, 2]].astype(np.float64)
    area = np.asarray(area_bounds_xz_m, dtype=np.float64)
    minimum = positions.min(axis=1)
    maximum = positions.max(axis=1)
    overlaps_area = (
        (maximum[:, 0] >= area[0])
        & (minimum[:, 0] <= area[2])
        & (maximum[:, 1] >= area[1])
        & (minimum[:, 1] <= area[3])
    )
    candidate_indices = np.flatnonzero(overlaps_area)
    refine = np.zeros(len(triangles), dtype=bool)
    if not len(candidate_indices):
        return refine
    candidates = triangles[candidate_indices]
    candidate_positions = positions[candidate_indices]
    weights = np.asarray(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.5, 0.5, 0.0),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
        ),
        dtype=np.float64,
    )
    samples = np.einsum(
        "sa,tac->tsc", weights, candidate_positions
    ).reshape(-1, 2)
    terrain_at_vertices = sample_heightmap_array(
        terrain_height_m,
        terrain_bounds,
        candidate_positions.reshape(-1, 2),
    ).reshape(-1, 3)
    rendered_world_y = terrain_at_vertices + candidates[:, :, 1]
    rendered_sample_y = np.einsum(
        "sa,ta->ts", weights, rendered_world_y
    ).reshape(-1)
    terrain_at_samples = sample_heightmap_array(
        terrain_height_m, terrain_bounds, samples
    )
    target_offset = np.einsum(
        "sa,ta->ts", weights, candidates[:, :, 1]
    ).reshape(-1)
    deviation = rendered_sample_y - terrain_at_samples - target_offset
    bounds = np.asarray(terrain_bounds, dtype=np.float64)
    inside = (
        (samples[:, 0] >= bounds[0])
        & (samples[:, 0] <= bounds[2])
        & (samples[:, 1] >= bounds[1])
        & (samples[:, 1] <= bounds[3])
    )
    land = ~_sample_water(water_mask, water_bounds, samples)
    visible_error = (
        inside & land & (np.abs(deviation) >= warning_deviation_m)
    ).reshape(-1, len(weights))
    refine[candidate_indices] = np.any(visible_error, axis=1)
    return refine


def _subdivide_attributes(triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, bc, ca = (a + b) * 0.5, (b + c) * 0.5, (c + a) * 0.5
    return np.stack(
        (
            np.stack((a, ab, ca), axis=1),
            np.stack((ab, b, bc), axis=1),
            np.stack((ca, bc, c), axis=1),
            np.stack((ab, bc, ca), axis=1),
        ),
        axis=1,
    ).reshape(-1, 3, triangles.shape[2])


def adaptive_terrain_tessellate(
    vertices: np.ndarray,
    terrain_height_m: np.ndarray,
    terrain_bounds: np.ndarray,
    water_mask: np.ndarray,
    water_bounds: np.ndarray,
    area_bounds_xz_m: tuple[float, float, float, float],
    *,
    warning_deviation_m: float = 0.05,
    maximum_subdivision_level: int = 2,
) -> tuple[np.ndarray, AdaptiveTessellationStats]:
    """Refine only land-road triangles whose rendered chord visibly departs.

    Midpoints interpolate all existing attributes, including continuous UVs,
    while the vertex shader samples the unchanged terrain height at every new
    position.  Water samples and triangles outside the evidence-scoped area
    are deliberately left untouched.
    """

    source = np.asarray(vertices, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 10 or len(source) % 3:
        raise ValueError("terrain tessellation requires complete 10-channel triangles")
    if warning_deviation_m <= 0.0:
        raise ValueError("warning_deviation_m must be positive")
    if maximum_subdivision_level < 0 or maximum_subdivision_level > 3:
        raise ValueError("maximum_subdivision_level must be between zero and three")
    active = source.reshape(-1, 3, 10)
    finished: list[np.ndarray] = []
    refined_counts: list[int] = []
    for _ in range(maximum_subdivision_level):
        refine = _refinement_mask(
            active,
            terrain_height_m,
            terrain_bounds,
            water_mask,
            water_bounds,
            area_bounds_xz_m,
            warning_deviation_m,
        )
        refined_counts.append(int(refine.sum()))
        if np.any(~refine):
            finished.append(active[~refine])
        if not np.any(refine):
            active = np.empty((0, 3, 10), dtype=np.float32)
            break
        active = _subdivide_attributes(active[refine]).astype(np.float32)
    if len(active):
        finished.append(active)
    output_triangles = (
        np.concatenate(finished, axis=0)
        if finished
        else np.empty((0, 3, 10), dtype=np.float32)
    )
    output = np.ascontiguousarray(output_triangles.reshape(-1, 10))
    stats = AdaptiveTessellationStats(
        input_triangles=len(source) // 3,
        output_triangles=len(output) // 3,
        refined_triangles_by_level=tuple(refined_counts),
        maximum_subdivision_level=maximum_subdivision_level,
        warning_deviation_m=warning_deviation_m,
        area_bounds_xz_m=tuple(float(value) for value in area_bounds_xz_m),
    )
    return output, stats
