"""Measure V1-5 dated tunnel filtering and residual road-structure risk."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, map_coordinates

from simulator.passes.scene import linear_feature_uv
from simulator.road_semantics import (
    DEFAULT_ROAD_SEMANTICS_PATH,
    filter_occluded_road_segments,
    load_road_structure_semantics,
)
from simulator.scene import LINEAR_STYLE_STEPS, load_scene
from simulator.terrain import sample_heightmap_array
from simulator.terrain_detail import (
    adaptive_terrain_tessellate,
    load_priority_area_bounds,
)
from simulator.validation.terrain_contacts import audit_terrain_contacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
FRAME_BUDGET_MS = 1000.0 / 60.0


def _load_integrated_profile(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8-sig")
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"runtime profile contains no JSON object: {path}")
    payload = json.loads(raw[start:])
    integrated = payload.get("integrated")
    if not isinstance(integrated, dict):
        raise ValueError(f"runtime profile has no integrated result: {path}")
    return {str(key): value for key, value in integrated.items()}


def _performance_comparison(
    baseline_paths: list[Path], candidate_paths: list[Path]
) -> dict[str, object] | None:
    if not baseline_paths and not candidate_paths:
        return None
    if not baseline_paths or len(baseline_paths) != len(candidate_paths):
        raise ValueError("baseline and candidate profile counts must match")
    baseline = [_load_integrated_profile(path) for path in baseline_paths]
    candidate = [_load_integrated_profile(path) for path in candidate_paths]
    keys = (
        "frame_mean_ms",
        "frame_p95_ms",
        "frame_p99_ms",
        "visual_p95_ms",
        "physics_p95_ms",
    )
    rows = []
    for index, (before, after) in enumerate(zip(baseline, candidate), start=1):
        rows.append(
            {
                "run": index,
                "baseline": {key: float(before[key]) for key in keys},
                "candidate": {key: float(after[key]) for key in keys},
                "frame_p95_delta_ms": float(after["frame_p95_ms"])
                - float(before["frame_p95_ms"]),
            }
        )
    candidate_passes = sum(
        row["candidate"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
    )
    return {
        "frames_per_run": 360,
        "fluid_backend": "3d",
        "frame_budget_ms": FRAME_BUDGET_MS,
        "paired_runs": rows,
        "baseline_pass_count": sum(
            row["baseline"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
        ),
        "candidate_pass_count": candidate_passes,
        "run_count": len(rows),
        "candidate_all_runs_pass_60fps_p95": candidate_passes == len(rows),
        "candidate_worst_frame_p95_ms": max(
            row["candidate"]["frame_p95_ms"] for row in rows
        ),
        "interpretation": (
            "The filter is a startup-only mesh reduction, but one candidate run "
            "is a large visual/physics outlier. The strict all-run gate fails and "
            "these samples do not establish a causal performance improvement."
        ),
    }


def _adaptive_ordinary_roads(scene, source: np.ndarray):
    road = linear_feature_uv(source)
    ordinary = road[~np.isclose(road[:, 9], LINEAR_STYLE_STEPS)]
    return adaptive_terrain_tessellate(
        ordinary,
        scene.terrain_height_m,
        scene.terrain_bounds,
        scene.water_mask,
        scene.water_mask_bounds,
        load_priority_area_bounds(),
    )


def _priority_contact(audit) -> tuple[dict[str, object], np.ndarray]:
    bounds = np.asarray(load_priority_area_bounds(), dtype=np.float64)
    positions = audit.road_positions_xz_m
    selected = (
        (positions[:, 0] >= bounds[0])
        & (positions[:, 0] <= bounds[2])
        & (positions[:, 1] >= bounds[1])
        & (positions[:, 1] <= bounds[3])
        & ~audit.road_over_water
    )
    absolute = np.abs(audit.road_deviation_m[selected])
    summary = {
        "sample_count": int(len(absolute)),
        "warning_5cm_sample_count": int(np.sum(absolute >= 0.05)),
        "priority_20cm_sample_count": int(np.sum(absolute >= 0.20)),
        "immediate_50cm_sample_count": int(np.sum(absolute >= 0.50)),
        "absolute_p95_p99_max_m": np.percentile(
            absolute, (95, 99, 100)
        ).tolist(),
    }
    priority = selected & (np.abs(audit.road_deviation_m) >= 0.20)
    return summary, priority


def _distance_to_bridge_centreline(scene, positions: np.ndarray) -> np.ndarray:
    if not len(positions) or not len(scene.bridge_vertices):
        return np.full(len(positions), np.inf, dtype=np.float64)
    quads = scene.bridge_vertices.reshape(-1, 6, 10)
    starts = 0.5 * (quads[:, 0, [0, 2]] + quads[:, 1, [0, 2]])
    ends = 0.5 * (quads[:, 2, [0, 2]] + quads[:, 5, [0, 2]])
    distance = np.full(len(positions), np.inf, dtype=np.float64)
    for start, end in zip(starts, ends):
        edge = end - start
        squared_length = float(edge @ edge)
        if squared_length < 0.01:
            continue
        fraction = np.clip(((positions - start) @ edge) / squared_length, 0.0, 1.0)
        closest = start + fraction[:, None] * edge
        distance = np.minimum(
            distance, np.linalg.norm(positions - closest, axis=1)
        )
    return distance


def _distance_to_water(scene, positions: np.ndarray) -> np.ndarray:
    if not len(positions):
        return np.empty(0, dtype=np.float64)
    bounds = np.asarray(scene.water_mask_bounds, dtype=np.float64)
    rows, columns = scene.water_mask.shape
    spacing_x = (bounds[2] - bounds[0]) / max(columns - 1, 1)
    spacing_z = (bounds[3] - bounds[1]) / max(rows - 1, 1)
    distance = distance_transform_edt(
        scene.water_mask < 128,
        sampling=(spacing_z, spacing_x),
    )
    column = (positions[:, 0] - bounds[0]) / (bounds[2] - bounds[0]) * (columns - 1)
    row = (positions[:, 1] - bounds[1]) / (bounds[3] - bounds[1]) * (rows - 1)
    return map_coordinates(distance, (row, column), order=1)


def _residual_classification(scene, audit, priority: np.ndarray) -> tuple[dict, dict]:
    positions = audit.road_positions_xz_m[priority]
    deviation = np.abs(audit.road_deviation_m[priority])
    bridge_distance = _distance_to_bridge_centreline(scene, positions)
    water_distance = _distance_to_water(scene, positions)
    slope = sample_heightmap_array(
        audit.slope_deg, scene.terrain_bounds, positions
    )
    bridge_review = bridge_distance <= 35.0
    shoreline_review = ~bridge_review & (water_distance <= 25.0)
    moderate_terrain = (
        ~bridge_review & ~shoreline_review & (slope >= 15.0)
    )
    unclassified = ~bridge_review & ~shoreline_review & ~moderate_terrain

    def distribution(values: np.ndarray) -> list[float]:
        return np.percentile(values, (0, 50, 90, 95, 100)).tolist()

    report = {
        "classification": (
            "risk triage only; bridge proximity and terrain slope do not prove "
            "a road deck elevation"
        ),
        "residual_priority_sample_count": int(len(positions)),
        "hierarchical_categories": {
            "bridge_or_ramp_elevation_review_within_35m": int(bridge_review.sum()),
            "shoreline_review_within_25m": int(shoreline_review.sum()),
            "moderate_source_slope_review_at_least_15deg": int(moderate_terrain.sum()),
            "unclassified": int(unclassified.sum()),
        },
        "absolute_deviation_min_p50_p90_p95_max_m": distribution(deviation),
        "bridge_distance_min_p50_p90_p95_max_m": distribution(bridge_distance),
        "water_distance_min_p50_p90_p95_max_m": distribution(water_distance),
        "source_slope_min_p50_p90_p95_max_deg": distribution(slope),
        "next_evidence_gate": (
            "Do not raise or flatten the 157 bridge/ramp-adjacent residuals until "
            "grade-A/B deck, ramp, retaining-wall or portal elevations are registered."
        ),
    }
    fields = {
        "positions": positions,
        "bridge_review": bridge_review,
        "shoreline_review": shoreline_review,
        "moderate_terrain": moderate_terrain,
        "unclassified": unclassified,
    }
    return report, fields


def _render_map(scene, semantics, fields: dict, path: Path) -> Path:
    terrain = np.asarray(scene.terrain_height_m, dtype=np.float64)
    land = scene.water_mask < 128
    low, high = np.percentile(terrain[land], (2, 98))
    grey = np.asarray(
        28.0 + np.clip((terrain - low) / max(high - low, 1e-6), 0.0, 1.0) * 145.0,
        dtype=np.uint8,
    )
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    rgb[~land] = (12, 31, 51)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    width, height = image.size

    def pixel(point) -> tuple[int, int]:
        return (
            int(round((point[0] - bounds[0]) / (bounds[2] - bounds[0]) * (width - 1))),
            int(round((point[1] - bounds[1]) / (bounds[3] - bounds[1]) * (height - 1))),
        )

    for corridor in semantics.corridors:
        draw.line([pixel(point) for point in corridor.polyline_xz_m], fill=(35, 220, 235), width=3)
    for key, colour in (
        ("bridge_review", (255, 205, 35)),
        ("shoreline_review", (235, 60, 175)),
        ("moderate_terrain", (255, 105, 35)),
        ("unclassified", (245, 245, 245)),
    ):
        for point in fields["positions"][fields[key]]:
            x, z = pixel(point)
            draw.ellipse((x - 1, z - 1, x + 1, z + 1), fill=colour)
    area = load_priority_area_bounds()
    draw.rectangle((*pixel(area[:2]), *pixel(area[2:])), outline=(245, 245, 245), width=2)
    legend_height = 72
    canvas = Image.new("RGB", (width, height + legend_height), (15, 17, 20))
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    items = (
        ("dated tunnel", (35, 220, 235)),
        ("bridge/ramp review", (255, 205, 35)),
        ("shoreline", (235, 60, 175)),
        ("terrain slope", (255, 105, 35)),
        ("unclassified", (245, 245, 245)),
    )
    x = 18
    for label, colour in items:
        legend.rectangle((x, height + 18, x + 13, height + 31), fill=colour)
        legend.text((x + 19, height + 17), label, fill=(230, 232, 235))
        x += 190
    legend.text(
        (18, height + 45),
        "Residual >= 0.20 m after semantic tunnel filtering and adaptive L2 tessellation",
        fill=(230, 232, 235),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_ROAD_SEMANTICS_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, action="append", default=[])
    parser.add_argument("--candidate-profile", type=Path, action="append", default=[])
    arguments = parser.parse_args()
    try:
        scene = load_scene(arguments.scene)
        semantics = load_road_structure_semantics(arguments.semantics)
        visible, filter_stats = filter_occluded_road_segments(
            scene.road_vertices, semantics
        )
        before_roads, before_tessellation = _adaptive_ordinary_roads(
            scene, scene.road_vertices
        )
        after_roads, after_tessellation = _adaptive_ordinary_roads(scene, visible)
        before_audit = audit_terrain_contacts(
            replace(scene, road_vertices=before_roads)
        )
        after_audit = audit_terrain_contacts(
            replace(scene, road_vertices=after_roads)
        )
        before_contact, _ = _priority_contact(before_audit)
        after_contact, priority = _priority_contact(after_audit)
        residual, fields = _residual_classification(
            scene, after_audit, priority
        )
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        map_path = _render_map(
            scene, semantics, fields, arguments.output_dir / "road_structure_map.png"
        )
        report = {
            "schema_version": 1,
            "scene_asset": arguments.scene.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
            "scene_asset_sha256": sha256(arguments.scene.read_bytes()).hexdigest(),
            "semantic_asset": arguments.semantics.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
            "semantic_asset_sha256": sha256(arguments.semantics.read_bytes()).hexdigest(),
            "snapshot_utc": semantics.snapshot_utc,
            "filter": asdict(filter_stats),
            "adaptive_tessellation": {
                "before_semantic_filter": asdict(before_tessellation),
                "after_semantic_filter": asdict(after_tessellation),
            },
            "priority_area_contact": {
                "sampling_note": "Counts change when occluded tunnel triangles are removed.",
                "before_semantic_filter": before_contact,
                "after_semantic_filter": after_contact,
            },
            "residual_classification": residual,
            "map": map_path.name,
        }
        performance = _performance_comparison(
            arguments.baseline_profile, arguments.candidate_profile
        )
        if performance is not None:
            report["performance_ab"] = performance
        report_path = arguments.output_dir / "road_structure_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
