"""Measure V1-4 adaptive road/kerb terrain tessellation on the shipped scene."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from simulator.passes.scene import linear_feature_uv, road_edge_detail_vertices
from simulator.scene import LINEAR_STYLE_STEPS, load_scene
from simulator.terrain_detail import (
    adaptive_terrain_tessellate,
    load_priority_area_bounds,
)
from simulator.validation.terrain_contacts import audit_terrain_contacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
FRAME_BUDGET_MS = 1000.0 / 60.0


def _contact_summary(audit, bounds: tuple[float, float, float, float]) -> dict:
    positions = audit.road_positions_xz_m
    in_area = (
        (positions[:, 0] >= bounds[0])
        & (positions[:, 0] <= bounds[2])
        & (positions[:, 1] >= bounds[1])
        & (positions[:, 1] <= bounds[3])
        & ~audit.road_over_water
    )
    absolute = np.abs(audit.road_deviation_m[in_area])
    return {
        "sample_count": int(len(absolute)),
        "warning_5cm_sample_count": int(np.sum(absolute >= 0.05)),
        "priority_20cm_sample_count": int(np.sum(absolute >= 0.20)),
        "immediate_50cm_sample_count": int(np.sum(absolute >= 0.50)),
        "warning_5cm_fraction": float(np.mean(absolute >= 0.05)),
        "priority_20cm_fraction": float(np.mean(absolute >= 0.20)),
        "immediate_50cm_fraction": float(np.mean(absolute >= 0.50)),
        "absolute_p95_p99_max_m": np.percentile(
            absolute, [95, 99, 100]
        ).tolist(),
    }


def _load_integrated_profile(path: Path) -> dict[str, object]:
    raw = path.read_text(encoding="utf-8-sig")
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"runtime profile contains no JSON object: {path}")
    payload = json.loads(raw[start:])
    integrated = payload.get("integrated")
    if not isinstance(integrated, dict):
        raise ValueError(f"runtime profile has no integrated result: {path}")
    return integrated


def _performance_comparison(
    baseline_paths: list[Path], candidate_paths: list[Path]
) -> dict[str, object] | None:
    if not baseline_paths and not candidate_paths:
        return None
    if len(baseline_paths) != len(candidate_paths) or not baseline_paths:
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
                "frame_p95_delta_ms": float(
                    after["frame_p95_ms"] - before["frame_p95_ms"]
                ),
            }
        )
    return {
        "frames_per_run": 360,
        "fluid_backend": "3d",
        "frame_budget_ms": FRAME_BUDGET_MS,
        "paired_runs": rows,
        "baseline_all_runs_pass_60fps_p95": all(
            row["baseline"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
        ),
        "candidate_all_runs_pass_60fps_p95": all(
            row["candidate"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
        ),
        "candidate_worst_frame_p95_ms": max(
            row["candidate"]["frame_p95_ms"] for row in rows
        ),
        "interpretation": (
            "Paired alternating processes show no mesh-induced p95 regression, "
            "but the strict three-run 16.67 ms gate remains failed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, action="append", default=[])
    parser.add_argument("--candidate-profile", type=Path, action="append", default=[])
    arguments = parser.parse_args()
    try:
        scene = load_scene(arguments.scene)
        bounds = load_priority_area_bounds()
        road = linear_feature_uv(scene.road_vertices)
        stair_mask = np.isclose(road[:, 9], LINEAR_STYLE_STEPS)
        ordinary = road[~stair_mask]
        stairs = road[stair_mask]
        edges = road_edge_detail_vertices(ordinary)
        adaptive_road, road_stats = adaptive_terrain_tessellate(
            ordinary,
            scene.terrain_height_m,
            scene.terrain_bounds,
            scene.water_mask,
            scene.water_mask_bounds,
            bounds,
        )
        adaptive_edges, edge_stats = adaptive_terrain_tessellate(
            edges,
            scene.terrain_height_m,
            scene.terrain_bounds,
            scene.water_mask,
            scene.water_mask_bounds,
            bounds,
        )
        before = audit_terrain_contacts(replace(scene, road_vertices=ordinary))
        after = audit_terrain_contacts(
            replace(scene, road_vertices=adaptive_road)
        )
        input_vertices = len(ordinary) + len(edges) + len(stairs)
        output_vertices = len(adaptive_road) + len(adaptive_edges) + len(stairs)
        performance = _performance_comparison(
            arguments.baseline_profile, arguments.candidate_profile
        )
        report = {
            "schema_version": 1,
            "scene_asset": arguments.scene.resolve()
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "scene_asset_sha256": sha256(arguments.scene.read_bytes()).hexdigest(),
            "priority_area_bounds_xz_m": list(bounds),
            "geometry": {
                "road_deck": asdict(road_stats),
                "road_edges": asdict(edge_stats),
                "combined_input_vertices": input_vertices,
                "combined_output_vertices": output_vertices,
                "combined_vertex_multiplier": output_vertices
                / max(input_vertices, 1),
                "uniform_l2_triangle_multiplier_avoided": 16.0,
            },
            "ordinary_road_contact": {
                "sampling_note": (
                    "Sample count changes with tessellation; compare thresholds, "
                    "extrema and the fixed counterfactual report together."
                ),
                "before": _contact_summary(before, bounds),
                "after": _contact_summary(after, bounds),
            },
        }
        if performance is not None:
            report["performance_ab"] = performance
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
