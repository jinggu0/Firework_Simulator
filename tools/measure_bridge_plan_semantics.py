"""Measure V1-7 dated Seogang Bridge plan replacement and evidence gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from simulator.bridge_semantics import (
    DEFAULT_BRIDGE_SEMANTICS_PATH,
    load_bridge_plan_semantics,
    replace_seogang_bridge_plan,
    replacement_deck_vertices,
    seogang_bridge_segment_mask,
)
from simulator.scene import load_scene


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "validation" / "bridge_plan_v1"
FRAME_BUDGET_MS = 1000.0 / 60.0


def _mesh_plan_area(vertices: np.ndarray) -> float:
    triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 10)
    a = triangles[:, 1, [0, 2]] - triangles[:, 0, [0, 2]]
    b = triangles[:, 2, [0, 2]] - triangles[:, 0, [0, 2]]
    return float(0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())


def _load_profile(path: Path) -> dict[str, float]:
    raw = path.read_text(encoding="utf-8-sig")
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"runtime profile contains no JSON object: {path}")
    integrated = json.loads(raw[start:]).get("integrated")
    if not isinstance(integrated, dict):
        raise ValueError(f"runtime profile has no integrated result: {path}")
    return {
        key: float(integrated[key])
        for key in (
            "frame_mean_ms",
            "frame_p95_ms",
            "frame_p99_ms",
            "visual_p95_ms",
            "physics_p95_ms",
        )
    }


def _performance(baseline: list[Path], candidate: list[Path]) -> dict | None:
    if not baseline and not candidate:
        return None
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate profile counts must match")
    rows = []
    for index, (before_path, after_path) in enumerate(
        zip(baseline, candidate), start=1
    ):
        before = _load_profile(before_path)
        after = _load_profile(after_path)
        rows.append(
            {
                "run": index,
                "baseline": before,
                "candidate": after,
                "frame_p95_delta_ms": (
                    after["frame_p95_ms"] - before["frame_p95_ms"]
                ),
            }
        )
    return {
        "frames_per_run": 360,
        "fluid_backend": "3d",
        "frame_budget_ms": FRAME_BUDGET_MS,
        "paired_runs": rows,
        "baseline_pass_count": sum(
            row["baseline"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
        ),
        "candidate_pass_count": sum(
            row["candidate"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
        ),
        "run_count": len(rows),
        "candidate_all_runs_pass_60fps_p95": all(
            row["candidate"]["frame_p95_ms"] < FRAME_BUDGET_MS for row in rows
        ),
        "interpretation": (
            "The replacement runs once during scene construction and reduces the "
            "bridge batch. Paired host measurements still cannot isolate its small "
            "cost from driver and scheduling variance."
        ),
    }


def _render_map(scene, semantics, removed: np.ndarray, path: Path) -> None:
    terrain = np.asarray(scene.terrain_height_m, dtype=np.float64)
    land = scene.water_mask < 128
    low, high = np.percentile(terrain[land], (2, 98))
    grey = np.asarray(
        30.0
        + np.clip((terrain - low) / max(high - low, 1e-6), 0.0, 1.0) * 135.0,
        dtype=np.uint8,
    )
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    rgb[~land] = (11, 31, 51)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image, mode="RGBA")
    bounds = np.asarray(scene.terrain_bounds, dtype=np.float64)
    width, height = image.size

    def pixel(point: np.ndarray) -> tuple[int, int]:
        return (
            int(
                round(
                    (point[0] - bounds[0])
                    / (bounds[2] - bounds[0])
                    * (width - 1)
                )
            ),
            int(
                round(
                    (point[1] - bounds[1])
                    / (bounds[3] - bounds[1])
                    * (height - 1)
                )
            ),
        )

    quads = scene.bridge_vertices.reshape(-1, 6, 10)
    for quad in quads[removed]:
        polygon = [pixel(quad[index, [0, 2]]) for index in (0, 1, 2, 5)]
        draw.polygon(
            polygon, fill=(255, 66, 55, 105), outline=(255, 115, 95, 210)
        )
    sections = semantics.cross_sections_xz_m
    replacement_ring = np.concatenate((sections[:, 0], sections[::-1, 1]), axis=0)
    draw.polygon(
        [pixel(point) for point in replacement_ring],
        fill=(33, 225, 229, 120),
        outline=(65, 245, 245, 255),
    )
    draw.line(
        [pixel(point) for point in semantics.outline_xz_m],
        fill=(255, 255, 255, 230),
        width=2,
    )
    legend_height = 58
    canvas = Image.new("RGB", (width, height + legend_height), (15, 17, 20))
    canvas.paste(image, (0, 0))
    legend = ImageDraw.Draw(canvas)
    legend.rectangle((18, height + 15, 31, height + 28), fill=(255, 66, 55))
    legend.text(
        (38, height + 14), "removed generic OSM strips", fill=(235, 238, 240)
    )
    legend.rectangle((270, height + 15, 283, height + 28), fill=(33, 225, 229))
    legend.text(
        (290, height + 14), "dated outline replacement", fill=(235, 238, 240)
    )
    legend.text(
        (18, height + 36),
        "Plan evidence only: replacement remains at the existing local Y=7 m.",
        fill=(235, 238, 240),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--semantics", type=Path, default=DEFAULT_BRIDGE_SEMANTICS_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-profile", type=Path, action="append", default=[])
    parser.add_argument("--candidate-profile", type=Path, action="append", default=[])
    arguments = parser.parse_args()

    scene = load_scene(arguments.scene)
    semantics = load_bridge_plan_semantics(arguments.semantics)
    removed = seogang_bridge_segment_mask(scene.bridge_vertices, semantics)
    replacement, stats = replace_seogang_bridge_plan(
        scene.bridge_vertices, semantics
    )
    removed_vertices = (
        scene.bridge_vertices.reshape(-1, 6, 10)[removed].reshape(-1, 10)
    )
    replacement_only = replacement_deck_vertices(semantics)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "bridge_plan_map.png"
    _render_map(scene, semantics, removed, map_path)
    report = {
        "schema_version": 1,
        "stage": "V1-7",
        "target_event_date": "2024-10-05",
        "scene_asset": arguments.scene.resolve()
        .relative_to(REPOSITORY_ROOT)
        .as_posix(),
        "scene_asset_sha256": sha256(arguments.scene.read_bytes()).hexdigest(),
        "semantic_asset": arguments.semantics.resolve()
        .relative_to(REPOSITORY_ROOT)
        .as_posix(),
        "semantic_asset_sha256": sha256(arguments.semantics.read_bytes()).hexdigest(),
        "replacement": {
            **asdict(stats),
            "removed_plan_area_m2": _mesh_plan_area(removed_vertices),
            "replacement_plan_area_m2": _mesh_plan_area(replacement_only),
            "input_local_y_values_m": np.unique(
                scene.bridge_vertices[:, 1]
            ).tolist(),
            "output_local_y_values_m": np.unique(replacement[:, 1]).tolist(),
            "vertical_coordinates_changed": not np.array_equal(
                np.unique(scene.bridge_vertices[:, 1]), np.unique(replacement[:, 1])
            ),
        },
        "evidence_gates": {
            "historical_plan_snapshot_registered": True,
            "minimum_matching_source_segments": (
                semantics.minimum_matching_source_segments
            ),
            "station_registration_passed": semantics.station_registration_passed,
            "station_length_residual_m": semantics.station_length_residual_m,
            "event_inside_paint_contract": semantics.event_inside_paint_contract,
            "event_day_construction_visual_state_known": (
                semantics.construction_visual_state_known
            ),
            "scaffolding_or_fresh_paint_drawn": False,
            "vertical_profile_application_allowed": False,
        },
        "performance": _performance(
            arguments.baseline_profile, arguments.candidate_profile
        ),
        "artifacts": {"plan_map": map_path.relative_to(REPOSITORY_ROOT).as_posix()},
        "passed": bool(
            stats.removed_generic_segments > stats.replacement_segments
            and not semantics.station_registration_passed
            and not semantics.construction_visual_state_known
            and np.allclose(replacement[:, 1], semantics.local_y_m)
        ),
    }
    report_path = output_dir / "bridge_plan_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
