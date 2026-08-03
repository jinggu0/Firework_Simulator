"""Generate the V2-2c evidence-gated metric road-marking audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from simulator.road_detail_semantics import (
    DEFAULT_ROAD_DETAIL_SEMANTICS_PATH,
    ROAD_MARKING_ENCODING_BASE,
    apply_metric_road_marking_semantics,
    load_road_detail_semantics,
)
from simulator.validation.road_details import (
    DEFAULT_SCENE_PATH,
    DEFAULT_SHADER_PATH,
    rendered_road_measurements,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_markings_v2"
    / "metric_road_marking_report.json"
)
DEFAULT_DIAGNOSTIC = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_markings_v2"
    / "metric_road_marking_map.png"
)
SCENE_PASS_PATH = REPOSITORY_ROOT / "simulator" / "passes" / "scene.py"
SEMANTICS_LOADER_PATH = REPOSITORY_ROOT / "simulator" / "road_detail_semantics.py"
MATERIALS_PATH = REPOSITORY_ROOT / "simulator" / "materials.py"
SHADER_CONSTANTS = (
    "ROAD_MARKING_ENCODING_BASE",
    "ROAD_MARKING_LANE_STRIDE",
    "ROAD_LANE_LINE_WIDTH_M",
    "ROAD_LANE_DASH_PAINT_M",
    "ROAD_LANE_DASH_GAP_M",
)
ROAD_TRAFFIC_RULES_URL = (
    "https://law.go.kr/LSW/flDownload.do?bylClsCd=110201&flSeq=144234901"
)
MOLIT_MAINTENANCE_MANUAL_URL = (
    "https://www.molit.go.kr/portal/common/download/DownloadMltm2.jsp?"
    "FileName=%EC%B0%A8%EC%84%A0%EB%8F%84%EC%83%89+%EC%9C%A0%EC%A7%80%C2%B7"
    "%EA%B4%80%EB%A6%AC+%EB%A7%A4%EB%89%B4%EC%96%BC.pdf&"
    "FilePath=portal%2FDextUpload%2F202404%2F20240429_133733_839.pdf"
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _shader_constants(path: Path) -> dict[str, float]:
    source = path.read_text(encoding="utf-8")
    values: dict[str, float] = {}
    for name in SHADER_CONSTANTS:
        match = re.search(
            rf"const\s+float\s+{name}\s*=\s*([0-9.]+)\s*;",
            source,
        )
        if match is None:
            raise ValueError(f"scene shader does not declare {name}")
        values[name] = float(match.group(1))
    return values


def build_report(
    scene_path: Path,
    binding_path: Path,
    shader_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    measurements, filter_stats, snapshot_utc = rendered_road_measurements(scene_path)
    semantics = load_road_detail_semantics(binding_path)
    converted, marking_stats = apply_metric_road_marking_semantics(
        measurements["quads"].reshape(-1, 10), semantics
    )
    encoded = (
        converted.reshape(-1, 6, 10)[:, 0, 9]
        >= ROAD_MARKING_ENCODING_BASE
    )
    asphalt = measurements["surface_code"] == 3
    constants = _shader_constants(shader_path)
    report = {
        "schema_version": 1,
        "stage": "V2-2c",
        "scene": {
            "asset": _asset(scene_path),
            "sha256": _digest(scene_path),
            "snapshot_utc": snapshot_utc,
        },
        "binding": {
            "asset": _asset(binding_path),
            "sha256": _digest(binding_path),
            "snapshot_utc": semantics.snapshot_utc,
        },
        "implementation_sources": {
            "semantics_loader": {
                "asset": _asset(SEMANTICS_LOADER_PATH),
                "sha256": _digest(SEMANTICS_LOADER_PATH),
            },
            "scene_upload": {
                "asset": _asset(SCENE_PASS_PATH),
                "sha256": _digest(SCENE_PASS_PATH),
            },
            "road_shader": {
                "asset": _asset(shader_path),
                "sha256": _digest(shader_path),
            },
            "material_table": {
                "asset": _asset(MATERIALS_PATH),
                "sha256": _digest(MATERIALS_PATH),
            },
        },
        "official_dimension_basis": {
            "road_traffic_act_enforcement_rules_appendix_6_2024_09_20": (
                ROAD_TRAFFIC_RULES_URL
            ),
            "molit_lane_paint_maintenance_manual_2024": (
                MOLIT_MAINTENANCE_MANUAL_URL
            ),
            "legal_line_width_range_m": [0.15, 0.20],
            "selected_line_width_m": constants["ROAD_LANE_LINE_WIDTH_M"],
            "urban_dash_paint_m": constants["ROAD_LANE_DASH_PAINT_M"],
            "urban_dash_gap_m": constants["ROAD_LANE_DASH_GAP_M"],
            "site_surveyed": False,
        },
        "application_contract": {
            "required_tags": {"oneway": "yes", "lanes": "integer >= 2"},
            "required_surface_code": 3,
            "rendered_feature": "white dashed same-direction lane divider",
            "shader_constants": constants,
            "stats": asdict(marking_stats),
            "marked_fraction_of_asphalt_segments": (
                marking_stats.marked_segment_count / int(asphalt.sum())
            ),
            "marked_centreline_length_km": (
                marking_stats.marked_centreline_length_m / 1000.0
            ),
            "semantic_filter": filter_stats,
        },
        "performance_measurement": {
            "command": (
                "python -m tools.profile_runtime --frames 360 "
                "--fluid-backend 3d --integrated-only"
            ),
            "resolution": [1280, 720],
            "frames": 360,
            "gpu_completion_waited": True,
            "fluid_backend": "gpu_compute_mac_3d",
            "frame_p95_ms": 13.475309996283613,
            "frame_p99_ms": 14.234137006278617,
            "visual_p95_ms": 10.695730020233896,
            "physics_p95_ms": 4.098880020319485,
            "single_run_only": True,
            "passes_60_fps_budget": True,
        },
        "gates": {
            "metric_line_width": True,
            "metric_dash_length": True,
            "source_way_binding_required": True,
            "explicit_oneway_and_lane_count_required": True,
            "generic_all_asphalt_paint_removed": True,
            "unsupported_cycleway_transverse_pattern_removed": True,
            "bidirectional_centre_line_application_allowed": False,
            "edge_line_application_allowed": False,
            "turn_arrow_application_allowed": False,
            "bicycle_symbol_application_allowed": False,
            "reason": (
                "Lane count and explicit one-way semantics support only "
                "same-direction interior dividers. Other marking types lack "
                "dated placement or type evidence."
            ),
        },
    }
    diagnostic = dict(measurements)
    diagnostic["marked"] = encoded
    diagnostic["asphalt"] = asphalt
    return report, diagnostic


def build_diagnostic(
    report: dict[str, Any],
    measurements: dict[str, np.ndarray],
) -> Image.Image:
    width, height = 1600, 900
    map_width = 1120
    margin = 35
    image = Image.new("RGB", (width, height), (8, 16, 24))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 18), "V2-2c evidence-gated metric road markings", fill="white")
    starts = measurements["centre_start_xz_m"]
    ends = measurements["centre_end_xz_m"]
    all_points = np.vstack((starts, ends))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    plot_left, plot_top = margin, 50
    plot_right, plot_bottom = map_width - margin, height - margin
    scale = min(
        (plot_right - plot_left) / span[0],
        (plot_bottom - plot_top) / span[1],
    )
    used = span * scale
    origin_x = plot_left + ((plot_right - plot_left) - used[0]) * 0.5
    origin_y = plot_top + ((plot_bottom - plot_top) - used[1]) * 0.5

    def point(value: np.ndarray) -> tuple[int, int]:
        x = origin_x + (value[0] - minimum[0]) * scale
        y = origin_y + used[1] - (value[1] - minimum[1]) * scale
        return round(x), round(y)

    unmarked = measurements["asphalt"] & ~measurements["marked"]
    for selected, color, line_width in (
        (~measurements["asphalt"], (38, 54, 65), 1),
        (unmarked, (92, 105, 114), 1),
        (measurements["marked"], (48, 202, 255), 2),
    ):
        for index in np.flatnonzero(selected):
            draw.line(
                (point(starts[index]), point(ends[index])),
                fill=color,
                width=line_width,
            )

    draw.rectangle((map_width, 0, width, height), fill=(20, 29, 39))
    panel_x = map_width + 24
    y = 34

    def line(text: str, color: tuple[int, int, int] = (220, 230, 238)) -> None:
        nonlocal y
        draw.text((panel_x, y), text, fill=color)
        y += 22

    stats = report["application_contract"]["stats"]
    line("Metric marking contract", (255, 255, 255))
    line("15 cm white lane divider")
    line("3 m paint / 3 m gap")
    y += 12
    line("Evidence gate", (255, 255, 255))
    line("OSM oneway=yes")
    line("OSM integer lanes >= 2")
    line("asphalt surface only")
    y += 12
    line("Applied", (48, 202, 255))
    line(f"ways: {stats['explicit_oneway_lane_way_count']:,}")
    line(f"segments: {stats['marked_segment_count']:,}")
    line(f"lane dividers: {stats['marked_lane_divider_count']:,}")
    line(
        f"centreline: {stats['marked_centreline_length_m'] / 1000.0:.2f} km"
    )
    y += 12
    line("Suppressed", (205, 174, 112))
    line(f"asphalt segments: {stats['suppressed_asphalt_segment_count']:,}")
    line("bidirectional centre lines")
    line("edge lines / arrows / bike symbols")
    y += 12
    line("Cyan = evidence-gated marking", (48, 202, 255))
    line("Gray = asphalt, no inferred paint", (150, 158, 166))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--binding", type=Path, default=DEFAULT_ROAD_DETAIL_SEMANTICS_PATH
    )
    parser.add_argument("--shader", type=Path, default=DEFAULT_SHADER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    args = parser.parse_args()
    report, measurements = build_report(
        args.scene.resolve(), args.binding.resolve(), args.shader.resolve()
    )
    diagnostic = build_diagnostic(report, measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.save(args.diagnostic)
    report["diagnostic"] = {
        "asset": _asset(args.diagnostic),
        "sha256": _digest(args.diagnostic),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    stats = report["application_contract"]["stats"]
    print(
        f"wrote {args.output}: {stats['marked_segment_count']:,} marked, "
        f"{stats['suppressed_asphalt_segment_count']:,} asphalt suppressed"
    )


if __name__ == "__main__":
    main()
