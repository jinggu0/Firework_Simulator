"""Generate the V2-2a road-detail scale and evidence audit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import numpy as np

from simulator.validation.road_details import (
    DEFAULT_DETAIL_OSM_PATH,
    DEFAULT_SCENE_PATH,
    DEFAULT_SHADER_PATH,
    SURFACE_LABELS,
    rendered_road_measurements,
    road_detail_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "docs/validation/road_detail_v1/road_detail_audit.json"
)
DEFAULT_DIAGNOSTIC = Path(
    "docs/validation/road_detail_v1/road_detail_scale_map.png"
)
SURFACE_COLORS = {
    3: (220, 220, 220),
    5: (108, 181, 255),
    6: (235, 78, 68),
    15: (198, 155, 88),
}


def _repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _format_widths(records: list[dict[str, Any]], limit: int = 5) -> str:
    ranked = sorted(records, key=lambda row: row["segment_count"], reverse=True)
    return ", ".join(
        f"{row['width_m']:g}m x{row['segment_count']}"
        for row in ranked[:limit]
    )


def build_diagnostic(
    report: dict[str, Any],
    measurements: dict[str, np.ndarray],
) -> Image.Image:
    width, height = 1600, 900
    map_width = 1120
    margin = 35
    image = Image.new("RGB", (width, height), (12, 15, 20))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 18), "V2-2a rendered road detail scale audit", fill="white")

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

    for code in (15, 5, 6, 3):
        selected = np.flatnonzero(measurements["surface_code"] == code)
        color = SURFACE_COLORS[code]
        for index in selected:
            draw.line((point(starts[index]), point(ends[index])), fill=color)

    # A 1.7 km circle is the current origin-distance limit for derived kerbs.
    centre_px = point(np.array([0.0, 0.0]))
    radius_px = round(1700.0 * scale)
    draw.ellipse(
        (
            centre_px[0] - radius_px,
            centre_px[1] - radius_px,
            centre_px[0] + radius_px,
            centre_px[1] + radius_px,
        ),
        outline=(255, 174, 65),
        width=2,
    )
    draw.text(
        (centre_px[0] + radius_px + 5, centre_px[1]),
        "generic kerb radius 1.7 km",
        fill=(255, 174, 65),
    )

    legend_y = 63
    for code in (3, 5, 6, 15):
        draw.line((50, legend_y + 5, 78, legend_y + 5), fill=SURFACE_COLORS[code], width=3)
        draw.text((86, legend_y), SURFACE_LABELS[code], fill=(220, 225, 232))
        legend_y += 20

    panel_left = map_width + 20
    draw.rectangle((map_width, 0, width, height), fill=(22, 27, 35))
    y = 28

    def line(text: str, color: tuple[int, int, int] = (220, 225, 232), gap: int = 19) -> None:
        nonlocal y
        draw.text((panel_left, y), text, fill=color)
        y += gap

    inventory = report["runtime_road_inventory"]
    line("Rendered inventory", (255, 255, 255), 24)
    line(f"segments: {inventory['visible_segment_count']:,}")
    line(f"centreline: {inventory['visible_centreline_length_m'] / 1000.0:.2f} km")
    y += 8
    for surface in inventory["surfaces"]:
        line(surface["surface"], SURFACE_COLORS[surface["surface_code"]])
        line(
            "  " + _format_widths(surface["rounded_authored_width_counts"]),
            (175, 181, 190),
        )
    y += 10
    kerb = report["kerb_contract"]
    line("Generic kerb contract", (255, 174, 65), 24)
    line(f"reveal/top: {kerb['reveal_height_m']:.2f} / {kerb['top_width_m']:.2f} m")
    line(f"source segments: {kerb['generated_source_segment_count']:,}")
    line("evidence: grade D, not site surveyed", (255, 150, 145))
    y += 10
    paint = report["asphalt_marking_contract"]
    line("Procedural asphalt paint", (255, 255, 255), 24)
    edge = paint["edge_line_support_width_m"]
    line(f"line width range: {edge['minimum']:.3f}-{edge['maximum']:.3f} m")
    line(f">0.20 m segments: {paint['segments_over_0_20_m_line_width']:,}")
    line(f">0.30 m segments: {paint['segments_over_0_30_m_line_width']:,}")
    line(f"dash support/core: {paint['dash_support_length_m']:.2f}/{paint['dash_core_length_m']:.2f} m")
    y += 10
    cycle = report["cycleway_marking_contract"]
    line("Cycleway surface bands", SURFACE_COLORS[6], 24)
    line(f"period: {cycle['stripe_period_m']:.2f} m")
    line(f"support width: {cycle['stripe_support_width_m']:.2f} m")
    y += 10
    line("Blocked evidence", (255, 150, 145), 24)
    line("- source way ids/tags absent from NPZ")
    line("- no dated manhole/crack positions")
    line("- no bicycle-symbol geometry")
    line("- no site-verified kerb profile")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--shader", type=Path, default=DEFAULT_SHADER_PATH)
    parser.add_argument("--detail-osm", type=Path, default=DEFAULT_DETAIL_OSM_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    arguments = parser.parse_args()
    scene_path = _repository_path(arguments.scene)
    shader_path = _repository_path(arguments.shader)
    detail_osm_path = _repository_path(arguments.detail_osm)
    report = road_detail_report(scene_path, shader_path, detail_osm_path)
    measurements, _, _ = rendered_road_measurements(scene_path)
    diagnostic = build_diagnostic(report, measurements)
    output = _repository_path(arguments.output)
    diagnostic_path = _repository_path(arguments.diagnostic)
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.save(diagnostic_path)
    report["diagnostic"] = {
        "asset": _display_path(diagnostic_path),
        "sha256": _digest(diagnostic_path),
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output}: "
        f"segments={report['runtime_road_inventory']['visible_segment_count']}, "
        f"kerb_sources={report['kerb_contract']['generated_source_segment_count']}"
    )


if __name__ == "__main__":
    main()
