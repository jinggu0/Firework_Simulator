"""Generate the V2-2d dated road-marking evidence and coverage audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from simulator.road_detail_semantics import (
    DEFAULT_ROAD_DETAIL_SEMANTICS_PATH,
    ROAD_MARKING_MAX_LANES,
    RoadDetailWaySemantics,
    load_road_detail_semantics,
)
from simulator.validation.road_details import (
    DEFAULT_SCENE_PATH,
    rendered_road_measurements,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "assets" / "yeouido_road_osm_2024-10-05.json"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_marking_evidence_v2"
    / "road_marking_evidence_report.json"
)
DEFAULT_DIAGNOSTIC = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_marking_evidence_v2"
    / "road_marking_evidence_map.png"
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
SMAP_GUIDE_URL = "https://smap.seoul.go.kr/guide/guide.html"
SMAP_URL = "https://smap.seoul.go.kr/"
SEOUL_BICYCLE_STATISTICS_URL = (
    "https://data.seoul.go.kr/bsp/wgs/dataView/data300View/428.do"
)
CYCLE_LANE_VALUES = {"lane", "opposite_lane", "shared_lane"}
EVIDENCE_TAG_KEYS = (
    "lanes",
    "lanes:forward",
    "lanes:backward",
    "oneway",
    "divider",
    "shoulder",
    "turn:lanes",
    "turn:lanes:forward",
    "turn:lanes:backward",
    "cycleway",
    "cycleway:both",
    "cycleway:both:lane",
    "cycleway:left",
    "cycleway:left:lane",
    "cycleway:right",
    "cycleway:right:lane",
    "crossing:markings",
    "road_marking",
    "lane_markings",
    "centre_line",
    "center_line",
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _strict_lane_count(tags: dict[str, str]) -> int | None:
    raw = tags.get("lanes", "")
    try:
        lanes = int(raw)
    except ValueError:
        return None
    if str(lanes) != raw.strip() or not 2 <= lanes <= ROAD_MARKING_MAX_LANES:
        return None
    return lanes


def _cycle_lane_tagged(tags: dict[str, str]) -> bool:
    for key, value in tags.items():
        if not key.startswith("cycleway"):
            continue
        if value in CYCLE_LANE_VALUES:
            return True
        if key.endswith(":lane") and value == "exclusive":
            return True
    return False


def _relevant_tags(tags: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(tags.items())
        if key in EVIDENCE_TAG_KEYS or key.startswith("cycleway:")
    }


def _candidate_group(
    ways: tuple[RoadDetailWaySemantics, ...],
    raw_tags: dict[int, dict[str, str]],
    lengths_m: np.ndarray,
    predicate: Callable[[RoadDetailWaySemantics, dict[str, str]], bool],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    segment_indices: list[int] = []
    for way in ways:
        tags = raw_tags.get(way.osm_way_id, way.tags)
        if not way.rendered_segment_indices or not predicate(way, tags):
            continue
        indices = list(way.rendered_segment_indices)
        segment_indices.extend(indices)
        records.append(
            {
                "osm_way_id": way.osm_way_id,
                "tags": _relevant_tags(tags),
                "rendered_segment_count": len(indices),
                "rendered_centreline_length_m": float(lengths_m[indices].sum()),
            }
        )
    unique_indices = sorted(set(segment_indices))
    if len(unique_indices) != len(segment_indices):
        raise ValueError("candidate group contains multiply bound rendered segments")
    return {
        "way_count": len(records),
        "rendered_segment_count": len(unique_indices),
        "rendered_centreline_length_m": float(lengths_m[unique_indices].sum()),
        "records": records,
        "rendered_segment_indices": unique_indices,
    }


def _tag_inventory(elements: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for element in elements:
        tags = element.get("tags", {})
        for key in EVIDENCE_TAG_KEYS:
            if key in tags:
                values[key][str(tags[key])] += 1
    return {
        key: {
            "way_count": sum(values[key].values()),
            "values": dict(sorted(values[key].items())),
        }
        for key in EVIDENCE_TAG_KEYS
    }


def build_report(
    source_path: Path,
    scene_path: Path,
    binding_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    elements = source.get("elements", [])
    raw_tags = {
        int(element["id"]): {
            str(key): str(value)
            for key, value in element.get("tags", {}).items()
        }
        for element in elements
    }
    semantics = load_road_detail_semantics(binding_path)
    measurements, filter_stats, snapshot_utc = rendered_road_measurements(scene_path)
    if snapshot_utc != semantics.snapshot_utc or snapshot_utc != source["snapshot_utc"]:
        raise ValueError("source, binding, and rendered scene timestamps differ")
    lengths_m = measurements["length_m"]
    ways = semantics.ways

    groups = {
        "already_rendered_oneway_lane_dividers": _candidate_group(
            ways,
            raw_tags,
            lengths_m,
            lambda way, tags: (
                way.derived_surface_code == 3
                and tags.get("oneway") == "yes"
                and _strict_lane_count(tags) is not None
            ),
        ),
        "centre_line_candidates_non_oneway_yes_multilane": _candidate_group(
            ways,
            raw_tags,
            lengths_m,
            lambda way, tags: (
                way.derived_surface_code == 3
                and tags.get("oneway") != "yes"
                and _strict_lane_count(tags) is not None
            ),
        ),
        "directional_lane_count_candidates": _candidate_group(
            ways,
            raw_tags,
            lengths_m,
            lambda way, tags: (
                way.derived_surface_code == 3
                and (
                    "lanes:forward" in tags or "lanes:backward" in tags
                )
            ),
        ),
        "cycle_lane_tag_candidates": _candidate_group(
            ways,
            raw_tags,
            lengths_m,
            lambda _way, tags: _cycle_lane_tagged(tags),
        ),
        "crossing_marking_presence_candidates": _candidate_group(
            ways,
            raw_tags,
            lengths_m,
            lambda _way, tags: "crossing:markings" in tags,
        ),
    }

    inventory = _tag_inventory(elements)
    report = {
        "schema_version": 1,
        "stage": "V2-2d",
        "event_date": "2024-10-05",
        "snapshot_utc": snapshot_utc,
        "inputs": {
            "historical_osm": {
                "asset": _asset(source_path),
                "sha256": _digest(source_path),
                "provider": source.get("provider", ""),
                "licence": source.get("licence", ""),
            },
            "rendered_scene": {
                "asset": _asset(scene_path),
                "sha256": _digest(scene_path),
            },
            "way_binding": {
                "asset": _asset(binding_path),
                "sha256": _digest(binding_path),
            },
        },
        "rendered_inventory": {
            "rendered_segment_count": len(lengths_m),
            "semantic_filter": filter_stats,
        },
        "source_capability": {
            "historical_osm": {
                "status": "candidate_geometry_only",
                "supports": [
                    "event-dated way identifiers",
                    "lane-count and directional candidate discovery",
                    "limited cycle-lane candidate discovery",
                ],
                "does_not_support": [
                    "way-specific centre-line colour or solid/dashed state",
                    "edge-line placement",
                    "turn-arrow geometry",
                    "bicycle-symbol geometry",
                ],
            },
            "official_marking_rules": {
                "status": "dimensions_and_classes_only",
                "road_traffic_act_enforcement_rules_appendix_6_2024_09_20": (
                    ROAD_TRAFFIC_RULES_URL
                ),
                "molit_lane_paint_maintenance_manual_2024": (
                    MOLIT_MAINTENANCE_MANUAL_URL
                ),
                "limitation": (
                    "The rules define permitted classes and physical dimensions, "
                    "but do not assign an event-date marking type to each OSM way."
                ),
            },
            "seoul_smap_2024_background": {
                "status": "available_not_acquired_or_registered",
                "service": SMAP_URL,
                "official_guide": SMAP_GUIDE_URL,
                "supports": "selection of a 2024 background-map year",
                "unresolved": [
                    "exact imagery acquisition date",
                    "event-date applicability",
                    "capture reuse terms",
                    "pixel-to-world registration residual",
                    "visibility of paint at source resolution",
                ],
            },
            "seoul_bicycle_road_statistics": {
                "status": "aggregate_classification_only",
                "source": SEOUL_BICYCLE_STATISTICS_URL,
                "limitation": (
                    "Route-type statistics define facility categories but do not "
                    "provide event-date, way-level paint geometry."
                ),
            },
        },
        "historical_osm_tag_inventory": inventory,
        "candidate_groups": groups,
        "application_gates": {
            "existing_same_direction_lane_dividers_remain_allowed": True,
            "bidirectional_centre_line_application_allowed": False,
            "centre_line_colour_selection_allowed": False,
            "centre_line_solid_dash_selection_allowed": False,
            "edge_line_application_allowed": False,
            "turn_arrow_application_allowed": False,
            "bicycle_symbol_application_allowed": False,
            "crossing_marking_shape_application_allowed": False,
            "runtime_geometry_changed_by_this_stage": False,
            "reason": (
                "Candidate topology is not way-specific evidence of marking type, "
                "placement, colour, or event-date state."
            ),
        },
        "next_registration_contract": {
            "source": "Seoul S-Map 2024 background map",
            "required_metadata": [
                "source URL and selected year layer",
                "imagery acquisition date and publication date",
                "capture timestamp, viewport, and source checksum",
                "permitted local validation and redistribution scope",
            ],
            "registration": {
                "minimum_control_points": 4,
                "minimum_independent_check_points": 2,
                "project_max_planimetric_residual_m": 1.0,
                "control_points_must_span_each_registered_crop": True,
            },
            "marking_acceptance": [
                "paint must be individually visible at native source resolution",
                "imagery date must be on or before the event or explicitly historical",
                "each accepted type must be associated with OSM way IDs",
                "solid/dashed state, colour, side, and longitudinal extent are recorded",
            ],
        },
        "conclusion": (
            "No new marking type passes the event-date placement gate. The audit "
            "narrows imagery review to bound candidate ways without changing runtime."
        ),
    }
    diagnostic = dict(measurements)
    for name, group in groups.items():
        mask = np.zeros(len(lengths_m), dtype=bool)
        mask[group.pop("rendered_segment_indices")] = True
        diagnostic[name] = mask
    return report, diagnostic


def build_diagnostic(
    report: dict[str, Any],
    measurements: dict[str, np.ndarray],
) -> Image.Image:
    width, height = 1720, 940
    map_width = 1160
    margin = 36
    image = Image.new("RGB", (width, height), (8, 16, 24))
    draw = ImageDraw.Draw(image)
    draw.text((margin, 18), "V2-2d road-marking evidence coverage", fill="white")
    starts = measurements["centre_start_xz_m"]
    ends = measurements["centre_end_xz_m"]
    points = np.vstack((starts, ends))
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1.0)
    plot_left, plot_top = margin, 52
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

    layers = (
        (np.ones(len(starts), dtype=bool), (42, 57, 68), 1),
        (
            measurements["already_rendered_oneway_lane_dividers"],
            (48, 202, 255),
            2,
        ),
        (
            measurements["centre_line_candidates_non_oneway_yes_multilane"],
            (255, 198, 74),
            3,
        ),
        (measurements["directional_lane_count_candidates"], (238, 94, 166), 4),
        (measurements["cycle_lane_tag_candidates"], (82, 214, 129), 4),
        (measurements["crossing_marking_presence_candidates"], (255, 111, 82), 5),
    )
    for selected, color, line_width in layers:
        for index in np.flatnonzero(selected):
            draw.line(
                (point(starts[index]), point(ends[index])),
                fill=color,
                width=line_width,
            )

    draw.rectangle((map_width, 0, width, height), fill=(20, 29, 39))
    panel_x = map_width + 24
    y = 32

    def line(text: str, color: tuple[int, int, int] = (220, 230, 238)) -> None:
        nonlocal y
        draw.text((panel_x, y), text, fill=color)
        y += 22

    line("Evidence state", (255, 255, 255))
    line("Snapshot: 2024-10-05 10:20 UTC")
    line("Runtime changes: none")
    y += 12
    labels = (
        (
            "already_rendered_oneway_lane_dividers",
            "allowed one-way dividers",
            (48, 202, 255),
        ),
        (
            "centre_line_candidates_non_oneway_yes_multilane",
            "centre-line review candidates",
            (255, 198, 74),
        ),
        (
            "directional_lane_count_candidates",
            "directional lane-count tags",
            (238, 94, 166),
        ),
        (
            "cycle_lane_tag_candidates",
            "cycle-lane tags",
            (82, 214, 129),
        ),
        (
            "crossing_marking_presence_candidates",
            "crossing marking presence",
            (255, 111, 82),
        ),
    )
    for key, label, color in labels:
        group = report["candidate_groups"][key]
        line(label, color)
        line(
            f"  {group['way_count']:,} ways / "
            f"{group['rendered_segment_count']:,} segments / "
            f"{group['rendered_centreline_length_m'] / 1000.0:.2f} km"
        )
    y += 12
    line("Blocked until registered imagery", (255, 255, 255))
    line("centre-line colour + solid/dashed")
    line("edge lines / turn arrows")
    line("bicycle symbols / crossing shapes")
    y += 12
    line("Candidate != placement evidence", (255, 198, 74))
    line("Overlapping colours are intentional.")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--binding", type=Path, default=DEFAULT_ROAD_DETAIL_SEMANTICS_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    args = parser.parse_args()
    report, measurements = build_report(
        args.source.resolve(), args.scene.resolve(), args.binding.resolve()
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
    candidates = report["candidate_groups"]
    centre = candidates["centre_line_candidates_non_oneway_yes_multilane"]
    cycle = candidates["cycle_lane_tag_candidates"]
    print(
        f"wrote {args.output}: centre review {centre['way_count']:,} ways, "
        f"cycle review {cycle['way_count']:,} ways, no runtime changes"
    )


if __name__ == "__main__":
    main()
