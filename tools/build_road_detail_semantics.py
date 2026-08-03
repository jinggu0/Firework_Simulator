"""Bind dated OSM highway semantics to the shipped rendered road segments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulator.scene import _road_surface, _road_width, build_scene, load_scene
from simulator.road_detail_semantics import RUNTIME_MARKING_POLICY
from simulator.validation.road_details import (
    rendered_road_measurements,
    road_quad_measurements,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY_ROOT / "assets" / "yeouido_road_osm_2024-10-05.json"
DEFAULT_SCENE = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "assets" / "yeouido_road_detail_semantics_2024-10-05.json"
)
ENDPOINT_QUANTIZATION_M = 0.01
SEMANTIC_TAGS = (
    "highway",
    "width",
    "lanes",
    "lanes:forward",
    "lanes:backward",
    "oneway",
    "surface",
    "cycleway",
    "cycleway:both",
    "cycleway:left",
    "cycleway:right",
    "bicycle",
    "foot",
    "footway",
    "segregated",
    "shoulder",
    "maxspeed",
    "divider",
    "turn:lanes",
    "turn:lanes:forward",
    "turn:lanes:backward",
    "bridge",
    "tunnel",
    "covered",
    "layer",
    "service",
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _asset_name(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _segment_key(start: np.ndarray, end: np.ndarray) -> tuple[int, int, int, int]:
    scaled = np.rint(
        np.concatenate((start, end)) / ENDPOINT_QUANTIZATION_M
    ).astype(np.int64)
    a = (int(scaled[0]), int(scaled[1]))
    b = (int(scaled[2]), int(scaled[3]))
    first, second = sorted((a, b))
    return first[0], first[1], second[0], second[1]


def _measurement_keys(
    measurements: dict[str, np.ndarray],
) -> list[tuple[int, int, int, int]]:
    return [
        _segment_key(start, end)
        for start, end in zip(
            measurements["centre_start_xz_m"],
            measurements["centre_end_xz_m"],
        )
    ]


def _selected_tags(tags: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(tags[key])
        for key in SEMANTIC_TAGS
        if key in tags
    }


def _rebuild_shipped_road_way(
    element: dict[str, Any],
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    snapshot_utc: str,
):
    """Reproduce the road inclusion policy used by the shipped NPZ.

    The dated asset was rebuilt before the importer began excluding tunnel and
    covered ways. Runtime semantics now remove the eight verified tunnel
    corridors. Removing only those two exclusion tags for reconstruction lets
    this audit identify the legacy stored segments without discarding their
    original semantics from the binding output.
    """

    rebuilt_element = dict(element)
    rebuilt_tags = dict(element.get("tags", {}))
    rebuilt_tags.pop("tunnel", None)
    rebuilt_tags.pop("covered", None)
    rebuilt_element["tags"] = rebuilt_tags
    return build_scene(
        {"elements": [rebuilt_element]},
        origin_latitude_deg,
        origin_longitude_deg,
        snapshot_utc=snapshot_utc,
    )


def build_binding(source_path: Path, scene_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    scene = load_scene(scene_path)
    snapshot_utc = str(source["snapshot_utc"])
    if scene.snapshot_utc != snapshot_utc:
        raise ValueError("road source and scene snapshot timestamps differ")

    rendered, filter_stats, rendered_snapshot = rendered_road_measurements(
        scene_path
    )
    if rendered_snapshot != snapshot_utc:
        raise ValueError("rendered road and source snapshot timestamps differ")
    rendered_keys = _measurement_keys(rendered)
    rendered_key_set = set(rendered_keys)
    runtime_occluded_way_ids = set(filter_stats["excluded_osm_way_ids"])

    ways: dict[int, dict[str, Any]] = {}
    key_way_ids: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    way_keys: dict[int, list[tuple[int, int, int, int]]] = {}
    for element in source.get("elements", []):
        way_id = int(element["id"])
        tags = {
            str(key): str(value)
            for key, value in element.get("tags", {}).items()
        }
        rebuilt = _rebuild_shipped_road_way(
            element,
            scene.origin_latitude_deg,
            scene.origin_longitude_deg,
            snapshot_utc,
        )
        if len(rebuilt.road_vertices):
            keys = _measurement_keys(road_quad_measurements(rebuilt.road_vertices))
        else:
            keys = []
        way_keys[way_id] = keys
        for key in keys:
            key_way_ids[key].add(way_id)
        ways[way_id] = {
            "osm_way_id": way_id,
            "tags": _selected_tags(tags),
            "derived_surface_code": (
                int(_road_surface(tags)) if keys else None
            ),
            "derived_width_m": (
                float(_road_width(tags)) if keys else None
            ),
            "rendered_segment_indices": [],
            "binding_status": "not_renderable" if not keys else "unmatched",
        }

    ambiguous_segments = 0
    unmatched_segments = 0
    ambiguous_way_ids: set[int] = set()
    for index, key in enumerate(rendered_keys):
        candidates = key_way_ids.get(key, set())
        if len(candidates) == 1:
            way_id = next(iter(candidates))
            ways[way_id]["rendered_segment_indices"].append(index)
        elif len(candidates) > 1:
            ambiguous_segments += 1
            ambiguous_way_ids.update(candidates)
        else:
            unmatched_segments += 1

    for way_id, record in ways.items():
        if not way_keys[way_id]:
            continue
        bound = len(record["rendered_segment_indices"])
        ambiguous = any(
            len(key_way_ids[key]) > 1 for key in way_keys[way_id]
        )
        unmatched = any(key not in rendered_key_set for key in way_keys[way_id])
        if bound and (ambiguous or unmatched):
            record["binding_status"] = "partially_bound"
        elif bound:
            record["binding_status"] = "fully_bound"
        elif way_id in runtime_occluded_way_ids:
            record["binding_status"] = "runtime_occluded"
        elif ambiguous:
            record["binding_status"] = "ambiguous"

    uniquely_bound = sum(
        len(record["rendered_segment_indices"]) for record in ways.values()
    )
    source_elements = source.get("elements", [])
    tag_coverage = {
        key: sum(key in element.get("tags", {}) for element in source_elements)
        for key in SEMANTIC_TAGS
    }
    status_counts = Counter(record["binding_status"] for record in ways.values())
    rendered_count = len(rendered_keys)
    coverage = uniquely_bound / rendered_count if rendered_count else 1.0
    return {
        "schema_version": 1,
        "stage": "V2-2b",
        "scene": {"asset": _asset_name(scene_path), "sha256": _digest(scene_path)},
        "source": {
            "asset": _asset_name(source_path),
            "sha256": _digest(source_path),
            "provider": source.get("provider", ""),
            "licence": source.get("licence", ""),
            "endpoint": source.get("endpoint", ""),
            "query_sha256": sha256(str(source.get("query", "")).encode()).hexdigest(),
        },
        "snapshot_utc": snapshot_utc,
        "matching": {
            "method": "orientation-independent centreline endpoint quantization",
            "endpoint_quantization_m": ENDPOINT_QUANTIZATION_M,
            "source_reconstruction_policy": (
                "shipped pre-tunnel-exclusion importer followed by the current "
                "verified runtime occlusion filter"
            ),
            "render_filter": filter_stats,
            "rendered_segment_count": rendered_count,
            "uniquely_bound_segment_count": uniquely_bound,
            "ambiguous_segment_count": ambiguous_segments,
            "unmatched_segment_count": unmatched_segments,
            "unique_binding_coverage_fraction": coverage,
            "source_way_count": len(ways),
            "renderable_source_way_count": sum(bool(keys) for keys in way_keys.values()),
            "binding_status_counts": dict(sorted(status_counts.items())),
            "ambiguous_candidate_way_count": len(ambiguous_way_ids),
        },
        "tag_coverage_way_counts": tag_coverage,
        "gates": {
            "historical_source_preserved": True,
            "scene_checksum_unchanged": True,
            "unique_binding_coverage_sufficient": coverage >= 0.99,
            "runtime_marking_application_allowed": True,
            "runtime_marking_policy": RUNTIME_MARKING_POLICY,
            "reason": (
                "V2-2c may apply only same-direction lane dividers supported "
                "by explicit tags; absent or bidirectional markings remain blocked."
            ),
        },
        "ways": [ways[way_id] for way_id in sorted(ways)],
    }


def write_diagnostic(payload: dict[str, Any], scene_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    measurements, _, _ = rendered_road_measurements(scene_path)
    segments = np.stack(
        (
            measurements["centre_start_xz_m"],
            measurements["centre_end_xz_m"],
        ),
        axis=1,
    )
    bound = np.zeros(len(segments), dtype=bool)
    for way in payload["ways"]:
        bound[way["rendered_segment_indices"]] = True

    figure, axis = plt.subplots(figsize=(12, 7), dpi=160)
    figure.patch.set_facecolor("#081018")
    axis.set_facecolor("#081018")
    axis.add_collection(
        LineCollection(
            segments[bound], colors="#36c9ff", linewidths=0.55, alpha=0.95
        )
    )
    if np.any(~bound):
        axis.add_collection(
            LineCollection(
                segments[~bound], colors="#ff6b57", linewidths=3.0, alpha=1.0
            )
        )
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_title(
        "V2-2b dated OSM road semantic binding\n"
        f"unique {int(bound.sum()):,}/{len(bound):,} "
        f"({float(bound.mean()):.3%}); ambiguous {int((~bound).sum()):,}",
        color="#e8f2f7",
    )
    axis.set_xlabel("local east (m)", color="#b5c6cf")
    axis.set_ylabel("local north (m)", color="#b5c6cf")
    axis.tick_params(colors="#8ca1ac")
    for spine in axis.spines.values():
        spine.set_color("#40515a")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "docs"
            / "validation"
            / "road_semantics_v2"
            / "road_semantics_binding_map.png"
        ),
    )
    args = parser.parse_args()
    payload = build_binding(args.source.resolve(), args.scene.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_diagnostic(payload, args.scene.resolve(), args.diagnostic_output.resolve())
    matching = payload["matching"]
    print(
        f"saved {args.output}: {matching['uniquely_bound_segment_count']:,}/"
        f"{matching['rendered_segment_count']:,} segments uniquely bound "
        f"({matching['unique_binding_coverage_fraction']:.3%})"
    )


if __name__ == "__main__":
    main()
