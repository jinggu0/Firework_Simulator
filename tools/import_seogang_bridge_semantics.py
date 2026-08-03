"""Build the dated Seogang Bridge plan-replacement evidence asset."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import urllib.parse
import urllib.request

import numpy as np

from simulator.scene import LocalTangentPlane


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "assets" / "seogang_bridge_semantics_2024-10-05.json"
)
SNAPSHOT_UTC = "2024-10-05T10:20:00Z"
OUTLINE_WAY_ID = 875036744
REQUIRED_WAY_IDS = (
    338588827,
    620357093,
    634521303,
    634521306,
    875036744,
    910747602,
    910747603,
)
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"


def _query() -> str:
    ids = ",".join(str(value) for value in REQUIRED_WAY_IDS)
    return (
        f'[out:json][timeout:90][date:"{SNAPSHOT_UTC}"];'
        f"way(id:{ids});out geom tags;"
    )


def _download() -> dict:
    request = urllib.request.Request(
        OVERPASS_URL,
        data=urllib.parse.urlencode({"data": _query()}).encode(),
        headers={"User-Agent": "Firework-Simulator/0.2 bridge-evidence"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segment)))
    target = np.linspace(0.0, distance[-1], count)
    return np.column_stack(
        [np.interp(target, distance, points[:, axis]) for axis in range(2)]
    )


def build_document(payload: dict) -> dict:
    elements = {int(item["id"]): item for item in payload.get("elements", [])}
    missing = set(REQUIRED_WAY_IDS) - set(elements)
    if missing:
        raise ValueError(f"historical Overpass result lacks ways {sorted(missing)}")
    outline = elements[OUTLINE_WAY_ID]
    geometry = outline.get("geometry", [])
    if len(geometry) != 36 or geometry[0] != geometry[-1]:
        raise ValueError("dated Seogang outline no longer matches the reviewed 36-node ring")

    plane = LocalTangentPlane(37.529, 126.935)
    ring = np.asarray(
        [plane.to_local(point["lat"], point["lon"])[[0, 2]] for point in geometry],
        dtype=np.float64,
    )
    # The frozen outline contains a six-node south cap (0..5), side A
    # (5..18), a four-node north cap (18..21), and side B (21..34 plus 0).
    # Indices are asserted above rather than guessed from a current map.
    side_a = ring[5:19]
    side_b = ring[np.r_[0, np.arange(34, 20, -1)]]
    side_a = _resample(side_a, 65)
    side_b = _resample(side_b, 65)
    centre = 0.5 * (side_a + side_b)
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1)))
    )
    widths = np.linalg.norm(side_b - side_a, axis=1)
    official_component_length_m = 1320.0 + 386.0
    selected = [elements[value] for value in sorted(elements)]
    source_hash = sha256(
        json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    scene_hash = sha256(
        (REPOSITORY_ROOT / "assets" / "yeouido_scene.npz").read_bytes()
    ).hexdigest()

    return {
        "schema_version": 1,
        "target_event_date": "2024-10-05",
        "snapshot_utc": SNAPSHOT_UTC,
        "bridge_id": "seogang_bridge",
        "scene_asset": "assets/yeouido_scene.npz",
        "scene_asset_sha256": scene_hash,
        "coordinate_system": "local East-Up-South metres; arrays store X/Z",
        "local_origin_wgs84": [37.529, 126.935],
        "sources": {
            "historical_plan": {
                "provider": "OpenStreetMap contributors via historical Overpass query",
                "licence": "ODbL 1.0",
                "confidence_grade": "C",
                "query": _query(),
                "endpoint": OVERPASS_URL,
                "retrieved_date": "2026-08-03",
                "selected_payload_sha256": source_hash,
                "notes": "Dated plan semantics, not a survey or a vertical profile."
            },
            "official_2024_paint_contract": {
                "provider": "서울특별시 건설알림이",
                "url": "https://cis.seoul.go.kr/TotalAlimi_new/PopInfo.action?cmd=info1&pjt_cd=7182024022199",
                "confidence_grade": "A",
                "work_period": ["2024-02-28", "2025-11-27"],
                "scope": {
                    "lower_steel_paint_area_m2": 17575.0,
                    "upper_steel_paint_area_m2": 2726.0,
                    "suspended_scaffold_area_m2": 14556.0
                },
                "event_day_status": "inside_contract_period_but_progress_location_unpublished"
            },
            "official_2024_budget_review": {
                "provider": "서울특별시의회",
                "url": "https://ms.smc.seoul.kr/attach/record/SEOUL/appendix/a11/A0062073.pdf?time=20260618031117",
                "confidence_grade": "A",
                "attachment_sha256": "64d124d1df186748d6de82cb3ee60a36b1d89b70e9aa7617b521e2574229db95",
                "reviewed_pdf_page": 22,
                "fact": "2024 budget 4,350,000 thousand KRW for continuing Seogang Bridge paint repair"
            }
        },
        "way_roles": {
            "outline": OUTLINE_WAY_ID,
            "south_carriageways": [910747602, 910747603],
            "north_carriageways": [338588827, 620357093],
            "main_sidewalks": [634521303, 634521306]
        },
        "deck_outline_xz_m": np.round(ring, 3).tolist(),
        "replacement_cross_sections": [
            {
                "osm_plan_distance_m": round(float(along), 3),
                "side_a_xz_m": np.round(left, 3).tolist(),
                "side_b_xz_m": np.round(right, 3).tolist()
            }
            for along, left, right in zip(distance, side_a, side_b)
        ],
        "plan_metrics": {
            "replacement_centreline_length_m": float(distance[-1]),
            "official_main_plus_north_approach_length_m": official_component_length_m,
            "length_residual_m": float(distance[-1] - official_component_length_m),
            "width_min_p50_p95_max_m": [
                float(widths.min()),
                float(np.percentile(widths, 50)),
                float(np.percentile(widths, 95)),
                float(widths.max())
            ],
            "station_registration_status": "failed_normalized_length_only",
            "station_registration_plan_rmse_m": None,
            "notes": "OSM plan distance cannot be substituted for completion-drawing station."
        },
        "render_policy": {
            "matching_distance_m": 8.0,
            "minimum_matching_source_segments": 80,
            "replacement_local_y_m": 7.0,
            "replacement_surface_code": 2.0,
            "horizontal_policy": "replace overlapping generic strips inside the dated bridge outline with one cross-section strip",
            "vertical_policy": "unchanged grade-D terrain-relative offset; V1-6 vertical gate remains closed",
            "construction_policy": "do not draw scaffolding or fresh-paint boundaries until an event-day progress photo is registered"
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    document = build_document(_download())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {arguments.output}: {len(document['replacement_cross_sections']) - 1} "
        f"deck segments, station residual {document['plan_metrics']['length_residual_m']:.3f} m"
    )


if __name__ == "__main__":
    main()
