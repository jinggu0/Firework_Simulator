from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pyproj import Transformer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("docs/validation/ngii_dem_v1/ngii_dem_metadata_report.json")
SCENE_BBOX_WGS84 = (126.910, 37.515, 126.960, 37.545)
ANALYSIS_EPSG = 5186


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_cp949_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="cp949", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key].strip())


def _scene_bbox_projected() -> tuple[float, float, float, float]:
    west, south, east, north = SCENE_BBOX_WGS84
    transform = Transformer.from_crs(4326, ANALYSIS_EPSG, always_xy=True)
    points = [
        transform.transform(longitude, latitude)
        for longitude in (west, east)
        for latitude in (south, north)
    ]
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _overlaps(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return (
        max(left[0], right[0]) <= min(left[2], right[2])
        and max(left[1], right[1]) <= min(left[3], right[3])
    )


def _sampled_union_covers(
    target: tuple[float, float, float, float],
    rectangles: list[tuple[float, float, float, float]],
    sample_count: int = 21,
) -> bool:
    for y_index in range(sample_count):
        y = target[1] + (target[3] - target[1]) * y_index / (sample_count - 1)
        for x_index in range(sample_count):
            x = target[0] + (target[2] - target[0]) * x_index / (
                sample_count - 1
            )
            if not any(
                rectangle[0] <= x <= rectangle[2]
                and rectangle[1] <= y <= rectangle[3]
                for rectangle in rectangles
            ):
                return False
    return True


def build_report(performance_csv: Path, basic_metadata_csv: Path) -> dict[str, Any]:
    performance_rows = _read_cp949_csv(performance_csv)
    basic_rows = _read_cp949_csv(basic_metadata_csv)
    scene_bbox = _scene_bbox_projected()
    candidates: list[dict[str, Any]] = []
    rectangles: list[tuple[float, float, float, float]] = []

    for row in performance_rows:
        if row.get("원시자료제작년도", "").strip() != "2009":
            continue
        if row.get("격자간격", "").replace(" ", "").lower() != "1mx1m":
            continue
        if row.get("원시자료획득방법", "").strip() != "항공레이저 측량":
            continue
        try:
            bounds = (
                _float(row, "원시좌하단의평면X좌표"),
                _float(row, "원시좌하단의평면Y좌표"),
                _float(row, "원시우상단의평면X좌표"),
                _float(row, "원시우상단의평면Y좌표"),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not _overlaps(scene_bbox, bounds):
            continue
        rectangles.append(bounds)
        candidates.append(
            {
                "sheet_id": row.get("도엽번호5000", "").strip(),
                "sheet_name": row.get("도엽명5000", "").strip(),
                "production_year": 2009,
                "grid_spacing_m": 1.0,
                "source_method": row.get("원시자료획득방법", "").strip(),
                "format": row.get("자료형식", "").strip(),
                "record_order": row.get("자료기록형식", "").strip(),
                "vertical_datum": row.get("표고기준", "").strip(),
                "height_type": row.get("높이값종류", "").strip(),
                "minimum_elevation_m": float(row["최저표고"]),
                "maximum_elevation_m": float(row["최고표고"]),
                "published_accuracy": row.get("정확도", "").strip() or None,
                "bounds_epsg5186": list(bounds),
            }
        )

    candidates.sort(key=lambda candidate: candidate["sheet_id"])
    accuracy_documented = bool(candidates) and all(
        candidate["published_accuracy"] is not None for candidate in candidates
    )

    return {
        "schema_version": 1,
        "observed_at": "2026-08-03",
        "sources": {
            "basic_metadata": {
                "file_name": basic_metadata_csv.name,
                "bytes": basic_metadata_csv.stat().st_size,
                "sha256": _digest(basic_metadata_csv),
                "row_count": len(basic_rows),
                "source_url": "https://www.data.go.kr/data/15067632/fileData.do",
            },
            "performance_metadata": {
                "file_name": performance_csv.name,
                "bytes": performance_csv.stat().st_size,
                "sha256": _digest(performance_csv),
                "row_count": len(performance_rows),
                "source_url": "https://www.data.go.kr/data/15067637/fileData.do",
            },
            "raw_files_redistributed": False,
        },
        "scene_scope": {
            "bbox_wgs84": list(SCENE_BBOX_WGS84),
            "analysis_crs": f"EPSG:{ANALYSIS_EPSG}",
            "bbox_epsg5186": [round(value, 3) for value in scene_bbox],
        },
        "legacy_lidar_candidates": {
            "count": len(candidates),
            "sheets": candidates,
            "sampled_scene_coverage": _sampled_union_covers(scene_bbox, rectangles),
            "coverage_sample_grid": [21, 21],
            "crs_interpretation": (
                "EPSG:5186 is used to reconcile the published GRS80 middle-origin "
                "bounds; the raster header is still required for final CRS verification."
            ),
        },
        "portal_delivery": {
            "requested_product": "2024 서울 37608 공개DEM",
            "submission_endpoint": (
                "https://map.ngii.go.kr/pd/shbtManage/"
                "shbtInsertTritVidoDem.do"
            ),
            "expected_response": "application/json orderDownList",
            "observed_http_status": 200,
            "observed_mime_type": "text/html",
            "observed_page": "요청하신 페이지를 찾을 수 없습니다",
            "raster_downloaded": False,
        },
        "validation_gate": {
            "maximum_plan_registration_rmse_m": 0.25,
            "maximum_vertical_uncertainty_m": 0.1,
            "published_accuracy_documented": accuracy_documented,
            "provider_raster_checksum_available": False,
            "scene_merge_allowed": False,
            "scene_vertices_modified": 0,
            "blocking_reasons": [
                "the 2024 provider raster was not delivered",
                "the portal DEM endpoint returned a page-not-found HTML document",
                "the metadata snapshot documents 2009 legacy LiDAR, not the 2024 package",
                "published accuracy is blank for all overlapping 1 m candidates",
                "CRS and NoData are not verified from a delivered raster header",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("performance_csv", type=Path)
    parser.add_argument("basic_metadata_csv", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(arguments.performance_csv, arguments.basic_metadata_csv)
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: candidates={report['legacy_lidar_candidates']['count']}, "
        f"scene_merge_allowed={report['validation_gate']['scene_merge_allowed']}"
    )


if __name__ == "__main__":
    main()
