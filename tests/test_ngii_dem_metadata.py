from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.audit_ngii_dem_metadata import build_report


REPORT = Path("docs/validation/ngii_dem_v1/ngii_dem_metadata_report.json")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="cp949", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_legacy_metadata_never_unlocks_the_scene(tmp_path: Path) -> None:
    performance = tmp_path / "performance.csv"
    basic = tmp_path / "basic.csv"
    row = {
        "도엽번호5000": "37608088",
        "도엽명5000": "서울088",
        "원시자료제작년도": "2009",
        "격자간격": "1m x 1m",
        "원시자료획득방법": "항공레이저 측량",
        "원시좌하단의평면X좌표": "190000",
        "원시좌하단의평면Y좌표": "540000",
        "원시우상단의평면X좌표": "200000",
        "원시우상단의평면Y좌표": "555000",
        "자료형식": "ASCII",
        "자료기록형식": "Upper-Left",
        "표고기준": "인천항의 평균해수면",
        "높이값종류": "정표고",
        "최저표고": "1.72",
        "최고표고": "99.27",
        "정확도": "",
    }
    _write_csv(performance, [row])
    _write_csv(basic, [{"도엽번호5000": "37608088"}])

    report = build_report(performance, basic)

    assert report["legacy_lidar_candidates"]["count"] == 1
    assert report["legacy_lidar_candidates"]["sampled_scene_coverage"]
    assert report["legacy_lidar_candidates"]["sheets"][0]["vertical_datum"] == (
        "인천항의 평균해수면"
    )
    assert not report["validation_gate"]["published_accuracy_documented"]
    assert not report["validation_gate"]["scene_merge_allowed"]
    assert report["validation_gate"]["scene_vertices_modified"] == 0


def test_committed_metadata_audit_locks_six_sheets_and_blocks_merge() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["legacy_lidar_candidates"]["count"] == 6
    assert [
        sheet["sheet_id"]
        for sheet in report["legacy_lidar_candidates"]["sheets"]
    ] == [
        "37608087",
        "37608088",
        "37608089",
        "37608097",
        "37608098",
        "37608099",
    ]
    assert not report["validation_gate"]["published_accuracy_documented"]
    assert not report["validation_gate"]["provider_raster_checksum_available"]
    assert not report["validation_gate"]["scene_merge_allowed"]
