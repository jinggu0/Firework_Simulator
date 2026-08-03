"""Acquire and register a local-only S-Map 2024 orthophoto focus crop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from simulator.geodesy import LocalTangentPlane
from simulator.road_detail_semantics import (
    DEFAULT_ROAD_DETAIL_SEMANTICS_PATH,
    load_road_detail_semantics,
)
from simulator.scene import load_scene
from simulator.validation.road_details import (
    DEFAULT_SCENE_PATH,
    rendered_road_measurements,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_REPORT = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_marking_evidence_v2"
    / "road_marking_evidence_report.json"
)
DEFAULT_LOCAL_OUTPUT = (
    REPOSITORY_ROOT / "local_reference" / "smap_2024_ortho"
)
DEFAULT_REPORT = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_marking_registration_v2"
    / "smap_2024_registration_report.json"
)
DEFAULT_PREVIEW = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_marking_registration_v2"
    / "smap_2024_candidate_coverage.png"
)
SMAP_SERVICE_URL = "https://smap.seoul.go.kr/"
SMAP_GUIDE_URL = "https://smap.seoul.go.kr/guide/guide.html"
SMAP_MAP_CONST_URL = "https://smap.seoul.go.kr/js/mapConst.js"
SMAP_MAP_CONST_SHA256 = (
    "4d96227c826048e6ed8ee557372b3dfa48b2cc06a4c00965785969ecc15f1cd1"
)
SMAP_MAP_CONST_BYTES = 104_281
SMAP_CRS = "EPSG:5186"
SMAP_PROJ4 = (
    "+ellps=GRS80 +proj=tmerc +lat_0=38 +lon_0=127 +k=1 "
    "+x_0=200000 +y_0=600000 +units=m +no_defs"
)
SMAP_ORIGIN_M = (153_468.0, 513_944.0)
SMAP_EXTENT_M = (153_468.0, 513_944.0, 219_004.0, 579_478.0)
SMAP_RESOLUTIONS_M_PER_PX = (
    256.0,
    128.0,
    64.0,
    32.0,
    16.0,
    8.0,
    4.0,
    2.0,
    1.0,
    0.5,
    0.25,
)
SMAP_TILE_SIZE_PX = 256
SMAP_ZOOM = 10
SMAP_LAYER = "ortho_drone_25cm_2024"
DEFAULT_TILE_RANGE = (634, 639, 520, 525)
USER_AGENT = "Firework-Simulator local visual validation"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _asset(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _local_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def tile_span_m(zoom: int = SMAP_ZOOM) -> float:
    return SMAP_RESOLUTIONS_M_PER_PX[zoom] * SMAP_TILE_SIZE_PX


def tile_bounds(x: int, y: int, zoom: int = SMAP_ZOOM) -> tuple[float, ...]:
    span = tile_span_m(zoom)
    west = SMAP_ORIGIN_M[0] + x * span
    south = SMAP_ORIGIN_M[1] + y * span
    return west, south, west + span, south + span


def coordinate_tile(easting_m: float, northing_m: float) -> tuple[int, int]:
    span = tile_span_m()
    return (
        math.floor((easting_m - SMAP_ORIGIN_M[0]) / span),
        math.floor((northing_m - SMAP_ORIGIN_M[1]) / span),
    )


def tile_url(x: int, y: int, zoom: int = SMAP_ZOOM) -> str:
    alias = x % 3 + 1
    return (
        f"https://smap{alias}.eseoul.go.kr:5432/tile.sqlite/"
        f"{SMAP_LAYER}/{zoom}/{x}/{y}.jpg"
    )


def _download_tile(url: str, output: Path) -> str:
    if output.is_file():
        with Image.open(output) as image:
            image.verify()
        return _digest(output)
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Referer": SMAP_SERVICE_URL},
    )
    with urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"S-Map tile returned HTTP {response.status}: {url}")
        content_type = response.headers.get_content_type()
        if content_type != "image/jpeg":
            raise RuntimeError(f"S-Map tile is not JPEG ({content_type}): {url}")
        payload = response.read()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    with Image.open(output) as image:
        image.verify()
    return sha256(payload).hexdigest()


def _projected_candidate_coverage(
    evidence_path: Path,
    scene_path: Path,
    binding_path: Path,
    tile_range: tuple[int, int, int, int],
) -> dict[str, Any]:
    try:
        import pyproj
    except ImportError as error:
        raise RuntimeError("pyproj is required for S-Map registration") from error
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    scene = load_scene(scene_path)
    measurements, _, _ = rendered_road_measurements(scene_path)
    semantics = load_road_detail_semantics(binding_path)
    by_way_id = {way.osm_way_id: way for way in semantics.ways}
    plane = LocalTangentPlane(
        scene.origin_latitude_deg, scene.origin_longitude_deg
    )
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", SMAP_CRS, always_xy=True
    )
    min_x, max_x, min_y, max_y = tile_range
    coverage: dict[str, Any] = {}
    projected_segments: list[dict[str, Any]] = []
    for group_name, group in evidence["candidate_groups"].items():
        way_ids = {int(record["osm_way_id"]) for record in group["records"]}
        group_way_ids: set[int] = set()
        group_segments = 0
        group_tiles: set[tuple[int, int]] = set()
        for way_id in way_ids:
            way = by_way_id[way_id]
            for index in way.rendered_segment_indices:
                midpoint = measurements["centre_midpoint_xz_m"][index]
                latitude, longitude, _ = plane.to_geodetic(
                    np.array([midpoint[0], 0.0, midpoint[1]], dtype=np.float64)
                )
                easting, northing = transformer.transform(longitude, latitude)
                tile_x, tile_y = coordinate_tile(easting, northing)
                if not (
                    min_x <= tile_x <= max_x and min_y <= tile_y <= max_y
                ):
                    continue
                group_segments += 1
                group_way_ids.add(way_id)
                group_tiles.add((tile_x, tile_y))
                projected_segments.append(
                    {
                        "group": group_name,
                        "osm_way_id": way_id,
                        "easting_m": float(easting),
                        "northing_m": float(northing),
                    }
                )
        coverage[group_name] = {
            "way_count": len(group_way_ids),
            "rendered_segment_count": group_segments,
            "tile_count": len(group_tiles),
            "osm_way_ids": sorted(group_way_ids),
        }
    return {"groups": coverage, "segments": projected_segments}


def _write_mosaic(
    tile_records: list[dict[str, Any]],
    output_dir: Path,
    tile_range: tuple[int, int, int, int],
) -> tuple[Path, Path]:
    min_x, max_x, min_y, max_y = tile_range
    columns = max_x - min_x + 1
    rows = max_y - min_y + 1
    mosaic = Image.new(
        "RGB",
        (columns * SMAP_TILE_SIZE_PX, rows * SMAP_TILE_SIZE_PX),
    )
    record_by_tile = {
        (int(record["x"]), int(record["y"])): record
        for record in tile_records
    }
    for tile_y in range(max_y, min_y - 1, -1):
        for tile_x in range(min_x, max_x + 1):
            record = record_by_tile[(tile_x, tile_y)]
            with Image.open(output_dir / record["local_asset"]) as tile:
                mosaic.paste(
                    tile.convert("RGB"),
                    (
                        (tile_x - min_x) * SMAP_TILE_SIZE_PX,
                        (max_y - tile_y) * SMAP_TILE_SIZE_PX,
                    ),
                )
    mosaic_path = output_dir / (
        f"focus_z{SMAP_ZOOM}_x{min_x}-{max_x}_y{min_y}-{max_y}.png"
    )
    mosaic.save(mosaic_path)
    resolution = SMAP_RESOLUTIONS_M_PER_PX[SMAP_ZOOM]
    west = tile_bounds(min_x, min_y)[0]
    north = tile_bounds(max_x, max_y)[3]
    world_path = mosaic_path.with_suffix(".pgw")
    world_path.write_text(
        "\n".join(
            (
                f"{resolution:.12f}",
                "0.000000000000",
                "0.000000000000",
                f"{-resolution:.12f}",
                f"{west + resolution * 0.5:.12f}",
                f"{north - resolution * 0.5:.12f}",
            )
        )
        + "\n",
        encoding="ascii",
    )
    return mosaic_path, world_path


def _write_candidate_preview(
    coverage: dict[str, Any],
    tile_range: tuple[int, int, int, int],
    output: Path,
) -> None:
    from PIL import ImageDraw

    min_x, max_x, min_y, max_y = tile_range
    span = tile_span_m()
    west, south = tile_bounds(min_x, min_y)[:2]
    east, north = tile_bounds(max_x, max_y)[2:]
    width = 1024
    height = 1024
    image = Image.new("RGB", (width, height), (8, 16, 24))
    draw = ImageDraw.Draw(image)
    for x in range(min_x, max_x + 2):
        px = round((SMAP_ORIGIN_M[0] + x * span - west) / (east - west) * width)
        draw.line((px, 0, px, height), fill=(42, 58, 69), width=1)
    for y in range(min_y, max_y + 2):
        py = height - round(
            (SMAP_ORIGIN_M[1] + y * span - south) / (north - south) * height
        )
        draw.line((0, py, width, py), fill=(42, 58, 69), width=1)
    colours = {
        "already_rendered_oneway_lane_dividers": (48, 202, 255),
        "centre_line_candidates_non_oneway_yes_multilane": (255, 198, 74),
        "directional_lane_count_candidates": (238, 94, 166),
        "cycle_lane_tag_candidates": (82, 214, 129),
        "crossing_marking_presence_candidates": (255, 111, 82),
    }
    for segment in coverage["segments"]:
        px = round((segment["easting_m"] - west) / (east - west) * width)
        py = height - round(
            (segment["northing_m"] - south) / (north - south) * height
        )
        colour = colours[segment["group"]]
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=colour)
    draw.rectangle((16, 16, 610, 122), fill=(20, 29, 39))
    draw.text((28, 28), "V2-2e S-Map 2024 local registration crop", fill="white")
    draw.text(
        (28, 53),
        f"EPSG:5186 / 0.25 m px / {east - west:.0f} x {north - south:.0f} m",
        fill=(210, 220, 228),
    )
    cycle = coverage["groups"]["cycle_lane_tag_candidates"]
    draw.text(
        (28, 78),
        f"cycle candidates: {cycle['way_count']} ways / "
        f"{cycle['rendered_segment_count']} segments",
        fill=colours["cycle_lane_tag_candidates"],
    )
    draw.text(
        (28, 101),
        "Provider pixels excluded from repository; dots show bound midpoints.",
        fill=(180, 190, 198),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def acquire(
    output_dir: Path,
    report_path: Path,
    preview_path: Path,
    tile_range: tuple[int, int, int, int],
    evidence_path: Path,
    scene_path: Path,
    binding_path: Path,
) -> dict[str, Any]:
    min_x, max_x, min_y, max_y = tile_range
    if min_x > max_x or min_y > max_y:
        raise ValueError("S-Map tile range is inverted")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = output_dir / "manifest.json"
    acquired_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    existing: dict[str, Any] = {}
    if existing_manifest_path.is_file():
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing.get("tile_range") == [min_x, max_x, min_y, max_y]:
            acquired_utc = str(existing["acquired_utc"])
    tile_records: list[dict[str, Any]] = []
    for tile_y in range(min_y, max_y + 1):
        for tile_x in range(min_x, max_x + 1):
            relative = Path(str(SMAP_ZOOM)) / str(tile_x) / f"{tile_y}.jpg"
            url = tile_url(tile_x, tile_y)
            digest = _download_tile(url, output_dir / relative)
            tile_records.append(
                {
                    "zoom": SMAP_ZOOM,
                    "x": tile_x,
                    "y": tile_y,
                    "url": url,
                    "local_asset": relative.as_posix(),
                    "sha256": digest,
                    "bounds_epsg5186_m": list(tile_bounds(tile_x, tile_y)),
                }
            )
    mosaic_path, world_path = _write_mosaic(tile_records, output_dir, tile_range)
    west, south = tile_bounds(min_x, min_y)[:2]
    east, north = tile_bounds(max_x, max_y)[2:]
    local_manifest = {
        "schema_version": 1,
        "acquired_utc": acquired_utc,
        "source": SMAP_SERVICE_URL,
        "layer": SMAP_LAYER,
        "tile_range": [min_x, max_x, min_y, max_y],
        "bbox_epsg5186_m": [west, south, east, north],
        "redistribution": "not authorised; local validation only",
        "initial_downloaded_tile_count": int(
            existing.get("initial_downloaded_tile_count", len(tile_records))
        ),
        "tiles": tile_records,
        "mosaic": {
            "asset": mosaic_path.name,
            "sha256": _digest(mosaic_path),
            "world_file": world_path.name,
            "world_file_sha256": _digest(world_path),
        },
    }
    existing_manifest_path.write_text(
        json.dumps(local_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    coverage = _projected_candidate_coverage(
        evidence_path, scene_path, binding_path, tile_range
    )
    _write_candidate_preview(coverage, tile_range, preview_path)
    report = {
        "schema_version": 1,
        "stage": "V2-2e",
        "source": {
            "service": SMAP_SERVICE_URL,
            "official_guide": SMAP_GUIDE_URL,
            "map_configuration": {
                "url": SMAP_MAP_CONST_URL,
                "sha256": SMAP_MAP_CONST_SHA256,
                "bytes": SMAP_MAP_CONST_BYTES,
                "browser_verified_utc": acquired_utc,
            },
            "provider_attribution": "© 서울특별시 모든 권리 보유",
            "guide_allows_map_view_png_download": True,
            "raw_tile_redistribution_authorised": False,
            "raw_pixels_committed_to_repository": False,
        },
        "layer_contract": {
            "layer": SMAP_LAYER,
            "code_comment": "[202410] 2024 orthophoto - 25 cm external network 20241209",
            "code_comment_period_hint": "2024-10",
            "exact_imagery_acquisition_date": None,
            "event_date": "2024-10-05",
            "event_date_applicability_confirmed": False,
            "crs": SMAP_CRS,
            "proj4": SMAP_PROJ4,
            "origin_m": list(SMAP_ORIGIN_M),
            "extent_m": list(SMAP_EXTENT_M),
            "resolutions_m_per_pixel": list(SMAP_RESOLUTIONS_M_PER_PX),
            "zoom": SMAP_ZOOM,
            "native_resolution_m_per_pixel": SMAP_RESOLUTIONS_M_PER_PX[
                SMAP_ZOOM
            ],
            "tile_size_px": [SMAP_TILE_SIZE_PX, SMAP_TILE_SIZE_PX],
        },
        "local_acquisition": {
            "manifest_asset": _local_path(existing_manifest_path),
            "manifest_sha256": _digest(existing_manifest_path),
            "tile_range": [min_x, max_x, min_y, max_y],
            "tile_count": len(tile_records),
            "initial_downloaded_tile_count": local_manifest[
                "initial_downloaded_tile_count"
            ],
            "verified_this_run": len(tile_records),
            "bbox_epsg5186_m": [west, south, east, north],
            "ground_coverage_m": [east - west, north - south],
            "mosaic_pixel_size": [
                (max_x - min_x + 1) * SMAP_TILE_SIZE_PX,
                (max_y - min_y + 1) * SMAP_TILE_SIZE_PX,
            ],
            "mosaic_sha256": _digest(mosaic_path),
            "world_file_sha256": _digest(world_path),
            "tile_records": tile_records,
        },
        "registration": {
            "method": "provider EPSG:5186 tile grid; no fitted control points",
            "pixel_origin": "north-west pixel centre",
            "pixel_size_m": [0.25, -0.25],
            "grid_registration_residual_m": 0.0,
            "independent_check_point_residual_m": None,
            "provider_grid_registration_passes": True,
            "independent_spatial_check_passes": False,
        },
        "candidate_coverage": coverage["groups"],
        "inputs": {
            "event_evidence_report": {
                "asset": _asset(evidence_path),
                "sha256": _digest(evidence_path),
            },
            "scene": {"asset": _asset(scene_path), "sha256": _digest(scene_path)},
            "binding": {
                "asset": _asset(binding_path),
                "sha256": _digest(binding_path),
            },
        },
        "diagnostic": {
            "asset": _asset(preview_path),
            "sha256": _digest(preview_path),
        },
        "application_gates": {
            "provider_pixels_acquired_locally": True,
            "provider_crs_and_grid_explicit": True,
            "native_25cm_resolution_confirmed": True,
            "exact_imagery_acquisition_date_confirmed": False,
            "independent_spatial_check_passes": False,
            "event_date_marking_classification_allowed": False,
            "runtime_geometry_changed_by_this_stage": False,
            "reason": (
                "The provider grid is directly registered, but a 2024-10 period "
                "hint does not prove capture on or before 2024-10-05. Raw-pixel "
                "redistribution and independent check-point terms also remain open."
            ),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-local-validation",
        action="store_true",
        help="acknowledge local-only use and download provider pixels",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOCAL_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE_REPORT)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--binding", type=Path, default=DEFAULT_ROAD_DETAIL_SEMANTICS_PATH
    )
    parser.add_argument("--min-tile-x", type=int, default=DEFAULT_TILE_RANGE[0])
    parser.add_argument("--max-tile-x", type=int, default=DEFAULT_TILE_RANGE[1])
    parser.add_argument("--min-tile-y", type=int, default=DEFAULT_TILE_RANGE[2])
    parser.add_argument("--max-tile-y", type=int, default=DEFAULT_TILE_RANGE[3])
    args = parser.parse_args()
    if not args.accept_local_validation:
        parser.error("--accept-local-validation is required")
    report = acquire(
        args.output_dir.resolve(),
        args.report.resolve(),
        args.preview.resolve(),
        (args.min_tile_x, args.max_tile_x, args.min_tile_y, args.max_tile_y),
        args.evidence.resolve(),
        args.scene.resolve(),
        args.binding.resolve(),
    )
    cycle = report["candidate_coverage"]["cycle_lane_tag_candidates"]
    print(
        f"acquired {report['local_acquisition']['tile_count']} local-only tiles; "
        f"registered {cycle['way_count']} cycle ways / "
        f"{cycle['rendered_segment_count']} rendered segments"
    )


if __name__ == "__main__":
    main()
