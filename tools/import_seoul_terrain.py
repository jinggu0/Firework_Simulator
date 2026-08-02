"""Refine the scene terrain from official Seoul contour and spot heights.

The source is Seoul Open Data OA-22241, the 2023 NGII-notified 1:5,000
topographic contour/spot-height dataset. The output grid is denser than the
source observations only to avoid renderer tessellation artefacts; it is not
described as a 5 m survey or a substitute for the non-public NGII 1 m DEM.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.parse
import urllib.request
import zipfile

import numpy as np

from simulator.geodesy import LocalTangentPlane
from simulator.scene import load_scene, save_scene
from simulator.terrain import constrained_heightmap, sample_heightmap_array

DATASET_PAGE = "https://data.seoul.go.kr/dataList/OA-22241/F/1/datasetView.do"
DOWNLOAD_URL = (
    "https://datafile.seoul.go.kr/bigfile/iot/inf/"
    "nio_download.do?&useCache=false"
)
SOURCE_ARCHIVE_SHA256 = (
    "4fbe3c7e061b5974e7403ec116855304ed8ae321eebcc0d12c31ca8fb7be30bf"
)


def _dependencies():
    try:
        import pyproj
        import shapefile
    except ImportError as error:
        raise RuntimeError(
            "install requirements-terrain.txt before importing Seoul terrain"
        ) from error
    return pyproj, shapefile


def _download_archive(path: Path) -> str:
    request = urllib.request.Request(
        DOWNLOAD_URL,
        data=urllib.parse.urlencode(
            {"infId": "OA-22241", "seqNo": "", "seq": "2", "infSeq": "1"}
        ).encode(),
        headers={"User-Agent": "FireworkSimulator/0.2 (terrain provenance)"},
    )
    digest = sha256()
    with urllib.request.urlopen(request, timeout=300) as response, path.open(
        "wb"
    ) as output:
        while block := response.read(1 << 20):
            digest.update(block)
            output.write(block)
    checksum = digest.hexdigest()
    if checksum != SOURCE_ARCHIVE_SHA256:
        raise RuntimeError(
            "Seoul terrain archive changed: "
            f"expected {SOURCE_ARCHIVE_SHA256}, received {checksum}"
        )
    return checksum


def _scene_projected_bounds(scene, transformer, margin_m: float) -> np.ndarray:
    minimum_x, minimum_z, maximum_x, maximum_z = scene.terrain_bounds
    corners = np.array(
        [
            [minimum_x, 0.0, minimum_z],
            [minimum_x, 0.0, maximum_z],
            [maximum_x, 0.0, minimum_z],
            [maximum_x, 0.0, maximum_z],
        ],
        dtype=np.float64,
    )
    plane = LocalTangentPlane(
        scene.origin_latitude_deg, scene.origin_longitude_deg
    )
    geodetic = np.array([plane.to_geodetic(corner) for corner in corners])
    east, north = transformer.transform(geodetic[:, 1], geodetic[:, 0])
    return np.array(
        [
            np.min(east) - margin_m,
            np.min(north) - margin_m,
            np.max(east) + margin_m,
            np.max(north) + margin_m,
        ]
    )


def _intersects(first: list[float], second: np.ndarray) -> bool:
    return not (
        first[2] < second[0]
        or first[0] > second[2]
        or first[3] < second[1]
        or first[1] > second[3]
    )


def _thin_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    targets = np.arange(0.0, distance[-1] + 1e-6, spacing_m)
    if not len(targets):
        return points[:1].copy()
    return np.column_stack(
        (
            np.interp(targets, distance, points[:, 0]),
            np.interp(targets, distance, points[:, 1]),
        )
    )


def _contour_samples(
    path: Path,
    source_bounds: np.ndarray,
    scene_bounds: np.ndarray,
    inverse_transformer,
    plane: LocalTangentPlane,
    spacing_m: float,
    margin_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    _, shapefile = _dependencies()
    positions: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    feature_count = 0
    reader = shapefile.Reader(str(path), encoding="cp949")
    expanded = scene_bounds + np.array(
        [-margin_m, -margin_m, margin_m, margin_m]
    )
    for record in reader.iterShapeRecords():
        shape = record.shape
        if not _intersects(shape.bbox, source_bounds):
            continue
        source = np.asarray(shape.points, dtype=np.float64)
        longitude, latitude = inverse_transformer.transform(
            source[:, 0], source[:, 1]
        )
        local = plane.to_local_array(latitude, longitude)[:, [0, 2]]
        used = False
        parts = list(shape.parts) + [len(local)]
        for start, stop in zip(parts[:-1], parts[1:]):
            line = local[start:stop]
            inside = (
                (line[:, 0] >= expanded[0])
                & (line[:, 0] <= expanded[2])
                & (line[:, 1] >= expanded[1])
                & (line[:, 1] <= expanded[3])
            )
            indices = np.flatnonzero(inside)
            if not len(indices):
                continue
            cuts = np.flatnonzero(np.diff(indices) > 1) + 1
            for run in np.split(indices, cuts):
                sampled = _thin_polyline(line[run], spacing_m)
                if not len(sampled):
                    continue
                positions.append(sampled)
                heights.append(
                    np.full(len(sampled), float(record.record["HEIGHT"]))
                )
                used = True
        feature_count += int(used)
    return np.concatenate(positions), np.concatenate(heights), feature_count


def _spot_samples(
    path: Path,
    source_bounds: np.ndarray,
    inverse_transformer,
    plane: LocalTangentPlane,
) -> tuple[np.ndarray, np.ndarray, int]:
    _, shapefile = _dependencies()
    positions: list[np.ndarray] = []
    heights: list[float] = []
    reader = shapefile.Reader(str(path), encoding="cp949")
    for record in reader.iterShapeRecords():
        shape = record.shape
        if not _intersects(
            [shape.points[0][0], shape.points[0][1], *shape.points[0]],
            source_bounds,
        ):
            continue
        longitude, latitude = inverse_transformer.transform(*shape.points[0])
        positions.append(plane.to_local(latitude, longitude)[[0, 2]])
        heights.append(float(record.record["HEIGHT"]))
    return np.asarray(positions), np.asarray(heights), len(positions)


def _land_samples(scene, positions: np.ndarray) -> np.ndarray:
    mask, bounds = scene.water_mask, scene.water_mask_bounds
    x = np.rint(
        np.clip(
            (positions[:, 0] - bounds[0]) / (bounds[2] - bounds[0]),
            0.0,
            1.0,
        )
        * (mask.shape[1] - 1)
    ).astype(np.int32)
    z = np.rint(
        np.clip(
            (positions[:, 1] - bounds[1]) / (bounds[3] - bounds[1]),
            0.0,
            1.0,
        )
        * (mask.shape[0] - 1)
    ).astype(np.int32)
    return mask[z, x] < 128


def _deduplicate(
    positions: np.ndarray, heights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    half_metre_cells = np.rint(positions * 2.0).astype(np.int64)
    _, inverse = np.unique(half_metre_cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    averaged_positions = np.column_stack(
        (
            np.bincount(inverse, weights=positions[:, 0]) / counts,
            np.bincount(inverse, weights=positions[:, 1]) / counts,
        )
    )
    averaged_heights = np.bincount(inverse, weights=heights) / counts
    return averaged_positions, averaged_heights


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene", type=Path, default=Path("assets/yeouido_scene.npz")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument(
        "--water-level",
        type=Path,
        default=Path("assets/yeouido_2024-10-05_water_level.json"),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("assets/yeouido_terrain_2023_provenance.json"),
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--contour-spacing-m", type=float, default=15.0)
    args = parser.parse_args()

    pyproj, _ = _dependencies()
    scene = load_scene(args.scene)
    water = json.loads(args.water_level.read_text(encoding="utf-8"))
    water_datum_m = float(water["reference_surface_elevation_el_m"])
    output = args.output or args.scene

    with TemporaryDirectory(prefix="firework-seoul-terrain-") as directory:
        working = Path(directory)
        archive = args.source_archive or working / "seoul_contours.zip"
        checksum = (
            _download_archive(archive)
            if args.source_archive is None
            else _sha256(archive)
        )
        if checksum != SOURCE_ARCHIVE_SHA256:
            raise RuntimeError(
                f"unexpected source archive checksum sha256:{checksum}"
            )
        extracted = working / "source"
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
        contour_path = next(extracted.rglob("N3L_F001.shp"))
        spot_path = next(extracted.rglob("N3P_F002.shp"))

        inverse = pyproj.Transformer.from_crs(5174, 4326, always_xy=True)
        forward = pyproj.Transformer.from_crs(4326, 5174, always_xy=True)
        plane = LocalTangentPlane(
            scene.origin_latitude_deg, scene.origin_longitude_deg
        )
        source_bounds = _scene_projected_bounds(scene, forward, 150.0)
        contour_xy, contour_el, contour_features = _contour_samples(
            contour_path,
            source_bounds,
            scene.terrain_bounds,
            inverse,
            plane,
            args.contour_spacing_m,
            100.0,
        )
        spot_xy, spot_el, spot_features = _spot_samples(
            spot_path, source_bounds, inverse, plane
        )
        positions = np.concatenate((contour_xy, spot_xy))
        absolute_height = np.concatenate((contour_el, spot_el))
        land = _land_samples(scene, positions)
        positions, absolute_height = _deduplicate(
            positions[land], absolute_height[land]
        )
        relative_height = absolute_height - water_datum_m
        refined, supported_fraction = constrained_heightmap(
            scene.terrain_height_m,
            scene.terrain_bounds,
            positions,
            relative_height,
            scene.water_mask,
            (args.width, args.height),
        )
        raster_error_m = sample_heightmap_array(
            refined, scene.terrain_bounds, positions
        ) - relative_height
        result = replace(
            scene,
            terrain_height_m=refined,
            elevation_datum_m=water_datum_m,
        )
        save_scene(result, output)

    provenance = {
        "schema_version": 1,
        "source_id": "seoul-open-data-OA-22241-ngii-2023",
        "source_page": DATASET_PAGE,
        "download_url": DOWNLOAD_URL,
        "source_archive_sha256": checksum,
        "source_crs": "EPSG:5174",
        "source_scale": "1:5,000",
        "source_description": "2023 NGII-notified Seoul contours and spot heights",
        "license": "Korea Open Government License Type 1 (attribution)",
        "water_level_source": water["source"],
        "water_datum_el_m": water_datum_m,
        "contour_feature_count": contour_features,
        "spot_feature_count": spot_features,
        "constraint_count_after_filtering": int(len(positions)),
        "contour_resample_spacing_m": args.contour_spacing_m,
        "output_resolution": [args.width, args.height],
        "output_spacing_m": [
            float(
                (scene.terrain_bounds[2] - scene.terrain_bounds[0])
                / (args.width - 1)
            ),
            float(
                (scene.terrain_bounds[3] - scene.terrain_bounds[1])
                / (args.height - 1)
            ),
        ],
        "official_support_fraction": supported_fraction,
        "constraint_fit": {
            "rmse_m": float(np.sqrt(np.mean(raster_error_m**2))),
            "mean_absolute_error_m": float(np.mean(np.abs(raster_error_m))),
            "p95_absolute_error_m": float(
                np.percentile(np.abs(raster_error_m), 95)
            ),
            "median_error_m": float(np.median(raster_error_m)),
        },
        "derived_asset_sha256": _sha256(output),
        "uncertainty": (
            "The dense grid is piecewise-linear interpolation of mapped contours "
            "and spot heights, not a 1 m/5 m surveyed DEM. Narrow levees, stairs "
            "and kerb profiles still require survey or photogrammetry."
        ),
    }
    args.provenance_output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved {output}: {refined.shape[1]} x {refined.shape[0]}, "
        f"{len(positions):,} official constraints, "
        f"{supported_fraction:.1%} official interpolation support, "
        f"water datum {water_datum_m:.2f} EL.m"
    )


if __name__ == "__main__":
    main()
