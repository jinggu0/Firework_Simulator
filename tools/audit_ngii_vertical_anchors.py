"""Audit whether NGII contours/spot heights can anchor structure profiles."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

import numpy as np

from simulator.geodesy import LocalTangentPlane
from simulator.ngii_delivery import (
    DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
    load_ngii_delivery_receipt,
    validate_delivery_packages,
)
from simulator.scene import load_scene


DEFAULT_STRUCTURES = Path("assets/yeouido_ngii_structures.json")
DEFAULT_ATTRIBUTES = Path("assets/yeouido_ngii_structure_attributes.json")
DEFAULT_SCENE = Path("assets/yeouido_scene.npz")
DEFAULT_TERRAIN_PROVENANCE = Path("assets/yeouido_terrain_2023_provenance.json")
DEFAULT_OUTPUT = Path(
    "docs/validation/ngii_vertical_anchor_v1/ngii_vertical_anchor_report.json"
)
FIELD_CONTOUR_HEIGHT = "\ub4f1\uace0\uc218\uce58"
FIELD_SPOT_HEIGHT = "\uc218\uce58"


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _path_digest(path: Path) -> str:
    return _digest(path.read_bytes())


def _package_paths(paths: Iterable[Path]) -> list[Path]:
    output: list[Path] = []
    for supplied in paths:
        if supplied.is_dir():
            output.extend(
                path
                for path in supplied.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".zip", ".dxf", ".ngi", ".xml"}
            )
        elif supplied.is_file():
            output.append(supplied)
    return output


def _thin_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    targets = np.concatenate(
        (np.arange(0.0, cumulative[-1], spacing_m), [cumulative[-1]])
    )
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)]
    )


def _resample_feature(feature: dict[str, Any], spacing_m: float = 0.5) -> np.ndarray:
    points = np.asarray(
        [[point[0], point[2]] for point in feature["points_eus_m"]],
        dtype=np.float64,
    )
    return _thin_polyline(points, spacing_m)


def _pair_candidates(
    structures: dict[str, Any], attributes: dict[str, Any]
) -> list[dict[str, Any]]:
    from scipy.spatial import cKDTree

    features = {feature["feature_id"]: feature for feature in structures["features"]}
    output: list[dict[str, Any]] = []
    for kind in ("embankment", "cut_or_fill", "retaining_wall"):
        upper = [
            item for item in attributes["features"]
            if item["kind"] == kind and item["edge_role"] == "upper"
        ]
        lower = [
            item for item in attributes["features"]
            if item["kind"] == kind and item["edge_role"] == "lower"
        ]
        for lower_item in lower:
            lower_points = _resample_feature(features[lower_item["feature_id"]])
            candidates: list[tuple[float, float, float, dict[str, Any]]] = []
            for upper_item in upper:
                upper_points = _resample_feature(features[upper_item["feature_id"]])
                distances = np.concatenate(
                    (
                        cKDTree(upper_points).query(lower_points)[0],
                        cKDTree(lower_points).query(upper_points)[0],
                    )
                )
                length_a = float(
                    lower_item["declared_length_m"]
                    or features[lower_item["feature_id"]]["length_m"]
                )
                length_b = float(
                    upper_item["declared_length_m"]
                    or features[upper_item["feature_id"]]["length_m"]
                )
                ratio = (
                    max(length_a, length_b) / min(length_a, length_b)
                    if min(length_a, length_b) > 0.0
                    else float("inf")
                )
                candidates.append(
                    (
                        float(np.median(distances)),
                        float(np.percentile(distances, 95)),
                        ratio,
                        upper_item,
                    )
                )
            median, p95, ratio, best = min(candidates, key=lambda item: item[0])
            strong = median <= 2.5 and p95 <= 3.0 and ratio <= 2.0
            output.append(
                {
                    "kind": kind,
                    "lower_feature_id": lower_item["feature_id"],
                    "upper_feature_id": best["feature_id"],
                    "lower_ufid": lower_item["ufid"],
                    "upper_ufid": best["ufid"],
                    "median_bidirectional_distance_m": median,
                    "p95_bidirectional_distance_m": p95,
                    "declared_length_ratio": None if not np.isfinite(ratio) else ratio,
                    "upper_declared_relative_height_m": best[
                        "declared_relative_height_m"
                    ],
                    "plan_pair_status": (
                        "strong_candidate" if strong else "partial_or_ambiguous"
                    ),
                    "vertical_anchor_status": "blocked",
                }
            )
    return output


def build_report(
    input_paths: Iterable[Path],
    *,
    structures_path: Path = DEFAULT_STRUCTURES,
    attributes_path: Path = DEFAULT_ATTRIBUTES,
    scene_path: Path = DEFAULT_SCENE,
    terrain_provenance_path: Path = DEFAULT_TERRAIN_PROVENANCE,
    receipt_path: Path = DEFAULT_NGII_DELIVERY_RECEIPT_PATH,
) -> dict[str, Any]:
    try:
        import pyproj
        import shapefile
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import Delaunay
    except ImportError as error:
        raise RuntimeError("install requirements-terrain.txt for vertical audit") from error

    supplied = list(input_paths)
    receipt = load_ngii_delivery_receipt(receipt_path)
    validate_delivery_packages(receipt, supplied)
    structures = json.loads(structures_path.read_text(encoding="utf-8"))
    attributes = json.loads(attributes_path.read_text(encoding="utf-8"))
    terrain_provenance = json.loads(
        terrain_provenance_path.read_text(encoding="utf-8")
    )
    scene = load_scene(scene_path)
    transformer = pyproj.Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    plane = LocalTangentPlane(scene.origin_latitude_deg, scene.origin_longitude_deg)
    contour_positions: list[np.ndarray] = []
    contour_heights: list[np.ndarray] = []
    spot_positions: list[np.ndarray] = []
    spot_heights: list[float] = []
    source_members: list[dict[str, Any]] = []
    contour_count = 0
    spot_count = 0

    for package in sorted(_package_paths(supplied), key=lambda path: path.name):
        if package.suffix.casefold() != ".zip":
            continue
        with zipfile.ZipFile(package) as archive:
            names = set(archive.namelist())
            for code in ("F0010000", "F0020000"):
                shp_member = next(
                    (name for name in names if name.upper().endswith(f"_{code}.SHP")),
                    None,
                )
                if shp_member is None:
                    continue
                base = shp_member[:-4]
                companions = [base + ext for ext in (".shp", ".shx", ".dbf", ".prj")]
                raw = {name: archive.read(name) for name in companions}
                source_members.append(
                    {
                        "package": package.name,
                        "layer": code,
                        "members": [
                            {
                                "member_path": name,
                                "sha256": _digest(raw[name]),
                                "bytes": len(raw[name]),
                            }
                            for name in companions
                        ],
                    }
                )
                reader = shapefile.Reader(
                    shp=io.BytesIO(raw[base + ".shp"]),
                    shx=io.BytesIO(raw[base + ".shx"]),
                    dbf=io.BytesIO(raw[base + ".dbf"]),
                    encoding="cp949",
                )
                fields = [field[0] for field in reader.fields[1:]]
                for item in reader.iterShapeRecords():
                    record = dict(zip(fields, item.record))
                    projected = np.asarray(item.shape.points, dtype=np.float64)
                    longitude, latitude = transformer.transform(
                        projected[:, 0], projected[:, 1]
                    )
                    local = plane.to_local_array(
                        np.asarray(latitude), np.asarray(longitude)
                    )[:, [0, 2]]
                    if code == "F0010000":
                        elevation = float(record[FIELD_CONTOUR_HEIGHT])
                        sampled = _thin_polyline(local, 2.0)
                        contour_positions.append(sampled)
                        contour_heights.append(np.full(len(sampled), elevation))
                        contour_count += 1
                    else:
                        spot_positions.append(local[0])
                        spot_heights.append(float(record[FIELD_SPOT_HEIGHT]))
                        spot_count += 1

    contours = np.concatenate(contour_positions)
    contour_values = np.concatenate(contour_heights)
    cells = np.rint(contours * 10.0).astype(np.int64)
    _, unique = np.unique(cells, axis=0, return_index=True)
    contours, contour_values = contours[unique], contour_values[unique]
    spots = np.asarray(spot_positions, dtype=np.float64)
    spot_values = np.asarray(spot_heights, dtype=np.float64)
    contour_interpolator = LinearNDInterpolator(
        Delaunay(contours), contour_values, fill_value=np.nan
    )
    predicted_spots = np.asarray(contour_interpolator(spots), dtype=np.float64)
    supported_spots = np.isfinite(predicted_spots)
    residuals = predicted_spots[supported_spots] - spot_values[supported_spots]
    combined = np.concatenate((contours, spots))
    combined_values = np.concatenate((contour_values, spot_values))
    combined_triangulation = Delaunay(combined)
    structure_points = np.asarray(
        [
            [point[0], point[2]]
            for feature in structures["features"]
            for point in feature["points_eus_m"]
        ],
        dtype=np.float64,
    )
    supported_structure = combined_triangulation.find_simplex(structure_points) >= 0
    pair_candidates = _pair_candidates(structures, attributes)
    strong_pairs = sum(
        item["plan_pair_status"] == "strong_candidate" for item in pair_candidates
    )
    constraint_fit = terrain_provenance["constraint_fit"]
    return {
        "schema_version": 1,
        "stage": "V1-11b",
        "target_event_date": structures["target_event_date"],
        "source_year": structures["source_year"],
        "temporal_relation": structures["temporal_relation"],
        "delivery_receipt_sha256": _path_digest(receipt_path),
        "structure_asset_sha256": _path_digest(structures_path),
        "attribute_asset_sha256": _path_digest(attributes_path),
        "source_crs": "EPSG:5186",
        "vertical_datum": "provider elevation fields; absolute datum label not explicit in member DBF",
        "source_members": source_members,
        "vertical_constraints": {
            "contour_feature_count": contour_count,
            "spot_height_count": spot_count,
            "resampled_contour_constraint_count": len(contours),
            "absolute_elevation_range_m": [
                float(min(np.min(contour_values), np.min(spot_values))),
                float(max(np.max(contour_values), np.max(spot_values))),
            ],
            "structure_point_convex_hull_support_fraction": float(
                np.mean(supported_structure)
            ),
        },
        "contour_to_spot_cross_validation": {
            "spot_count": len(spots),
            "supported_spot_count": int(np.count_nonzero(supported_spots)),
            "rmse_m": float(np.sqrt(np.mean(residuals**2))),
            "mean_absolute_error_m": float(np.mean(np.abs(residuals))),
            "p95_absolute_error_m": float(np.percentile(np.abs(residuals), 95)),
        },
        "existing_terrain_limit": {
            "source_scale": terrain_provenance["source_scale"],
            "output_spacing_m": terrain_provenance["output_spacing_m"],
            "constraint_rmse_m": constraint_fit["rmse_m"],
            "constraint_p95_absolute_error_m": constraint_fit[
                "p95_absolute_error_m"
            ],
        },
        "plan_pair_candidates": pair_candidates,
        "summary": {
            "lower_edge_count": len(pair_candidates),
            "strong_plan_pair_count": strong_pairs,
            "vertical_anchor_pass_count": 0,
            "scene_vertices_modified": 0,
        },
        "gate": {
            "passed": False,
            "mesh_merge_allowed": False,
            "reasons": [
                "only a subset of spot heights lies inside the contour convex hull",
                "contour-to-spot cross-validation error exceeds sub-metre structure needs",
                "the existing 1:5,000 terrain fit error exceeds many 0.3-8.0 m structure heights",
                "the F001/F002 member metadata does not explicitly name the absolute vertical datum",
                "2025 elevation constraints are not verified as unchanged on 2024-10-05",
            ],
            "next_source_requirement": (
                "Acquire provider-documented high-resolution DEM or surveyed elevations "
                "with explicit vertical datum and uncertainty. Require <=0.25 m plan "
                "registration RMSE and <=0.10 m vertical uncertainty before runtime merge."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("--structures", type=Path, default=DEFAULT_STRUCTURES)
    parser.add_argument("--attributes", type=Path, default=DEFAULT_ATTRIBUTES)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--terrain-provenance", type=Path, default=DEFAULT_TERRAIN_PROVENANCE
    )
    parser.add_argument(
        "--delivery-receipt", type=Path, default=DEFAULT_NGII_DELIVERY_RECEIPT_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    try:
        report = build_report(
            arguments.input,
            structures_path=arguments.structures,
            attributes_path=arguments.attributes,
            scene_path=arguments.scene,
            terrain_provenance_path=arguments.terrain_provenance,
            receipt_path=arguments.delivery_receipt,
        )
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = report["summary"]
    validation = report["contour_to_spot_cross_validation"]
    print(
        f"wrote {arguments.output}: {summary['strong_plan_pair_count']} strong "
        f"plan pairs, 0 vertical anchors; contour/spot RMSE "
        f"{validation['rmse_m']:.3f} m"
    )


if __name__ == "__main__":
    main()
