"""Audit V1-9 NGII public controls, CRS axes, and bridge-use blocking."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from simulator.geodesy import LocalTangentPlane
from simulator.ngii_control_evidence import (
    DEFAULT_NGII_CONTROL_EVIDENCE_PATH,
    load_ngii_control_evidence,
)
from simulator.scene import load_scene


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
DEFAULT_SOURCE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "assets" / "yeouido_ngii_1000_source_manifest.json"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "ngii_public_controls_v1"
    / "ngii_public_controls_report.json"
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_report(
    evidence_path: Path = DEFAULT_NGII_CONTROL_EVIDENCE_PATH,
    scene_path: Path = DEFAULT_SCENE_PATH,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST_PATH,
) -> dict[str, object]:
    try:
        import pyproj
    except ImportError as error:
        raise RuntimeError("pyproj is required for the NGII control audit") from error

    evidence = load_ngii_control_evidence(evidence_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_link = source_manifest.get("public_control_evidence", {})
    evidence_hash = _hash(evidence_path)
    source_link_matches = bool(
        source_link.get("asset")
        == evidence_path.relative_to(REPOSITORY_ROOT).as_posix()
        and source_link.get("sha256") == evidence_hash
    )
    scene = load_scene(scene_path)
    transformer = pyproj.Transformer.from_crs(
        "EPSG:5186", "EPSG:4326", always_xy=True
    )
    crs = pyproj.CRS.from_epsg(5186)
    plane = LocalTangentPlane(
        scene.origin_latitude_deg, scene.origin_longitude_deg
    )
    bounds = scene.terrain_bounds
    transformed: list[dict[str, object]] = []
    correct_inside_count = 0
    swapped_inside_count = 0
    for control in evidence.controls:
        longitude, latitude = transformer.transform(
            control.easting_m, control.northing_m
        )
        swapped_longitude, swapped_latitude = transformer.transform(
            control.northing_m, control.easting_m
        )
        local = plane.to_local(latitude, longitude)
        swapped_local = plane.to_local(swapped_latitude, swapped_longitude)
        inside = bool(
            bounds[0] <= local[0] <= bounds[2]
            and bounds[1] <= local[2] <= bounds[3]
        )
        swapped_inside = bool(
            bounds[0] <= swapped_local[0] <= bounds[2]
            and bounds[1] <= swapped_local[2] <= bounds[3]
        )
        correct_inside_count += inside
        swapped_inside_count += swapped_inside
        transformed.append(
            {
                "control_id": control.control_id,
                "status": control.status,
                "longitude_deg": float(longitude),
                "latitude_deg": float(latitude),
                "local_east_m": float(local[0]),
                "local_south_m": float(local[2]),
                "unapplied_runtime_y_reference_m": (
                    control.orthometric_height_m - scene.elevation_datum_m
                ),
                "inside_scene_bounds": inside,
                "swapped_axis_inside_scene_bounds": swapped_inside,
                "bridge_landmark": control.bridge_landmark,
            }
        )

    native_axes = [
        {
            "name": axis.name,
            "abbreviation": axis.abbrev,
            "direction": axis.direction,
            "unit": axis.unit_name,
        }
        for axis in crs.axis_info
    ]
    passed = bool(
        evidence.catalogue_crs_reference_allowed
        and not evidence.digital_map_crs_verified
        and evidence.destroyed_control_count == len(evidence.controls)
        and evidence.active_field_control_count == 0
        and evidence.active_bridge_control_count == 0
        and not evidence.bridge_station_registration_allowed
        and not evidence.vertical_profile_allowed
        and correct_inside_count == len(evidence.controls)
        and swapped_inside_count == 0
        and source_link_matches
    )
    return {
        "schema_version": 1,
        "stage": "V1-9",
        "target_event_date": "2024-10-05",
        "evidence_asset": evidence_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "evidence_sha256": evidence_hash,
        "source_manifest_link": {
            "manifest": source_manifest_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "expected_asset": source_link.get("asset"),
            "expected_sha256": source_link.get("sha256"),
            "matches": source_link_matches,
        },
        "scene_asset": scene_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "scene_sha256": _hash(scene_path),
        "catalogue_product": evidence.document["catalogue_product"],
        "crs_audit": {
            "catalogue_control_crs": "EPSG:5186",
            "crs_name": crs.name,
            "native_axis_order": native_axes,
            "portal_mapping": "minx=easting, miny=northing",
            "runtime_policy": "always_xy=True",
            "correct_mapping_inside_scene_count": correct_inside_count,
            "swapped_mapping_inside_scene_count": swapped_inside_count,
            "control_count": len(evidence.controls),
            "digital_map_delivery_crs_verified": evidence.digital_map_crs_verified,
            "scope_warning": (
                "The control catalogue explicitly identifies EPSG:5186. The "
                "undelivered 2025 DXF only displays GRS80, so its projected CRS "
                "remains unverified until package metadata is inspected."
            ),
        },
        "controls": transformed,
        "qualification": {
            "destroyed_control_count": evidence.destroyed_control_count,
            "active_field_control_count": evidence.active_field_control_count,
            "active_bridge_control_count": evidence.active_bridge_control_count,
            "catalogue_crs_reference_allowed": (
                evidence.catalogue_crs_reference_allowed
            ),
            "field_control_use_allowed": evidence.active_field_control_count > 0,
            "bridge_station_registration_allowed": (
                evidence.bridge_station_registration_allowed
            ),
            "vertical_profile_allowed": evidence.vertical_profile_allowed,
            "blocking_reasons": list(evidence.reasons),
        },
        "runtime_impact": {
            "scene_vertices_modified": 0,
            "frame_path_changed": False,
            "expected_frame_time_delta_ms": 0.0,
        },
        "passed": passed,
        "next_evidence_gate": (
            "Acquire an authenticated 2024-or-earlier 1:1,000 delivery, verify "
            "its embedded projected CRS and licence, and obtain at least three "
            "active controls tied to Seogang Bridge landmarks before station "
            "registration."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=DEFAULT_NGII_CONTROL_EVIDENCE_PATH
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST_PATH
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()
    report = build_report(
        arguments.evidence.resolve(),
        arguments.scene.resolve(),
        arguments.source_manifest.resolve(),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
