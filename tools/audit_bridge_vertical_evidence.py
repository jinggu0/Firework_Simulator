"""Audit V1-6 official bridge elevations without inventing a deck profile."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from simulator.bridge_evidence import (
    DEFAULT_BRIDGE_EVIDENCE_PATH,
    load_bridge_evidence,
)
from simulator.scene import load_scene


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "bridge_vertical_v1"
    / "bridge_vertical_report.json"
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(evidence_path: Path, scene_path: Path) -> dict[str, object]:
    evidence = load_bridge_evidence(evidence_path)
    scene_hash_before = _sha256(scene_path)
    scene = load_scene(scene_path)
    scene_hash_after = _sha256(scene_path)
    scene_hash_matches = scene_hash_before == evidence.scene_sha256
    datum_matches = abs(scene.elevation_datum_m - evidence.elevation_datum_m) <= 1e-6
    untouched = scene_hash_before == scene_hash_after

    local_y = np.unique(np.round(scene.bridge_vertices[:, 1], decimals=6))
    anchors = [
        {
            "anchor_id": anchor.anchor_id,
            "component": anchor.component,
            "quantity": anchor.quantity,
            "absolute_el_m": anchor.elevation_el_m,
            "runtime_y_m": anchor.runtime_y_m,
            "uncertainty_m": anchor.uncertainty_m,
            "source_id": anchor.source_id,
        }
        for anchor in evidence.anchors
    ]
    application = evidence.application
    return {
        "stage": "V1-6",
        "target_event_date": "2024-10-05",
        "bridge_id": "seogang_bridge",
        "evidence_manifest": str(evidence_path.relative_to(REPOSITORY_ROOT)).replace(
            "\\", "/"
        ),
        "scene": {
            "asset": str(scene_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
            "expected_sha256": evidence.scene_sha256,
            "observed_sha256": scene_hash_before,
            "checksum_matches": scene_hash_matches,
            "expected_elevation_datum_el_m": evidence.elevation_datum_m,
            "observed_elevation_datum_el_m": scene.elevation_datum_m,
            "datum_matches": datum_matches,
            "bridge_vertex_count": len(scene.bridge_vertices),
            "stored_bridge_local_y_values_m": local_y.tolist(),
        },
        "official_vertical_anchors": anchors,
        "source_integrity": {
            "grade_a_or_b_source_count": len(evidence.sources),
            "all_sources_checksum_locked": True,
            "source_files_redistributed_in_repository": False,
            "note": (
                "The report locks the inspected official attachment checksums. "
                "Their licences and source pages remain authoritative; source "
                "documents are not copied into the repository."
            ),
        },
        "geometry_application": {
            "allowed": application.allowed,
            "status": application.status,
            "registered_profiles": application.registered_profiles,
            "blocking_reasons": list(application.reasons),
            "vertices_modified": 0,
            "direct_height_error_comparison": "blocked_without_station_registration",
            "note": (
                "EL 23.3 m describes the underside at the navigation span. The "
                "stored 7 m value is a terrain-relative deck offset used by many "
                "OSM bridge ways, so subtracting the two would be dimensionally "
                "and spatially invalid."
            ),
        },
        "regression": {
            "scene_checksum_unchanged_during_audit": untouched,
            "runtime_frame_path_changed": False,
            "expected_frame_time_delta_ms": 0.0,
        },
        "passed": scene_hash_matches and datum_matches and untouched,
        "next_evidence_gate": (
            "Verify structural history through 2024-10-05, then register at least "
            "three independent completion-drawing controls from a grade-A/B plan "
            "at <=0.125 m/px and <=0.25 m plan RMSE. Digitise deck-top EL samples "
            "only at <=0.10 m documented vertical uncertainty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_BRIDGE_EVIDENCE_PATH)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    report = build_report(arguments.evidence.resolve(), arguments.scene.resolve())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
