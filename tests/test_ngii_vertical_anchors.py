from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


REPORT = Path(
    "docs/validation/ngii_vertical_anchor_v1/ngii_vertical_anchor_report.json"
)
STRUCTURES = Path("assets/yeouido_ngii_structures.json")
ATTRIBUTES = Path("assets/yeouido_ngii_structure_attributes.json")


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_vertical_anchor_audit_is_bound_to_structure_evidence() -> None:
    report = _report()

    assert report["structure_asset_sha256"] == sha256(
        STRUCTURES.read_bytes()
    ).hexdigest()
    assert report["attribute_asset_sha256"] == sha256(
        ATTRIBUTES.read_bytes()
    ).hexdigest()
    assert report["vertical_constraints"]["contour_feature_count"] == 35
    assert report["vertical_constraints"]["spot_height_count"] == 12
    assert report["vertical_constraints"][
        "structure_point_convex_hull_support_fraction"
    ] == 0.8802660753880266


def test_failed_vertical_accuracy_gate_keeps_scene_unchanged() -> None:
    report = _report()
    validation = report["contour_to_spot_cross_validation"]

    assert validation["supported_spot_count"] == 5
    assert validation["rmse_m"] > 4.0
    assert report["summary"] == {
        "lower_edge_count": 9,
        "strong_plan_pair_count": 4,
        "vertical_anchor_pass_count": 0,
        "scene_vertices_modified": 0,
    }
    assert not report["gate"]["passed"]
    assert not report["gate"]["mesh_merge_allowed"]
