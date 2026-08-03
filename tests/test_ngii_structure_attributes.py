from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


STRUCTURES = Path("assets/yeouido_ngii_structures.json")
ATTRIBUTES = Path("assets/yeouido_ngii_structure_attributes.json")
RECEIPT = Path("assets/yeouido_ngii_delivery_receipt.json")


def test_attribute_audit_covers_every_normalized_feature() -> None:
    structures = json.loads(STRUCTURES.read_text(encoding="utf-8"))
    attributes = json.loads(ATTRIBUTES.read_text(encoding="utf-8"))

    assert attributes["structure_asset_sha256"] == sha256(
        STRUCTURES.read_bytes()
    ).hexdigest()
    assert attributes["delivery_receipt_sha256"] == sha256(
        RECEIPT.read_bytes()
    ).hexdigest()
    assert {item["feature_id"] for item in attributes["features"]} == {
        feature["feature_id"] for feature in structures["features"]
    }
    assert attributes["summary"] == {
        "feature_count": 71,
        "upper_edge_count": 62,
        "lower_edge_count": 9,
        "unknown_edge_role_count": 0,
        "relative_height_count": 65,
        "positive_relative_height_count": 59,
        "relative_height_range_m": [0.3, 8.0],
    }


def test_relative_height_is_not_promoted_to_absolute_elevation() -> None:
    attributes = json.loads(ATTRIBUTES.read_text(encoding="utf-8"))

    assert not attributes["application"]["mesh_merge_allowed"]
    assert not attributes["application"]["absolute_elevation_available"]
    assert all(
        not feature["height_usable_for_absolute_elevation"]
        for feature in attributes["features"]
    )
    assert all(
        point[1] is None
        for feature in json.loads(STRUCTURES.read_text(encoding="utf-8"))["features"]
        for point in feature["points_eus_m"]
    )
