from __future__ import annotations

from datetime import date
import json
import io
import math
from pathlib import Path

import pytest

from tools.import_ngii_structures import (
    DEFAULT_LAYER_KINDS,
    build_normalized_asset,
    iter_dxf_sources,
    parse_ascii_dxf,
    parse_shp_polylines,
    resolve_layer_kind,
    validate_delivery_evidence,
    validate_source_year,
)
from simulator.ngii_delivery import load_ngii_delivery_receipt


def _dxf(*entity_lines: str) -> str:
    lines = ["0", "SECTION", "2", "ENTITIES"]
    lines.extend(entity_lines)
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"


def test_parser_reads_supported_linear_entities_and_elevation() -> None:
    text = _dxf(
        "0", "LINE", "8", "F0040000", "10", "100", "20", "200",
        "30", "8.5", "11", "110", "21", "205", "31", "9.0",
        "0", "LWPOLYLINE", "8", "C0050000", "38", "4.2", "90", "3",
        "10", "1", "20", "2", "10", "3", "20", "4", "10", "5", "20", "8",
        "0", "POLYLINE", "8", "F0030000", "30", "6.0",
        "0", "VERTEX", "10", "7", "20", "8",
        "0", "VERTEX", "10", "9", "20", "10", "30", "6.5",
        "0", "SEQEND",
        "0", "LINE", "8", "B0010000", "10", "0", "20", "0",
        "11", "1", "21", "1",
    )

    entities = parse_ascii_dxf(text)

    assert [entity.entity_type for entity in entities] == [
        "LINE", "LWPOLYLINE", "POLYLINE", "LINE"
    ]
    assert entities[0].points == ((100.0, 200.0, 8.5), (110.0, 205.0, 9.0))
    assert entities[1].points == (
        (1.0, 2.0, 4.2), (3.0, 4.0, 4.2), (5.0, 8.0, 4.2)
    )
    assert entities[2].points == ((7.0, 8.0, 6.0), (9.0, 10.0, 6.5))


def test_parser_rejects_malformed_group_pairs_and_vertices() -> None:
    with pytest.raises(ValueError, match="no value"):
        parse_ascii_dxf("0\nSECTION\n2\n")
    with pytest.raises(ValueError, match="matching x"):
        parse_ascii_dxf(_dxf("0", "LWPOLYLINE", "8", "C0050000", "20", "2"))


def test_shp_parser_splits_multipart_structure_lines() -> None:
    shapefile = pytest.importorskip("shapefile")
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYLINE)
    writer.field("UFID", "C")
    writer.line([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    writer.record("test")
    writer.close()

    entities = parse_shp_polylines(shp.getvalue(), "N1L_F0040000.shp")

    assert len(entities) == 2
    assert {entity.layer for entity in entities} == {"F0040000"}
    assert entities[0].points == ((1.0, 2.0, None), (3.0, 4.0, None))
    assert entities[1].points == ((5.0, 6.0, None), (7.0, 8.0, None))


def test_lwpolyline_preserves_bulge_arcs_and_closed_flag() -> None:
    arc = parse_ascii_dxf(
        _dxf(
            "0", "LWPOLYLINE", "8", "F0040000", "90", "2",
            "10", "0", "20", "0", "42", str(math.tan(math.pi / 8.0)),
            "10", "2", "20", "0",
        )
    )[0]
    closed = parse_ascii_dxf(
        _dxf(
            "0", "LWPOLYLINE", "8", "C0050000", "70", "1", "90", "3",
            "10", "0", "20", "0", "10", "2", "20", "0",
            "10", "2", "20", "2",
        )
    )[0]

    assert len(arc.points) == 19
    arc_length = sum(
        math.dist(first[:2], second[:2])
        for first, second in zip(arc.points, arc.points[1:])
    )
    assert arc_length == pytest.approx(math.pi / math.sqrt(2.0), rel=0.002)
    assert closed.points[0] == closed.points[-1]


def test_layer_filter_is_exact_or_delimited() -> None:
    assert resolve_layer_kind("f0040000", DEFAULT_LAYER_KINDS) == "retaining_wall"
    assert resolve_layer_kind("F0040000_DETAIL", DEFAULT_LAYER_KINDS) == "retaining_wall"
    assert resolve_layer_kind("F00400001", DEFAULT_LAYER_KINDS) is None
    assert resolve_layer_kind("B0010000", DEFAULT_LAYER_KINDS) is None


def test_post_event_source_requires_explicit_override() -> None:
    event = date(2024, 10, 5)
    with pytest.raises(ValueError, match="later than event"):
        validate_source_year(2025, event)
    assert (
        validate_source_year(2025, event, allow_post_event_source=True)
        == "official_post_event"
    )
    assert validate_source_year(2024, event) == "official_same_year_date_unverified"
    assert validate_source_year(2023, event) == "official_pre_event"


def test_normalized_asset_keeps_unknown_height_null() -> None:
    pyproj = pytest.importorskip("pyproj")
    from simulator.scene import load_scene

    scene_path = Path("assets/yeouido_scene.npz")
    scene = load_scene(scene_path)
    projected = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:5186", always_xy=True
    )
    east, north = projected.transform(
        scene.origin_longitude_deg, scene.origin_latitude_deg
    )
    text = _dxf(
        "0", "LINE", "8", "F0040000",
        "10", str(east), "20", str(north),
        "11", str(east + 10.0), "21", str(north),
    ).encode("ascii")

    asset = build_normalized_asset(
        [("synthetic.dxf", text)],
        source_crs="EPSG:5186",
        source_year=2024,
        scene_path=scene_path,
    )

    assert asset["schema_version"] == 2
    assert asset["summary"]["feature_count"] == 1
    feature = asset["features"][0]
    assert feature["kind"] == "retaining_wall"
    assert len(feature["feature_id"]) == 24
    assert feature["has_source_elevation"] is False
    assert all(point[1] is None for point in feature["points_eus_m"])
    assert feature["length_m"] == pytest.approx(10.0, abs=0.02)


def test_directory_and_zip_discovery_is_case_insensitive(tmp_path: Path) -> None:
    direct = tmp_path / "A.DXF"
    direct.write_text(_dxf(), encoding="ascii")
    archive = tmp_path / "sheets.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("nested/B.DxF", _dxf())

    sources = list(iter_dxf_sources([tmp_path]))

    assert len(sources) == 2
    assert sources[0][0].endswith("A.DXF")
    assert sources[1][0] == "sheets.zip:nested/B.DxF"


def test_source_manifest_records_explicit_post_event_adoption() -> None:
    path = Path("assets/yeouido_ngii_1000_source_manifest.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["target_event_date"] == "2024-10-05"
    assert {sheet["sheet_id"] for sheet in data["event_area_sheets"]} == {
        "376082447", "376082448", "376082457", "376082458"
    }
    assert {sheet["production_year_shown"] for sheet in data["event_area_sheets"]} == {2025}
    assert data["product"]["projected_crs"] == "EPSG:5186"
    assert data["ingestion_status"] == (
        "authenticated_2025_delivery_normalized_as_post_event_planimetry"
    )
    assert data["normalized_structure_asset"]["feature_count"] == 71
    assert data["normalized_structure_asset"]["features_with_source_elevation"] == 0


def test_real_import_accepts_the_authorized_post_event_delivery_evidence() -> None:
    receipt = load_ngii_delivery_receipt()

    validate_delivery_evidence(
        receipt,
        source_crs="EPSG:5186",
        source_year=2025,
    )
    assert receipt.post_event_authorized
    assert not receipt.historical_identity_verified
