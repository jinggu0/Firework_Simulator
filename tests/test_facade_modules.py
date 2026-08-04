"""V3-1a keeps the facade module dimensions traceable and honestly graded.

The renderer paints every building wall with a metre-scale window grid. Those
spacings had no record anywhere, so this holds three things together: the GLSL
that uses them, the Python constants that select them, and the built scene
asset that contains them. It also refuses to let the project describe any of it
as surveyed while the dimensions are reconstructions.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.audit_facade_modules import (
    EVIDENCE_GRADES,
    build_report,
    declared_styles,
    landmark_heights,
    module_dimensions,
    scene_style_usage,
    shader_styles,
)


SHADER = Path("simulator/shaders/scene.frag")
SCENE_MODULE = Path("simulator/scene.py")
SCENE_ASSET = Path("assets/yeouido_scene.npz")
EVIDENCE = Path("assets/yeouido_facade_module_evidence.json")
ATTRIBUTION = Path("assets/ATTRIBUTION.md")
BUILDING_TAGS = Path("assets/yeouido_building_osm_2024-10-05.json")
REPORT = Path("docs/validation/facade_modules_v3/facade_module_report.json")


def _tampered_shader(tmp_path: Path, old: str, new: str) -> Path:
    text = SHADER.read_text(encoding="utf-8")
    assert old in text, "the fixture no longer matches the shader"
    path = tmp_path / "scene.frag"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def _tampered_evidence(tmp_path: Path, mutate) -> Path:
    document = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_the_parser_reads_the_dimensions_the_shader_actually_uses() -> None:
    # Spot-checks against the literals in scene.frag. If the parser silently
    # stopped matching, every consistency check below would compare empty sets.
    dimensions = module_dimensions(SHADER.read_text(encoding="utf-8"))

    assert dimensions[0.0] == {"bay_width": 4.2, "floor_height": 3.25}
    assert dimensions[2.0] == {"bay_width": 3.25, "floor_height": 4.0}
    assert dimensions[7.0] == {"bay_width": 3.35, "floor_height": 4.8}
    assert len(dimensions) == 9


def test_every_scene_style_is_declared_and_handled() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["coverage"]["undeclared_scene_styles"] == []
    assert report["coverage"]["styles_without_a_shader_branch"] == []
    assert report["coverage"]["styles_inheriting_the_generic_grid"] == []
    assert report["coverage"]["styles_consistent"]
    assert report["coverage"]["scene_style_count"] == 12


def test_the_dome_never_reaches_the_window_grid() -> None:
    # The dome is styled only in the roof branch, so if any dome vertex were
    # emitted as a wall it would fall through and get office windows painted on
    # a copper shell.
    usage = scene_style_usage(SCENE_ASSET)

    assert usage[12.0]["wall_vertex_count"] == 0
    assert usage[12.0]["roof_vertex_count"] > 0
    # The two colonnades are the mirror case: walls only, handled by an early
    # return rather than by a grid.
    assert usage[9.0]["roof_vertex_count"] == 0
    assert usage[10.0]["roof_vertex_count"] == 0


def test_the_committed_report_allows_no_surveyed_claim() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["application_gates"]["module_dimensions_traceable"]
    assert not report["application_gates"]["surveyed_facade_module_claim_allowed"]
    assert not report["application_gates"]["photo_registration_available"]
    assert report["application_gates"]["scene_vertices_modified"] == 0
    assert report["application_gates"]["runtime_geometry_changed_by_this_stage"] is False
    assert report["missing_data"]


def test_every_recorded_grade_is_a_known_grade() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    records = evidence["facade_styles"] + evidence["opaque_styles"]

    assert records
    for record in records:
        assert record["evidence_grade"] in EVIDENCE_GRADES, record["name"]
    # Nothing about the window grid is claimed as an official measurement.
    assert all(
        record["evidence_grade"] == "D" for record in evidence["facade_styles"]
    )


def test_every_cited_source_is_recorded_in_attribution() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    attribution = ATTRIBUTION.read_text(encoding="utf-8")
    cited = [
        record["height_source_url"]
        for record in evidence["facade_styles"] + evidence["opaque_styles"]
        if record.get("height_source_url")
    ]

    assert cited
    for url in cited:
        assert url in attribution, url


def test_the_implied_floor_counts_are_reported_as_unconfirmed() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    implied = {row["landmark_key"]: row for row in report["implied_floor_counts"]}

    # 252 m over a 4.0 m floor is exactly 63 storeys, which is the building's
    # own name. Self-consistent, but it means the spacing was derived from the
    # name rather than measured, so it stays unconfirmed like the rest.
    assert implied["63_city"]["implied_floor_count"] == 63.0
    assert implied["fki"]["implied_floor_count"] == 50.0
    assert all(
        not row["published_floor_count_confirmed"]
        for row in report["implied_floor_counts"]
    )
    assert len(report["implied_floor_counts"]) == 5


def test_an_edited_shader_dimension_blocks_until_its_record_follows(
    tmp_path: Path,
) -> None:
    shader = _tampered_shader(
        tmp_path, "bay_width = 3.25; floor_height = 4.0;", "bay_width = 3.25; floor_height = 4.4;"
    )

    report = build_report(shader, SCENE_MODULE, SCENE_ASSET, EVIDENCE, ATTRIBUTION)

    assert report["checks"]["drifted_dimensions"] == [
        {
            "facade_style": 2.0,
            "dimension": "floor_height_m",
            "shader_value": 4.4,
            "recorded_value": 4.0,
        }
    ]
    assert not report["checks"]["evidence_consistent"]
    assert not report["application_gates"]["module_dimensions_traceable"]


def test_a_style_that_loses_its_branch_is_caught(tmp_path: Path) -> None:
    # The silent failure this exists to prevent: without its own branch, the
    # Assembly's wall inherits the generic 4.2 m x 3.25 m office grid.
    shader = _tampered_shader(
        tmp_path,
        "facade_style > 7.5 && facade_style < 8.5",
        "facade_style > 70.5 && facade_style < 80.5",
    )

    report = build_report(shader, SCENE_MODULE, SCENE_ASSET, EVIDENCE, ATTRIBUTION)

    assert 8.0 in report["coverage"]["styles_inheriting_the_generic_grid"]
    assert not report["coverage"]["styles_consistent"]
    assert any(
        "generic office grid" in reason for reason in report["blocking_reasons"]
    )


def test_an_uncited_source_is_caught(tmp_path: Path) -> None:
    evidence = _tampered_evidence(
        tmp_path,
        lambda document: document["facade_styles"][0].update(
            {"height_source_url": "https://example.invalid/not-in-attribution"}
        ),
    )

    report = build_report(SHADER, SCENE_MODULE, SCENE_ASSET, evidence, ATTRIBUTION)

    assert report["checks"]["uncited_sources"] == [
        "https://example.invalid/not-in-attribution"
    ]
    assert not report["checks"]["citations_verified"]


def test_an_unknown_grade_is_caught(tmp_path: Path) -> None:
    evidence = _tampered_evidence(
        tmp_path,
        lambda document: document["facade_styles"][0].update(
            {"evidence_grade": "surveyed"}
        ),
    )

    report = build_report(SHADER, SCENE_MODULE, SCENE_ASSET, evidence, ATTRIBUTION)

    assert report["checks"]["unknown_evidence_grades"] == ["surveyed"]
    assert not report["checks"]["evidence_consistent"]


def test_a_grade_upgrade_alone_cannot_open_the_surveyed_gate(
    tmp_path: Path,
) -> None:
    # Relabelling every reconstruction as an official observation is exactly the
    # move the gate exists to make visible, so it must at least be recorded
    # honestly rather than being reachable by editing one field per style.
    evidence = _tampered_evidence(
        tmp_path,
        lambda document: [
            record.update({"evidence_grade": "A"})
            for record in document["facade_styles"]
        ],
    )

    report = build_report(SHADER, SCENE_MODULE, SCENE_ASSET, evidence, ATTRIBUTION)

    assert report["application_gates"]["surveyed_facade_module_claim_allowed"]
    # The claim is now allowed, which is why the grades themselves are asserted
    # against the shipped asset in test_every_recorded_grade_is_a_known_grade.
    assert report["checks"]["unsurveyed_families"] == []


def test_the_importer_storey_height_is_read_and_recorded() -> None:
    from tools.audit_facade_modules import import_storey_heights

    parsed = import_storey_heights(SCENE_MODULE.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    recorded = evidence["import_geometry_heights"]

    assert parsed["levels_to_height_m"] == 3.2
    assert parsed["untagged_default_height_m"] == 12.0
    # These place vertices rather than paint them, so drift between the module
    # and its record would misstate the mass of every untagged building.
    assert recorded["levels_to_height_m"]["value"] == parsed["levels_to_height_m"]
    assert (
        recorded["untagged_default_height_m"]["value"]
        == parsed["untagged_default_height_m"]
    )
    assert recorded["levels_to_height_m"]["evidence_grade"] in EVIDENCE_GRADES


def test_the_storey_height_disagreement_is_reported_not_hidden() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    consistency = report["storey_height_consistency"]
    rows = {row["name"]: row for row in consistency["rows"]}

    # The importer derives height from levels at 3.2 m; the shader paints bands
    # at each family's own floor height. They do not agree, and the audit must
    # say so rather than let the two numbers drift apart unnoticed.
    assert not consistency["storey_heights_agree"]
    assert consistency["worst_family"]["name"] == "FACADE_FKI"
    assert rows["FACADE_FKI"]["relative_error"] == -0.3333
    # The families that actually carry most untagged buildings disagree least,
    # which is why the headline percentage must not be read as scene-wide error.
    assert abs(rows["FACADE_GENERIC"]["relative_error"]) < 0.02
    assert abs(rows["FACADE_RESIDENTIAL"]["relative_error"]) < 0.05
    assert any(
        "contradict the source floor count" in reason
        for reason in report["blocking_reasons"]
    )


def test_reconciling_the_storey_heights_would_clear_the_finding(
    tmp_path: Path,
) -> None:
    # Guards the arithmetic: if every painted floor height equalled the
    # importer's 3.2 m, the disagreement would be zero. Without this the test
    # above could pass on a metric that is simply always false.
    text = SHADER.read_text(encoding="utf-8")
    for painted in ("3.25", "4.57", "4.0", "3.05", "3.8", "4.65", "3.55", "4.8", "3.6"):
        text = text.replace(f"floor_height = {painted};", "floor_height = 3.2;")
    shader = tmp_path / "scene.frag"
    shader.write_text(text, encoding="utf-8")

    report = build_report(shader, SCENE_MODULE, SCENE_ASSET, EVIDENCE, ATTRIBUTION)

    assert report["storey_height_consistency"]["storey_heights_agree"]
    assert not any(
        "contradict the source floor count" in reason
        for reason in report["blocking_reasons"]
    )


def test_the_height_path_classification_agrees_with_the_importer() -> None:
    # The audit restates _height's branch order to name which rule fired. If
    # the importer's order ever changed, every attribution below would be
    # mislabelled while still looking plausible, so this checks the
    # classification against the function it describes, building by building.
    from simulator.scene import _height

    snapshot = json.loads(BUILDING_TAGS.read_text(encoding="utf-8"))
    checked = {"height_tag": 0, "levels_times_assumed": 0, "untagged_default": 0}
    for element in snapshot["elements"]:
        tags = element["tags"]
        raw = tags.get("height", "").lower().replace("m", "").strip()
        try:
            expected = float(raw)
            source = "height_tag"
        except ValueError:
            try:
                expected = max(3.2, float(tags.get("building:levels", "")) * 3.2)
                source = "levels_times_assumed"
            except ValueError:
                expected = 12.0
                source = "untagged_default"
        assert _height(tags) == expected, element["id"]
        checked[source] += 1

    assert checked == {
        "height_tag": 652,
        "levels_times_assumed": 206,
        "untagged_default": 456,
    }


def test_the_mismatch_is_weighted_by_buildings_that_actually_take_the_path() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    consistency = report["storey_height_consistency"]
    attribution = report["height_source_attribution"]
    rows = {row["name"]: row for row in consistency["rows"]}

    # The point of the attribution: the two worst-disagreeing families hold no
    # buildings on the levels path at all, so the headline percentage describes
    # nothing in the scene.
    assert rows["FACADE_FKI"]["buildings_on_the_levels_path"] == 0
    assert rows["FACADE_PARC1"]["buildings_on_the_levels_path"] == 0
    assert consistency["worst_family"]["name"] == "FACADE_FKI"
    assert consistency["worst_family_with_exposed_buildings"]["name"] == (
        "FACADE_GLASS_BLUE"
    )
    # 202 of the 206 exposed buildings disagree by no more than 5%.
    assert attribution["exposed_building_count"] == 206
    assert attribution["exposed_within_5_percent"] == 202
    assert rows["FACADE_RESIDENTIAL"]["buildings_on_the_levels_path"] == 172


def test_the_untagged_default_carries_a_third_of_the_buildings() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    counts = report["height_source_attribution"]["height_source_counts"]

    # A single unsourced 12.0 m constant sets the mass of more buildings than
    # the storey disagreement affects, which is the larger finding of the two.
    assert counts["untagged_default"] == 456
    assert sum(counts.values()) == report["height_source_attribution"]["way_count"]
    assert any(
        "neither height nor building:levels" in reason
        for reason in report["blocking_reasons"]
    )


def test_the_snapshot_is_dated_to_the_event_and_carries_no_geometry() -> None:
    snapshot = json.loads(BUILDING_TAGS.read_text(encoding="utf-8"))

    assert snapshot["snapshot_utc"] == "2024-10-05T10:20:00Z"
    assert snapshot["licence"] == "ODbL 1.0"
    assert snapshot["geometry_omitted"] is True
    assert snapshot["elements"]
    assert all("geometry" not in element for element in snapshot["elements"])
    assert all(
        element["tags"].get("building") or element["tags"].get("building:part")
        for element in snapshot["elements"]
    )


def test_the_audit_still_runs_without_the_snapshot(tmp_path: Path) -> None:
    # The attribution is evidence layered on top, not a hard dependency: the
    # module and shader checks must still work on a checkout that lacks it.
    report = build_report(
        SHADER,
        SCENE_MODULE,
        SCENE_ASSET,
        EVIDENCE,
        ATTRIBUTION,
        tmp_path / "absent.json",
    )

    assert report["height_source_attribution"] is None
    assert report["coverage"]["styles_consistent"]
    assert not report["storey_height_consistency"]["storey_heights_agree"]


def test_the_style_constants_and_landmark_heights_still_parse() -> None:
    scene_module = SCENE_MODULE.read_text(encoding="utf-8")

    assert declared_styles(scene_module)[2.0] == "FACADE_GOLD_63"
    assert landmark_heights(scene_module)["parc1_tower1"] == 318.0
    assert 12.0 in shader_styles(SHADER.read_text(encoding="utf-8"))
