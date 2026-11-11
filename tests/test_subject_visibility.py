from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import moderngl
import numpy as np
import pytest

from simulator import shaders
from simulator.validation.subject_visibility import measure_visibility, visibility_gates
from simulator.validation.views import load_visual_regression_suite
from tools.audit_subject_visibility import visibility_fragment


def test_visibility_distinguishes_screen_coverage_and_occlusion():
    full = np.zeros((8, 16, 3), dtype=np.float16)
    isolated = full.copy()
    full[3:5, 6:10, 0] = .5
    isolated[3:5, 6:10, 0] = 1
    full[0, :, 1] = .25
    result = measure_visibility(full, isolated)
    assert result["visible_screen_fraction"] == .03125
    assert result["central_subject_fraction"] == .5
    assert result["visible_projected_fraction"] == .5
    assert result["near_non_subject_fraction"] == .03125
    assert visibility_gates("bridge_structure", result) == {
        "screen_coverage": True, "central_coverage": True,
        "projected_visibility": False, "near_occluders": True,
    }


def test_empty_target_and_close_blocker_cannot_pass():
    full = np.zeros((8, 8, 3))
    alone = full.copy()
    full[:4, :, 1] = 1
    result = measure_visibility(full, alone)
    assert result["visible_projected_fraction"] is None
    assert not any(visibility_gates("road_ground", result).values())


def test_separate_suite_exports_the_freshly_verified_winners():
    directory = Path(__file__).resolve().parents[1] / "docs/validation/subject_visibility_v0"
    suite = load_visual_regression_suite(directory / "candidate_views.json")
    canonical = load_visual_regression_suite()
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert suite.suite_id != canonical.suite_id
    assert suite.scene_asset_sha256 == report["scene_asset_sha256"]
    for view in suite.views:
        if view.subject in report["subjects"]:
            outcome = report["subjects"][view.subject]
            assert view.view_id != view.subject
            assert json.loads(json.dumps(asdict(view))) == outcome["selected_view"]
            fresh = outcome["fresh_captures"][view.view_id]
            assert fresh["gates"] == visibility_gates(view.subject, fresh["metrics"])
            assert all(fresh["gates"].values())


@pytest.mark.parametrize("error", ["visible_without_support", "isolated_blocker", "overlap", "nan", "shape"])
def test_visibility_rejects_impossible_masks(error):
    full = np.zeros((8, 8, 3))
    alone = full.copy()
    if error == "visible_without_support":
        full[0, 0, 0] = .125
    elif error == "isolated_blocker":
        alone[0, 0, 1] = .125
    elif error == "overlap":
        full[0, 0] = (.5, .5, .5)
    elif error == "nan":
        full[0, 0, 0] = np.nan
    else:
        alone = alone[:-1]
    with pytest.raises(ValueError):
        measure_visibility(full, alone)


@pytest.mark.opengl
def test_visibility_replay_shaders_link_and_keep_production_discards():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        for kind in ("scene", "land", "water"):
            original = shaders.source(f"{kind}.frag")
            modified = visibility_fragment(original, kind)
            assert modified.count("discard;") == original.count("discard;") + 1
            program = ctx.program(vertex_shader=shaders.source(f"{kind}.vert"), fragment_shader=modified)
            try:
                assert "audit_eye" in program and "audit_isolated" in program
                assert ("audit_material" in program) == (kind == "scene")
            finally:
                program.release()
        assert ctx.error == "GL_NO_ERROR"
    finally:
        ctx.release()
