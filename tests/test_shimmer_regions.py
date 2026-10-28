from __future__ import annotations

import moderngl
import numpy as np
import pytest

from simulator import shaders
from tools.shimmer_regions import ASSIGNMENT, region_masks, region_statistics, semantic_fragment


def fixture():
    fractions = np.zeros((3, 32, 48, 3), np.float16)
    fractions[:, 4:24, 4:28, 0] = 1
    fractions[:, 10:15, 12:18, 1] = 1
    fractions[:, 15:18, 12:18, 1] = .5
    fractions[:, 22:, 32:, 2] = 1
    return fractions


@pytest.mark.parametrize("radius", [2, 4])
def test_partition_is_disjoint_and_covers_every_pixel(radius):
    masks = region_masks(fixture(), radius)
    assert np.all(np.stack(list(masks.values())).sum(axis=0) == 1)
    assert masks["pane_interior"][12, 14]
    assert masks["pane_transition"][16, 14]
    assert masks["water_interior"][25, 40]
    assert masks["wall_interior"][12, 22]


def test_later_pose_changes_membership_not_just_first_pose():
    fractions = fixture()
    fractions[2, 12, 14, 1] = 0
    masks = region_masks(fractions, 2)
    assert not masks["pane_interior"][12, 14]
    assert masks["pane_transition"][12, 14]
    fractions[2, 12, 14, :2] = (.5, 0)
    assert region_masks(fractions, 2)["facade_boundary_band"][12, 14]


def test_spatial_mass_is_conserved_and_empty_scores_do_not_produce_nan():
    fractions = fixture()
    display = np.arange(32 * 48).reshape(32, 48) / 100
    hdr = display / 50
    scored = np.ones(display.shape, bool)
    run = {"display_score": display, "hdr_score": hdr, "scored": scored}
    result = region_statistics(run, fractions, 2)
    for key, values, threshold in (("display", display, 2), ("hdr", hdr, .05)):
        assert sum(r[f"{key}_excess_share"] for r in result["regions"].values()) == pytest.approx(1)
        reconstructed = sum(r["pixel_fraction"] * r[f"{key}_excess_mean"] for r in result["regions"].values())
        assert reconstructed == pytest.approx(np.maximum(values - threshold, 0).mean())
    run["scored"][:] = False
    assert all(r["display_excess_share"] == 0 and r["pixels"] == 0
               for r in region_statistics(run, fractions, 2)["regions"].values())


def test_invalid_channel_fractions_fail_closed():
    fractions = fixture()
    fractions[0, 0, 0, 1] = 1
    with pytest.raises(ValueError, match="channels"):
        region_masks(fractions, 2)
    with pytest.raises(ValueError, match="positive integer"):
        region_masks(fixture(), 0)
    with pytest.raises(ValueError, match="contract"):
        semantic_fragment(shaders.source("scene.frag") + "\nfrag_color=vec4(0);", "scene")


@pytest.mark.opengl
def test_semantic_programs_link_and_keep_original_clip_tests():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        for kind in ("scene", "land", "water"):
            source = shaders.source(kind + ".frag")
            patched = semantic_fragment(source, kind)
            assert ASSIGNMENT.sub("OUTPUT;", source) == ASSIGNMENT.sub("OUTPUT;", patched)
            assert patched.count("discard;") == source.count("discard;")
            program = ctx.program(vertex_shader=shaders.source(kind + ".vert"), fragment_shader=patched)
            program.release()
        assert ctx.error == "GL_NO_ERROR"
    finally:
        ctx.release()
