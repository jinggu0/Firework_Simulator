from __future__ import annotations

import copy

import moderngl
import numpy as np
import pytest

from simulator import shaders
from tools.measure_background_shimmer import (
    BACKGROUND, PARENT_REGIONS, REGIONS, background_fragment, background_pair_masks, compare_parent_partition,
)
from tools.measure_subject_shimmer import subject_pair_masks
from tools.measure_temporal_shimmer import render_pan


@pytest.mark.parametrize("radius", [1, 7])
def test_background_split_preserves_subject_and_transition_masks(radius):
    previous = np.zeros((48, 96, 3), dtype=np.float16)
    previous[4:44, 3:27, 0] = 1.
    previous[4:44, 28:53, 1] = 1.
    previous[4:44, 53:78, 2] = 1.
    current = np.roll(previous, -3, axis=1)
    coordinates = np.mgrid[:48, :96].astype(float)
    coordinates[1] += 3.6
    masks = background_pair_masks(previous, current, coordinates, radius)
    parent = subject_pair_masks(previous, current, coordinates, radius)
    assert tuple(masks) == REGIONS
    assert np.all(np.stack(list(masks.values())).sum(axis=0) == 1)
    for name, children in PARENT_REGIONS.items():
        assert np.array_equal(np.logical_or.reduce([masks[child] for child in children]), parent[name])
    assert masks["facade_interior"][20, 36]
    assert masks["water_interior"][20, 62]
    assert masks["facade_water_mixed"][20, 49]
    assert masks["subject_transition"][20, 23]  # subject edge takes precedence over backdrop
    assert masks["other"][20, 83]


def test_partial_facade_and_water_remain_mixed_and_overlap_fails_closed():
    labels = np.zeros((20, 40, 3), dtype=np.float16)
    labels[:, 8:18, 1] = .5
    labels[:, 20:30, 2] = .5
    coords = np.mgrid[:20, :40]
    result = background_pair_masks(labels, labels, coords, 1)
    assert result["facade_mixed"][10, 12]
    assert result["water_mixed"][10, 25]
    labels[10, 12] = (.5, .5, .5)
    with pytest.raises(ValueError, match="overlapping"):
        background_pair_masks(labels, labels, coords, 1)
    with pytest.raises(ValueError, match="requires subject capture"):
        render_pan(None, "human_vision", 24, .6, subject_fragment_builder=background_fragment)


def test_collapsing_to_historical_partitions_checks_counts_and_both_energies():
    def row(samples, display, hdr):
        return dict(samples=samples, display_energy=display, hdr_energy=hdr)

    current = {"source_support_radius": 1, "scored_pixel_pairs": 8,
               "regions": {name: row(1, 2., 3.) for name in REGIONS}}
    old = {"source_support_radius": 1, "scored_pixel_pairs": 8,
           "regions": {"subject_interior": row(1, 2., 3.), "subject_transition": row(1, 2., 3.),
                       "non_subject": row(len(BACKGROUND), 12., 18.)}}
    assert compare_parent_partition(current, old)["non_subject"]["hdr_energy"]["difference"] == 0.
    for key, message in (("samples", "sample count"), ("display_energy", "display_energy"), ("hdr_energy", "hdr_energy")):
        bad = copy.deepcopy(current)
        bad["regions"]["facade_mixed"][key] += 1
        with pytest.raises(ValueError, match=message):
            compare_parent_partition(bad, old)


@pytest.mark.opengl
def test_background_shader_keeps_discards_and_links_against_shipped_geometry():
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as error:
        pytest.skip(f"OpenGL unavailable: {error}")
    try:
        for kind in ("scene", "land", "water"):
            original = shaders.source(f"{kind}.frag")
            modified = background_fragment(original, kind)
            assert modified.count("discard;") == original.count("discard;") + 1
            program = ctx.program(vertex_shader=shaders.source(f"{kind}.vert"), fragment_shader=modified)
            try:
                assert "audit_isolated" in program
                assert ("audit_eye" in program) == (kind == "scene")
                assert ("audit_bounds" in program) == (kind == "scene")
            finally:
                program.release()
        assert ctx.error == "GL_NO_ERROR"
    finally:
        ctx.release()
