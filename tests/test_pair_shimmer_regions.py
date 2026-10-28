from __future__ import annotations

import numpy as np
import pytest

from simulator.validation.temporal_shimmer import pan_source_coordinates, shimmer_scores
from tools.pair_shimmer_regions import SUPPORT_RADII, pair_masks, pair_statistics, source_support


def coordinates(height=48, width=64, shift=.6):
    result = np.mgrid[:height, :width].astype(float)
    result[1] += shift
    return result


def fractions():
    result = np.zeros((48, 64, 3), np.float16)
    result[4:40, 4:48, 0] = 1
    result[12:32, 16:36, 1] = 1
    result[20:, 52:, 2] = 1
    return result


@pytest.mark.parametrize("radius", SUPPORT_RADII)
def test_support_matches_raw_source_rectangle_not_target_erosion(radius):
    mask = np.random.default_rng(12).random((48, 64)) > .01
    coords = coordinates()
    coords[0] += .3
    actual = source_support(mask, coords, radius)
    for row, col in ((24, 30), (12, 16), (8, 8), (46, 63), (0, 0)):
        r, c = np.floor(coords[:, row, col]).astype(int)
        lo_r, lo_c, hi_r, hi_c = r - radius, c - radius, r + radius + 2, c + radius + 2
        expected = (lo_r >= 0 and lo_c >= 0 and hi_r <= 48 and hi_c <= 64
                    and mask[lo_r:hi_r, lo_c:hi_c].all())
        assert actual[row, col] == expected


@pytest.mark.parametrize("radius", SUPPORT_RADII)
def test_pair_partition_is_disjoint_complete_and_interior_survives(radius):
    source = fractions()
    masks = pair_masks(source, source, coordinates(), radius)
    assert np.all(np.stack(list(masks.values())).sum(axis=0) == 1)
    assert masks["pane_interior_clean_support"][22, 26]
    edge_class = "facade_boundary_or_mixed_support" if radius else "wall_interior_clean_support"
    assert masks[edge_class][4, 4]


def test_reprojection_uses_earlier_source_not_same_screen_coordinate():
    source = fractions()
    current = source.copy()
    current[..., 1] = 0
    current[12:32, 11:31, 1] = 1
    masks = pair_masks(source, current, coordinates(shift=5.3), 1)
    assert source[20, 12, 1] == 0
    assert masks["pane_interior_clean_support"][20, 12]
    assert not pair_masks(source, current, coordinates(shift=-5.3), 1)["pane_interior_clean_support"][20, 12]


def test_pane_centres_are_not_clean_when_cubic_support_touches_wall():
    source = fractions()
    # Query col=16.6: its 2x2 raw centres are pane, but its cubic
    # coefficient footprint includes col=15, which is wall.
    coords = coordinates()
    centre = pair_masks(source, source, coords, 0)
    cubic = pair_masks(source, source, coords, 1)
    assert centre["pane_interior_clean_support"][20, 16]
    assert cubic["pane_centres_mixed_support"][20, 16]
    assert not cubic["pane_interior_clean_support"][20, 16]
    # A facade edge in that same support takes precedence over pane labels.
    source[20, 15, :2] = 0
    assert pair_masks(source, fractions(), coords, 1)["facade_boundary_or_mixed_support"][20, 16]


def test_large_support_cannot_create_clean_pane_samples():
    source = fractions()
    masks = [pair_masks(source, source, coordinates(), radius) for radius in SUPPORT_RADII]
    for smaller, larger in zip(masks, masks[1:]):
        assert np.all(~larger["pane_interior_clean_support"] | smaller["pane_interior_clean_support"])


@pytest.mark.parametrize("empty", [False, True])
def test_pair_energy_conserves_existing_scores_and_empty_bins_are_null(empty):
    rng = np.random.default_rng(8)
    values = rng.random((21, 48, 64)) + .1
    matrices = np.repeat(np.eye(4)[None], len(values), axis=0)
    coords = np.stack([pan_source_coordinates(matrices[0], matrices[1], 64, 48)] * 20)
    display = values * 120
    display_score, scored = shimmer_scores(display, coords, relative=False)
    hdr_score, _ = shimmer_scores(values, coords, relative=True)
    if empty:
        scored[:] = False
    pan = {"display_luma": display, "linear_hdr": values, "matrices": matrices,
           "semantic_fractions": np.repeat(fractions()[None], len(values), axis=0)}
    run = {"scored": scored, "display_score": display_score, "hdr_score": hdr_score}
    for result in pair_statistics(pan, run):
        assert sum(r["samples"] for r in result["regions"].values()) == 20 * scored.sum()
        for key in ("display", "hdr"):
            total_share = sum(r[f"{key}_energy_share"] for r in result["regions"].values())
            assert total_share == pytest.approx(0 if empty else 1)
            assert result["conservation"][key]["pair_energy_sum"] == pytest.approx(
                20 * (run[f"{key}_score"][scored] ** 2).sum())
            if empty:
                assert all(r[f"{key}_conditional_rms"] is None for r in result["regions"].values())
    run["display_score"] += 1
    if not empty:
        with pytest.raises(ValueError, match="reconstruct"):
            pair_statistics(pan, run)


def test_invalid_semantics_coordinates_and_radius_fail_closed():
    source = fractions()
    bad = source.copy()
    bad[0, 0, 1] = 1
    with pytest.raises(ValueError, match="semantic"):
        pair_masks(source, bad, coordinates(), 1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        source_support(source[..., 0], coordinates(), -1)
    with pytest.raises(ValueError, match="finite source"):
        source_support(source[..., 0], coordinates() * np.nan, 1)


def test_pair_report_cli_keeps_sequences_control_and_portable_defaults(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from tools import map_pair_shimmer as command

    rng = np.random.default_rng(2)
    values = rng.random((21, 48, 64)) + .1
    matrices = np.repeat(np.eye(4)[None], len(values), axis=0)
    sequence = {"linear_hdr": values, "display_luma": values * 120, "matrices": matrices,
                "semantic_fractions": np.repeat(fractions()[None], len(values), axis=0),
                "gl": {}, "yaw_step_deg": 0., "render_config": {"msaa_samples": 8},
                "semantic_sources": {"scene": "test"}}
    captures = []

    def render(*args, **kwargs):
        captures.append(kwargs.get("semantic_capture", False))
        return dict(sequence)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(command, "render_pan", render)
    monkeypatch.setattr(command, "load_visual_regression_suite", lambda: SimpleNamespace(
        display_mode="test", view=lambda name: SimpleNamespace(view_id=name)))
    monkeypatch.setattr(command, "load_known_residuals", lambda: [])
    monkeypatch.setattr(command, "select_residuals", lambda *args: [])
    monkeypatch.setattr(sys, "argv", ["map_pair_shimmer", "--view", "test_view", "--frames", "21"])
    command.main()
    assert captures == [False, True]
    report_path = tmp_path / "docs/validation/shimmer_ab_v2/diagnostics/results/pair_test_view_attribution.json"
    report = json.loads(report_path.read_text())
    assert report["control_within_one_percent"]
    assert report["support_radii_declared_before_measurement"] == [0, 1, 7]
    assert report_path.with_suffix(".png").exists()
    with np.load(tmp_path / "local_reference/shimmer_regions_v2/pair_test_view.npz") as cached:
        for key in ("linear_hdr", "display_luma", "semantic_fractions", "matrices"):
            assert np.array_equal(cached[key], sequence[key])
