from __future__ import annotations

import numpy as np
import pytest

from tools import measure_subject_shimmer as probe
from tools.measure_temporal_shimmer import render_pan


def subject_mask():
    mask = np.zeros((32, 48, 3), dtype=np.float16)
    mask[5:28, 15:36, 0] = 1
    return mask


def test_subject_labels_follow_reprojection_and_keep_mixed_footprint():
    previous = subject_mask()
    current = np.roll(previous, -4, axis=1)
    coords = np.mgrid[:32, :48].astype(float)
    coords[1] += 4.6
    result = probe.subject_pair_masks(previous, current, coords, 1)
    assert np.all(np.stack(list(result.values())).sum(axis=0) == 1)
    assert result["subject_interior"][15, 12] and previous[15, 12, 0] == 0
    assert result["subject_transition"][15, 11]  # source window reaches the wall
    assert result["non_subject"][15, 6]
    assert probe.subject_pair_masks(previous, current, coords, 7)["subject_transition"][15, 12]
    current[15, 12, 0] = .5
    assert probe.subject_pair_masks(previous, current, coords, 1)["subject_transition"][15, 12]


def test_invalid_coverage_and_mixed_capture_modes_fail_before_rendering():
    mask = subject_mask()
    mask[15, 20, 0] = np.nan
    with pytest.raises(ValueError, match="subject coverage"):
        probe.subject_pair_masks(mask, mask, np.mgrid[:32, :48], 1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        render_pan(None, "human_vision", 24, .6, subject_capture=True, semantic_capture=True)
    with pytest.raises(ValueError, match="presented grid"):
        render_pan(None, "human_vision", 24, .6, subject_capture=True, supersample=2)


def test_replay_check_uses_fixed_floor_and_one_percent():
    control = dict(display_excess=1., hdr_excess=0., mean_display_code=20.)
    close = dict(display_excess=1.005, hdr_excess=1e-7, mean_display_code=20.1)
    assert probe.replay_control(control, close)["passed"]
    close["hdr_excess"] = 2e-6
    assert not probe.replay_control(control, close)["passed"]


def test_subject_baseline_controls_and_partition_use_same_scored_domain(tmp_path, monkeypatch):
    rng = np.random.default_rng(51)
    values = rng.random((probe.FRAMES, 32, 48)) + .1
    sequence = {
        "linear_hdr": values, "display_luma": 40 * values, "gl": {},
        "matrices": [np.eye(4)] * probe.FRAMES, "yaw_step_deg": 0.,
        "render_config": {"msaa_samples": 8}, "semantic_sources": {"scene": "fixture"},
        "semantic_fractions": np.repeat(subject_mask()[None], probe.FRAMES, axis=0),
    }
    calls = []

    def capture(_view, _mode, frames, speed, *, subject_capture=False):
        calls.append((frames, speed, subject_capture))
        if speed:
            return dict(sequence)
        still = np.repeat(values[:1], frames, axis=0)
        return {**sequence, "linear_hdr": still, "display_luma": 40 * still}

    monkeypatch.setattr(probe, "render_pan", capture)
    from simulator.validation.views import VisualRegressionView
    view = VisualRegressionView("sample", "road_ground", (0., 1.75, 0.), (0., 0., -20.), "land", 1.7, "test")
    report = probe.measure_view(view, "human_vision", {}, tmp_path)
    assert calls == [(probe.FRAMES, probe.PIXELS_PER_FRAME, False), (probe.FRAMES, 0, False),
                     (probe.FRAMES, probe.PIXELS_PER_FRAME, True)]
    assert report["replay_control"]["passed"]
    assert report["static"]["display_excess"] == 0.
    assert report["static"]["scored_pixels"] == report["plain_moving"]["scored_pixels"]
    assert (tmp_path / "sample.png").is_file()
    for partition in report["partitions"]:
        regions = partition["regions"]
        assert sum(item["samples"] for item in regions.values()) == partition["scored_pixel_pairs"]
        assert sum(item["display_energy_share"] for item in regions.values()) == pytest.approx(1.)
        assert sum(item["hdr_energy_share"] for item in regions.values()) == pytest.approx(1.)
