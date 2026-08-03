from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulator.validation.material_detail import (
    projection_sampling_report,
    scanned_material_report,
    temporal_delta_metrics,
)


REPORT = Path("docs/validation/material_detail_v1/material_detail_report.json")


def test_ground_footprint_and_anisotropy_grow_with_distance() -> None:
    samples = projection_sampling_report()["samples"]

    assert [sample["distance_m"] for sample in samples] == [
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ]
    assert all(
        current["along_view_m_per_px"] < following["along_view_m_per_px"]
        for current, following in zip(samples, samples[1:])
    )
    assert samples[-1]["footprint_anisotropy"] > 20.0


def test_scanned_material_scale_matches_manifest_and_files() -> None:
    records = scanned_material_report()

    assert [record["asset_id"] for record in records] == [
        "asphalt_04",
        "concrete_pavers",
        "leafy_grass",
        "concrete",
    ]
    assert all(record["resolution_px"] == [1024, 1024] for record in records)
    assert all(record["texel_pitch_mm"] > 0.0 for record in records)
    assert all(len(record["sampling"]) == 6 for record in records)


def test_temporal_metric_separates_static_and_phase_changing_frames() -> None:
    static = np.ones((3, 32, 32, 4), dtype=np.float32)
    checker = np.indices((32, 32)).sum(axis=0) % 2
    moving = static.copy()
    moving[1, ..., :3] = checker[..., None]
    moving[2, ..., :3] = (1 - checker)[..., None]

    static_metrics = temporal_delta_metrics(static)
    moving_metrics = temporal_delta_metrics(moving)

    assert static_metrics["mean_absolute_frame_delta"] == 0.0
    assert static_metrics["two_percent_signal_flip_fraction"] == 0.0
    assert moving_metrics["mean_absolute_frame_delta"] > 0.0
    assert moving_metrics["high_frequency_delta_fraction"] > 1.0


def test_committed_material_baseline_is_measurement_not_site_tuning() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["stage"] == "V2-1a"
    assert report["material_source"]["site_identity_confidence"] == "D"
    assert not report["material_source"]["sampled_at_yeouido"]
    assert report["runtime_contract"]["world_metric_uv"]
    assert report["runtime_contract"]["mipmaps_built"]
    assert report["runtime_contract"]["trilinear_minification"]
    assert not report["runtime_contract"][
        "anisotropic_filter_explicitly_configured"
    ]
    assert report["motion_capture"]["performed"]
    assert [
        capture["view_id"] for capture in report["motion_capture"]["captures"]
    ] == ["grass_close", "road_ground"]
    assert not report["gates"]["site_colour_tuning_allowed"]
    assert not report["gates"]["temporal_shimmer_gate_defined"]
