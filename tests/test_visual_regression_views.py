from __future__ import annotations

import json

import numpy as np
import pytest

from simulator.camera import FreeCamera
from simulator.scene import load_scene
from simulator.terrain import TerrainSurface
from simulator.validation.views import (
    VisualViewError,
    load_visual_regression_suite,
)
from tools.capture_visual_baselines import (
    compare_capture_directories,
    select_views,
)


def test_canonical_visual_suite_is_tied_to_the_shipped_scene() -> None:
    suite = load_visual_regression_suite()
    scene_path = suite.verify_scene_asset()
    assert suite.scenario_id == "yeouido-2024-10-05"
    assert suite.source["confidence_grade"] == "C"
    assert len(suite.views) == 6
    assert scene_path.name == "yeouido_scene.npz"


def test_visual_views_are_safe_and_land_anchored() -> None:
    suite = load_visual_regression_suite()
    scene = load_scene(suite.verify_scene_asset())
    terrain = TerrainSurface(
        scene.terrain_height_m,
        scene.terrain_bounds,
        scene.water_mask,
        scene.water_mask_bounds,
    )
    for view in suite.views:
        x_m, y_m, z_m = view.position_eus_m
        assert view.expected_surface == "land"
        assert not terrain.is_water(x_m, z_m)
        assert (
            y_m - terrain.height_at(x_m, z_m)
            >= view.minimum_ground_clearance_m
        )


def test_visual_view_orientation_matches_its_declared_target() -> None:
    camera = FreeCamera()
    for view in load_visual_regression_suite().views:
        view.apply(camera)
        expected = np.asarray(view.target_eus_m) - np.asarray(
            view.position_eus_m
        )
        expected /= np.linalg.norm(expected)
        assert np.dot(camera.forward, expected) == pytest.approx(1.0, abs=1e-6)
        assert not camera.walking
        assert np.allclose(camera.velocity_mps, 0.0)


def test_view_selection_retains_canonical_order() -> None:
    suite = load_visual_regression_suite()
    selected = select_views(suite, ["water_reflection", "terrain_shoreline"])
    assert [view.view_id for view in selected] == [
        "terrain_shoreline",
        "water_reflection",
    ]
    with pytest.raises(VisualViewError, match="unknown visual-regression"):
        select_views(suite, ["historical_observer"])


def test_capture_comparison_rejects_a_changed_view_definition(tmp_path) -> None:
    directories = [tmp_path / "before", tmp_path / "after"]
    for directory, digest in zip(directories, ("a" * 64, "b" * 64)):
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "suite_id": "suite",
                    "scenario_id": "scenario",
                    "scene_asset_sha256": "c" * 64,
                    "views_asset_sha256": digest,
                    "display_mode": "human_vision",
                    "frames": 8,
                    "captures": [],
                }
            ),
            encoding="utf-8",
        )
    comparison = compare_capture_directories(*directories)
    assert not comparison["passed"]
    assert "views_asset_sha256" in comparison["metadata_mismatches"]
