"""Structural checks on the render passes that need no OpenGL context."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from simulator import passes, shaders
from simulator.passes import haze, land, particles, post, scene, sky, smoke, water
from simulator.renderer import TERRAIN_UNIT, WATER_MASK_UNIT, Renderer


def test_texture_units_do_not_collide() -> None:
    # Units are bare integers spread across pass modules now, so a collision
    # would silently bind the wrong texture rather than fail to compile.
    assignments = {
        "water_mask": WATER_MASK_UNIT,
        "terrain": TERRAIN_UNIT,
        "smoke_state_2d": smoke.SMOKE_STATE_2D_UNIT,
        "smoke_state_3d": smoke.SMOKE_STATE_3D_UNIT,
        "scene_depth": smoke.SCENE_DEPTH_UNIT,
        "reflection": water.REFLECTION_UNIT,
        "cloud_noise": sky.CLOUD_NOISE_UNIT,
        "star_catalogue": sky.STAR_CATALOGUE_UNIT,
        "hdr": post.HDR_UNIT,
        "bloom": post.BLOOM_UNIT,
        "adaptation": post.ADAPTATION_UNIT,
        "previous_adaptation": post.PREVIOUS_ADAPTATION_UNIT,
        "airlight": haze.AIRLIGHT_UNIT,
    }
    units = list(assignments.values())
    assert len(units) == len(set(units)), sorted(assignments.items())


def test_passes_agree_on_the_shared_unit_numbers() -> None:
    assert water.WATER_MASK_UNIT == land.WATER_MASK_UNIT == WATER_MASK_UNIT
    assert land.TERRAIN_UNIT == TERRAIN_UNIT
    # Same texture, same unit: haze and smoke both terminate against opaque
    # geometry and never bind it at the same time.
    assert haze.SCENE_DEPTH_UNIT == smoke.SCENE_DEPTH_UNIT


def test_pass_modules_do_not_import_the_renderer() -> None:
    # The dependency runs one way: the coordinator knows the passes, not the
    # other way round. A cycle here would defeat the split.
    for module in (sky, scene, water, land, particles, post, smoke, haze):
        text = inspect.getsource(module)
        assert "from ..renderer" not in text, module.__name__
        assert "import renderer" not in text, module.__name__


def test_public_pass_surface_is_exported() -> None:
    for name in (
        "SkyPass",
        "ScenePass",
        "WaterPass",
        "LandPass",
        "ParticlePass",
        "SmokePass",
        "HazePass",
        "PostProcessPass",
        "RenderTargets",
    ):
        assert hasattr(passes, name), name


def test_empty_scene_data_covers_the_whole_extent() -> None:
    # Used when the geospatial asset is absent; the mask must read as water
    # everywhere so the river pass still draws something coherent.
    data = scene.SceneData.empty()
    assert data.water_mask.min() == 255
    assert data.static_light_positions.shape == (0, 3)
    assert data.water_mask_bounds[0] < 0 < data.water_mask_bounds[2]
    assert data.terrain_height_m.max() == 0.0


def test_wind_bearing_conversion_matches_the_meteorological_convention() -> None:
    # An East-Up-South wind vector blowing toward the south came from the
    # north, which is bearing 360 rather than 180.
    assert water.wind_from_bearing_deg(np.array([0.0, 3.0])) == pytest.approx(
        0.0, abs=1e-6
    )
    assert water.wind_from_bearing_deg(np.array([4.0, 0.0])) == pytest.approx(
        270.0, abs=1e-6
    )


def test_particle_stride_matches_the_vertex_layout() -> None:
    # position(3) + trail start(3) + colour(3) + power(1)
    assert particles.STRIDE == 10


def test_renderer_exposes_the_accessors_tools_depend_on() -> None:
    for name in (
        "significant_wave_height_m",
        "star_catalogue_is_measured",
        "atmospheric_optics",
        "hdr_texture",
        "frame_index",
        "visibility_m",
        "surface_extinction",
    ):
        attribute = getattr(Renderer, name)
        assert isinstance(attribute, property), name


def test_smoke_box_indices_form_twelve_triangles() -> None:
    assert smoke.BOX_INDICES.shape == (36,)
    assert set(smoke.BOX_INDICES.tolist()) == set(range(8))


def test_the_plume_is_drawn_for_the_reflection_too() -> None:
    # The river shows the plume above it. Both draws share one pass, so the
    # reflected one is a flag rather than a second object.
    signature = inspect.signature(smoke.SmokePass.draw)
    assert "reflected_path" in signature.parameters
    assert signature.parameters["reflected_path"].default is False
    source = inspect.getsource(smoke)
    assert 'self.program["reflected_path"]' in source


def test_the_facing_test_is_set_per_draw_not_per_fluid_revision() -> None:
    # The frame draws the plume from two cameras — the viewer's and the one
    # mirrored below the water datum — and only one of them can be inside the
    # proxy box. Deciding that when the fluid last changed would carry the
    # wrong answer into the second draw.
    assert hasattr(smoke.SmokePass, "set_camera_inside")
    refresh = inspect.getsource(smoke.SmokePass._refresh_bounds)
    assert "camera_inside" not in refresh
    assert "camera_inside" not in inspect.getsource(Renderer._update_camera)


def test_the_reflected_plume_clips_its_air_path_at_the_datum() -> None:
    # The mirrored ray's below-datum half is the water pixel's own path to the
    # eye, which the main haze pass applies to that pixel.
    source = shaders.source("smoke.frag")
    assert "uniform int reflected_path;" in source
    assert "datum_m" in source and "air_height_m" in source
