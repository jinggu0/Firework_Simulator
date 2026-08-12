from __future__ import annotations

from unittest.mock import MagicMock, Mock, call

from tools.probe_facade_batch_ladder import (
    STAGES,
    _refresh_frame_uniforms,
    stage_draws,
)


def test_batch_ladder_stage_order_is_cumulative() -> None:
    app = Mock()

    assert STAGES == (
        "building_batch",
        "scene_vaos_2",
        "scene_vaos_3",
        "scene_vaos_4",
        "scene_vaos_5",
        "scene_pass",
        "land_scene",
        "sky_land_scene",
        "uniform_refresh_sky_land_scene",
    )
    assert tuple(stage_draws(app)) == STAGES


def test_frame_uniform_refresh_matches_main_update_order() -> None:
    app = MagicMock()
    renderer = app.renderer
    app.world.atmosphere = "atmosphere"
    app.celestial = "celestial"
    app.camera = "camera"

    _refresh_frame_uniforms(app)

    assert renderer.method_calls[:5] == [
        call._update_environment_animation("atmosphere"),
        call._update_static_lights("camera"),
        call._update_celestial("celestial", "atmosphere"),
        call._update_dynamic_lights(app.world),
        call._update_camera("camera"),
    ]
    renderer.scene.program.__setitem__.assert_any_call("static_light_count", 0)
    renderer.scene.program.__setitem__.assert_any_call("dynamic_light_count", 0)
    renderer.land.program.__setitem__.assert_called_with("static_light_count", 0)
