"""Locate the residual trigger beyond the single-quad production program.

V0-12 showed that the exact facade quad, production shader, viewport, and
depth state are stable in isolation.  This ladder cumulatively restores the
full building VAO, all scene VAOs, preceding opaque draws, and per-frame
uniform rewrites while reading only the known eight-pixel rectangle.

Example::

    python -m tools.probe_facade_batch_ladder --iterations 2048
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import moderngl  # noqa: E402
import pygame  # noqa: E402

from simulator.app import SimulatorApp  # noqa: E402
from simulator.config import SimulationConfig  # noqa: E402
from simulator.renderer import TERRAIN_UNIT, WATER_MASK_UNIT  # noqa: E402
from simulator.scenario import DEFAULT_SCENARIO_PATH  # noqa: E402
from simulator.validation.views import load_visual_regression_suite  # noqa: E402
from tools.probe_facade_pass_ladder import (  # noqa: E402
    REGION_HALF_OPEN,
    _measure_draws,
    locate_residual_quad,
)
from tools.probe_facade_shader_reproducer import _context_metadata  # noqa: E402


STAGES = (
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


def _bind_opaque_textures(app: SimulatorApp) -> None:
    renderer = app.renderer
    renderer.water_mask_texture.use(WATER_MASK_UNIT)
    renderer.terrain_texture.use(TERRAIN_UNIT)
    renderer.scene.scanned_materials.bind()


def _zero_light_counts(app: SimulatorApp) -> None:
    renderer = app.renderer
    renderer.scene.program["static_light_count"] = 0
    renderer.scene.program["dynamic_light_count"] = 0
    renderer.land.program["static_light_count"] = 0


def _draw_building_batch(app: SimulatorApp) -> None:
    vao, vertex_count = app.renderer.scene.vaos[0]
    vao.render(moderngl.TRIANGLES, vertices=vertex_count)


def _draw_scene_vao_prefix(app: SimulatorApp, count: int) -> None:
    for vao, vertex_count in app.renderer.scene.vaos[:count]:
        vao.render(moderngl.TRIANGLES, vertices=vertex_count)


def _draw_scene(app: SimulatorApp) -> None:
    app.renderer.scene.draw()


def _draw_land_scene(app: SimulatorApp) -> None:
    renderer = app.renderer
    renderer.land.draw()
    renderer.scene.draw()


def _draw_sky_land_scene(app: SimulatorApp) -> None:
    renderer = app.renderer
    app.ctx.disable(moderngl.DEPTH_TEST)
    renderer.sky.draw()
    app.ctx.enable(moderngl.DEPTH_TEST)
    renderer.land.draw()
    renderer.scene.draw()


def _refresh_frame_uniforms(app: SimulatorApp) -> None:
    """Repeat the main-frame uniform writes without running another pass."""

    renderer = app.renderer
    renderer._update_environment_animation(app.world.atmosphere)
    renderer._update_static_lights(app.camera)
    renderer._update_celestial(app.celestial, app.world.atmosphere)
    renderer._update_dynamic_lights(app.world)
    renderer._update_camera(app.camera)
    # V0-10 proved that the residual survives with static lights disabled.
    # Keep that condition so this stage isolates the writes, not light energy.
    _zero_light_counts(app)


def _uniform_refresh_draw(app: SimulatorApp) -> None:
    _refresh_frame_uniforms(app)
    _bind_opaque_textures(app)
    _draw_sky_land_scene(app)


def stage_draws(app: SimulatorApp) -> dict[str, Callable[[], None]]:
    """Return the ordered cumulative draw callbacks used by the probe."""

    return {
        "building_batch": lambda: _draw_building_batch(app),
        "scene_vaos_2": lambda: _draw_scene_vao_prefix(app, 2),
        "scene_vaos_3": lambda: _draw_scene_vao_prefix(app, 3),
        "scene_vaos_4": lambda: _draw_scene_vao_prefix(app, 4),
        "scene_vaos_5": lambda: _draw_scene_vao_prefix(app, 5),
        "scene_pass": lambda: _draw_scene(app),
        "land_scene": lambda: _draw_land_scene(app),
        "sky_land_scene": lambda: _draw_sky_land_scene(app),
        "uniform_refresh_sky_land_scene": lambda: _uniform_refresh_draw(app),
    }


def probe(iterations: int) -> dict[str, Any]:
    """Execute every batch/state condition in one production GL context."""

    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    quad = locate_residual_quad()
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        load_visual_regression_suite().view("water_reflection").apply(app.camera)
        # Establish the exact production texture and uniform state. Subsequent
        # stages draw only into private validation targets.
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        _zero_light_counts(app)
        _bind_opaque_textures(app)

        draws = stage_draws(app)
        if tuple(draws) != STAGES:
            raise RuntimeError("batch ladder stage order changed")
        results: dict[str, dict[str, Any]] = {}
        for stage, draw in draws.items():
            _zero_light_counts(app)
            _bind_opaque_textures(app)
            results[stage] = _measure_draws(
                app.ctx,
                iterations,
                draw,
                depth=True,
            )
        varying = [stage for stage in STAGES if not results[stage]["bit_deterministic"]]
        scene = app.renderer.scene
        return {
            "schema_version": 1,
            "probe": "facade_batch_state_ladder_v0_13",
            "iterations_per_stage": iterations,
            "region_gl": list(REGION_HALF_OPEN),
            "gpu": _context_metadata(app.ctx),
            "quad": {
                "triangle_indices": list(quad.triangle_indices),
                "vertex_range": list(quad.vertex_range),
            },
            "batch": {
                "building_vertices": int(scene.vaos[0][1]),
                "scene_vao_vertices": [int(count) for _, count in scene.vaos],
                "grass_chunk_count": len(scene.grass_vaos),
            },
            "stages": results,
            "verdict": {
                "first_varying_stage": varying[0] if varying else None,
                "all_stages_bit_deterministic": not varying,
            },
        }
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        report = probe(arguments.iterations)
    except ValueError as error:
        parser.error(str(error))
    print(f"{report['gpu']['GL_VENDOR']} / {report['gpu']['GL_RENDERER']}")
    print(
        f"building_vertices={report['batch']['building_vertices']}  "
        f"scene_vaos={len(report['batch']['scene_vao_vertices'])}"
    )
    for stage, result in report["stages"].items():
        state = "stable" if result["bit_deterministic"] else "VARYING"
        print(
            f"{stage:34s} {state:7s}  "
            f"states={result['unique_states']}  "
            f"different={result['differing_iterations']}"
        )
    first = report["verdict"]["first_varying_stage"]
    print(f"verdict: first varying stage={first or 'none'}")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
