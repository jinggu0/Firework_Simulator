"""Separate building-buffer offset from the number of vertices processed.

V0-13 reproduced residual B with the complete building VAO but not a compact
six-vertex copy of the affected quad.  This probe keeps the production VAO and
target vertex offset fixed, then grows the draw range backwards in triangle-
aligned steps.  It also compares the original building boundary with the
rooftop-detail-extended batch.

Example::

    python -m tools.probe_facade_draw_threshold --iterations 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import moderngl  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from simulator.app import SimulatorApp  # noqa: E402
from simulator.config import SimulationConfig  # noqa: E402
from simulator.passes.scene import VERTEX_LAYOUT  # noqa: E402
from simulator.renderer import TERRAIN_UNIT  # noqa: E402
from simulator.scenario import DEFAULT_SCENARIO_PATH  # noqa: E402
from simulator.validation.views import load_visual_regression_suite  # noqa: E402
from tools.probe_facade_batch_ladder import (  # noqa: E402
    _bind_opaque_textures,
    _zero_light_counts,
)
from tools.probe_facade_pass_ladder import (  # noqa: E402
    REGION_HALF_OPEN,
    _measure_draws,
    locate_residual_quad,
)
from tools.probe_facade_shader_reproducer import _context_metadata  # noqa: E402


ORIGINAL_BUILDING_VERTICES = 103_866
FULL_BUILDING_VERTICES = 115_446
PRECEDING_VERTEX_STEPS = (0, 258, 1_026, 4_098, 16_386, 65_538)
ROOFTOP_EXTRA_STEPS = (
    3,
    96,
    384,
    1_446,
    2_895,
    5_790,
    8_685,
    10_131,
    10_854,
    11_217,
    11_307,
    11_310,
    11_313,
    11_316,
    11_319,
    11_322,
    11_325,
    11_328,
    11_331,
    11_334,
    11_337,
    11_352,
    11_376,
    11_388,
    11_394,
    11_397,
    11_400,
    11_490,
    11_535,
    11_559,
    11_571,
    11_577,
)


@dataclass(frozen=True, slots=True)
class DrawSpec:
    name: str
    first: int
    vertices: int
    compact: bool = False

    @property
    def end(self) -> int:
        return self.first + self.vertices

    @property
    def preceding_vertices(self) -> int:
        return max(self.vertices - 6, 0) if self.end == TARGET_END else -1


TARGET_FIRST = 103_764
TARGET_END = 103_770
VERTEX_BYTES = 10 * np.dtype(np.float32).itemsize


def draw_specs(
    target_first: int = TARGET_FIRST,
    original_vertices: int = ORIGINAL_BUILDING_VERTICES,
    full_vertices: int = FULL_BUILDING_VERTICES,
) -> tuple[DrawSpec, ...]:
    """Return triangle-aligned ranges that all include the affected quad."""

    if target_first % 3 or original_vertices % 3 or full_vertices % 3:
        raise ValueError("building draw boundaries must be triangle-aligned")
    target_end = target_first + 6
    if not target_end <= original_vertices <= full_vertices:
        raise ValueError("target/original/full building boundaries are inconsistent")
    specs = [DrawSpec("compact_quad", 0, 6, compact=True)]
    for preceding in PRECEDING_VERTEX_STEPS:
        if preceding > target_first:
            continue
        first = target_first - preceding
        specs.append(
            DrawSpec(
                "full_vbo_target" if preceding == 0 else f"preceding_{preceding}",
                first,
                preceding + 6,
            )
        )
    if target_first not in PRECEDING_VERTEX_STEPS:
        specs.append(DrawSpec("prefix_to_target", 0, target_end))
    specs.extend(
        (DrawSpec("original_buildings", 0, original_vertices),)
    )
    rooftop_vertices = full_vertices - original_vertices
    for extra in ROOFTOP_EXTRA_STEPS:
        if extra >= rooftop_vertices:
            continue
        specs.append(DrawSpec(f"rooftop_extra_{extra}", 0, original_vertices + extra))
    specs.append(DrawSpec("full_with_rooftops", 0, full_vertices))
    for spec in specs[1:]:
        if spec.first % 3 or spec.vertices % 3 or not (
            spec.first <= target_first and spec.end >= target_end
        ):
            raise RuntimeError(f"invalid threshold draw range: {spec}")
    return tuple(specs)


def _draw_result(
    app: SimulatorApp,
    spec: DrawSpec,
    iterations: int,
    compact_vao: moderngl.VertexArray,
) -> dict[str, Any]:
    building_vao, building_count = app.renderer.scene.vaos[0]
    if spec.compact:
        draw = lambda: compact_vao.render(moderngl.TRIANGLES, vertices=6)
    else:
        if spec.end > building_count:
            raise RuntimeError(f"draw {spec.name} exceeds building VAO")
        draw = lambda: building_vao.render(
            moderngl.TRIANGLES,
            vertices=spec.vertices,
            first=spec.first,
        )
    result = _measure_draws(app.ctx, iterations, draw, depth=True)
    result["draw"] = {
        "first": spec.first,
        "vertices": spec.vertices,
        "end": spec.end,
        "compact_buffer": spec.compact,
    }
    return result


def _replacement_triangle_stage(
    app: SimulatorApp,
    iterations: int,
    *,
    replace_with_previous: bool,
) -> dict[str, Any]:
    """Keep 115,197 processed vertices while changing the last triangle data."""

    boundary_vertices = ORIGINAL_BUILDING_VERTICES + 11_331
    building_buffer = app.renderer.scene.buffers[0]
    last_offset = (boundary_vertices - 3) * VERTEX_BYTES
    original = building_buffer.read(3 * VERTEX_BYTES, last_offset)
    if replace_with_previous:
        previous_offset = (boundary_vertices - 6) * VERTEX_BYTES
        replacement = building_buffer.read(3 * VERTEX_BYTES, previous_offset)
        building_buffer.write(replacement, offset=last_offset)
    building_vao = app.renderer.scene.vaos[0][0]
    try:
        result = _measure_draws(
            app.ctx,
            iterations,
            lambda: building_vao.render(
                moderngl.TRIANGLES, vertices=boundary_vertices, first=0
            ),
            depth=True,
        )
        result["draw"] = {
            "first": 0,
            "vertices": boundary_vertices,
            "end": boundary_vertices,
            "compact_buffer": False,
            "last_triangle_replaced_with_previous": replace_with_previous,
        }
        return result
    finally:
        if replace_with_previous:
            building_buffer.write(original, offset=last_offset)


def _split_draw_stage(
    app: SimulatorApp, iterations: int, split_at: int
) -> dict[str, Any]:
    """Draw the same complete building VAO in two triangle-aligned calls."""

    if split_at % 3 or not TARGET_END <= split_at < FULL_BUILDING_VERTICES:
        raise ValueError("building split must be aligned and follow the target quad")
    building_vao = app.renderer.scene.vaos[0][0]

    def draw() -> None:
        building_vao.render(moderngl.TRIANGLES, vertices=split_at, first=0)
        building_vao.render(
            moderngl.TRIANGLES,
            vertices=FULL_BUILDING_VERTICES - split_at,
            first=split_at,
        )

    result = _measure_draws(app.ctx, iterations, draw, depth=True)
    result["draw"] = {
        "first": 0,
        "vertices": FULL_BUILDING_VERTICES,
        "end": FULL_BUILDING_VERTICES,
        "compact_buffer": False,
        "split_at": split_at,
    }
    return result


def _trigger_triangle_stage(
    app: SimulatorApp, iterations: int, mode: str
) -> dict[str, Any]:
    """Include or omit the exact first-positive rooftop triangle."""

    trigger_first = ORIGINAL_BUILDING_VERTICES + 11_328
    building_vao = app.renderer.scene.vaos[0][0]

    def draw_full_without_trigger() -> None:
        building_vao.render(moderngl.TRIANGLES, vertices=trigger_first, first=0)
        after = trigger_first + 3
        building_vao.render(
            moderngl.TRIANGLES,
            vertices=FULL_BUILDING_VERTICES - after,
            first=after,
        )

    def draw_target_then_trigger() -> None:
        building_vao.render(moderngl.TRIANGLES, vertices=6, first=TARGET_FIRST)
        building_vao.render(moderngl.TRIANGLES, vertices=3, first=trigger_first)

    def draw_trigger_then_target() -> None:
        building_vao.render(moderngl.TRIANGLES, vertices=3, first=trigger_first)
        building_vao.render(moderngl.TRIANGLES, vertices=6, first=TARGET_FIRST)

    draws = {
        "full_without_trigger": draw_full_without_trigger,
        "target_then_trigger": draw_target_then_trigger,
        "trigger_then_target": draw_trigger_then_target,
    }
    if mode not in draws:
        raise ValueError(f"unknown trigger triangle comparison: {mode}")
    result = _measure_draws(app.ctx, iterations, draws[mode], depth=True)
    result["draw"] = {
        "first": 0,
        "vertices": FULL_BUILDING_VERTICES if mode == "full_without_trigger" else 9,
        "end": FULL_BUILDING_VERTICES,
        "compact_buffer": False,
        "trigger_triangle_first": trigger_first,
        "mode": mode,
    }
    return result


def _compact_trigger_order_stage(
    app: SimulatorApp,
    iterations: int,
    *,
    mode: str,
) -> dict[str, Any]:
    """Render the trigger and target from a compact nine-vertex buffer."""

    trigger_offset = (ORIGINAL_BUILDING_VERTICES + 11_328) * VERTEX_BYTES
    building_buffer = app.renderer.scene.buffers[0]
    target_raw = building_buffer.read(6 * VERTEX_BYTES, TARGET_FIRST * VERTEX_BYTES)
    trigger_raw = building_buffer.read(3 * VERTEX_BYTES, trigger_offset)
    buffer = app.ctx.buffer(trigger_raw + target_raw)
    vao = app.ctx.vertex_array(
        app.renderer.scene.program, [(buffer, *VERTEX_LAYOUT)]
    )

    def draw_trigger_then_target() -> None:
        vao.render(moderngl.TRIANGLES, vertices=3, first=0)
        vao.render(moderngl.TRIANGLES, vertices=6, first=3)

    def draw_target_then_trigger() -> None:
        vao.render(moderngl.TRIANGLES, vertices=6, first=3)
        vao.render(moderngl.TRIANGLES, vertices=3, first=0)

    draws = {
        "target_only": lambda: vao.render(
            moderngl.TRIANGLES, vertices=6, first=3
        ),
        "target_then_trigger": draw_target_then_trigger,
        "trigger_then_target": draw_trigger_then_target,
    }
    if mode not in draws:
        raise ValueError(f"unknown compact order comparison: {mode}")
    try:
        draw = draws[mode]
        result = _measure_draws(app.ctx, iterations, draw, depth=True)
        result["draw"] = {
            "first": 0,
            "vertices": 9,
            "end": 9,
            "compact_buffer": True,
            "mode": mode,
        }
        return result
    finally:
        vao.release()
        buffer.release()


def _select_specs(
    specs: tuple[DrawSpec, ...],
    focus_rooftop: bool,
    focus_final_boundary: bool = False,
) -> tuple[DrawSpec, ...]:
    if focus_final_boundary:
        return tuple(
            spec
            for spec in specs
            if spec.name in {"rooftop_extra_11217", "full_with_rooftops"}
            or (
                spec.name.startswith("rooftop_extra_")
                and 11_307 <= int(spec.name.rsplit("_", 1)[1]) <= 11_331
            )
        )
    if not focus_rooftop:
        return specs
    return tuple(
        spec
        for spec in specs
        if spec.name == "original_buildings"
        or spec.name.startswith("rooftop_extra_")
        or spec.name == "full_with_rooftops"
    )


def probe(
    iterations: int,
    *,
    focus_rooftop: bool = False,
    focus_final_boundary: bool = False,
    compare_replacement: bool = False,
    compare_split: bool = False,
    compare_trigger: bool = False,
    compare_compact_order: bool = False,
) -> dict[str, Any]:
    """Measure every draw range in one initialised production context."""

    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    quad = locate_residual_quad()
    if quad.vertex_range != (TARGET_FIRST, TARGET_END):
        raise RuntimeError("residual quad vertex offset changed")
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    compact_buffer: moderngl.Buffer | None = None
    compact_vao: moderngl.VertexArray | None = None
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        load_visual_regression_suite().view("water_reflection").apply(app.camera)
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        _zero_light_counts(app)
        _bind_opaque_textures(app)
        app.renderer.terrain_texture.use(TERRAIN_UNIT)

        program = app.renderer.scene.program
        compact_buffer = app.ctx.buffer(quad.source_vertices.tobytes())
        compact_vao = app.ctx.vertex_array(
            program, [(compact_buffer, *VERTEX_LAYOUT)]
        )
        specs = _select_specs(draw_specs(), focus_rooftop, focus_final_boundary)
        if (
            compare_replacement
            or compare_split
            or compare_trigger
            or compare_compact_order
        ):
            specs = ()
        results: dict[str, dict[str, Any]] = {}
        for spec in specs:
            _zero_light_counts(app)
            _bind_opaque_textures(app)
            results[spec.name] = _draw_result(
                app, spec, iterations, compact_vao
            )
        if compare_replacement:
            for replaced in (False, True):
                name = "boundary_original_data" if not replaced else "boundary_replaced_data"
                results[name] = _replacement_triangle_stage(
                    app, iterations, replace_with_previous=replaced
                )
        if compare_split:
            results["full_single_draw"] = _draw_result(
                app,
                DrawSpec("full_single_draw", 0, FULL_BUILDING_VERTICES),
                iterations,
                compact_vao,
            )
            results["split_before_trigger"] = _split_draw_stage(
                app, iterations, ORIGINAL_BUILDING_VERTICES + 11_328
            )
            results["split_at_original_boundary"] = _split_draw_stage(
                app, iterations, ORIGINAL_BUILDING_VERTICES
            )
        if compare_trigger:
            results["full_single_draw"] = _draw_result(
                app,
                DrawSpec("full_single_draw", 0, FULL_BUILDING_VERTICES),
                iterations,
                compact_vao,
            )
            for mode in (
                "full_without_trigger",
                "target_then_trigger",
                "trigger_then_target",
            ):
                results[mode] = _trigger_triangle_stage(app, iterations, mode)
        if compare_compact_order:
            for mode in (
                "target_only",
                "target_then_trigger",
                "trigger_then_target",
            ):
                results[f"compact_{mode}"] = _compact_trigger_order_stage(
                    app, iterations, mode=mode
                )
        varying = [
            name for name, result in results.items() if not result["bit_deterministic"]
        ]
        return {
            "schema_version": 1,
            "probe": "facade_draw_threshold_v0_14",
            "iterations_per_stage": iterations,
            "measurement_scope": (
                "final_rooftop_boundary"
                if focus_final_boundary
                else "rooftop_boundary"
                if focus_rooftop
                else "all"
            ),
            "region_gl": list(REGION_HALF_OPEN),
            "gpu": _context_metadata(app.ctx),
            "target_vertex_range": [TARGET_FIRST, TARGET_END],
            "boundaries": {
                "original_building_vertices": ORIGINAL_BUILDING_VERTICES,
                "full_building_vertices": FULL_BUILDING_VERTICES,
            },
            "stages": results,
            "verdict": {
                "first_varying_stage": varying[0] if varying else None,
                "varying_stages": varying,
                "full_vbo_offset_is_sufficient": (
                    not results["full_vbo_target"]["bit_deterministic"]
                    if "full_vbo_target" in results
                    else None
                ),
                "compact_quad_is_deterministic": (
                    results["compact_quad"]["bit_deterministic"]
                    if "compact_quad" in results
                    else None
                ),
            },
        }
    finally:
        if compact_vao is not None:
            compact_vao.release()
        if compact_buffer is not None:
            compact_buffer.release()
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4096)
    parser.add_argument(
        "--focus-rooftop",
        action="store_true",
        help="measure only the original-to-rooftop tail boundary",
    )
    parser.add_argument(
        "--compare-compact-order",
        action="store_true",
        help="compare both draw orders in a compact nine-vertex buffer",
    )
    parser.add_argument(
        "--compare-trigger",
        action="store_true",
        help="omit or isolate the exact first-positive rooftop triangle",
    )
    parser.add_argument(
        "--compare-split",
        action="store_true",
        help="compare one building draw with two safe split boundaries",
    )
    parser.add_argument(
        "--compare-replacement",
        action="store_true",
        help="compare the first-positive last triangle with a previous-triangle copy",
    )
    parser.add_argument(
        "--focus-final-boundary",
        action="store_true",
        help="measure only the final 183-vertex rooftop threshold interval",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        report = probe(
            arguments.iterations,
            focus_rooftop=arguments.focus_rooftop,
            focus_final_boundary=arguments.focus_final_boundary,
            compare_replacement=arguments.compare_replacement,
            compare_split=arguments.compare_split,
            compare_trigger=arguments.compare_trigger,
            compare_compact_order=arguments.compare_compact_order,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"{report['gpu']['GL_VENDOR']} / {report['gpu']['GL_RENDERER']}")
    print(f"target_vertices={report['target_vertex_range']}")
    for stage, result in report["stages"].items():
        state = "stable" if result["bit_deterministic"] else "VARYING"
        draw = result["draw"]
        print(
            f"{stage:24s} {state:7s}  first={draw['first']:6d}  "
            f"vertices={draw['vertices']:6d}  different={result['differing_iterations']}"
        )
    offset_verdict = report["verdict"]["full_vbo_offset_is_sufficient"]
    if offset_verdict is not None:
        print(
            "verdict: full-vbo target offset "
            + ("is sufficient" if offset_verdict else "not observed")
        )
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
