"""Find which render target and which pass a frame-to-frame difference comes from.

A capture that is not bit-reproducible tells you only that something moved.
This repeats one representative view with time frozen and reads every
intermediate target, so the first target that disagrees with its own earlier
output names the stage; then `--disable` removes one main-pass draw at a time,
so the configuration where the difference stops appearing names the pass.

Freezing time is the part that matters. With a live timestep the water surface
animates and every target drifts legitimately, which buries the thing being
looked for. Passing ``frame_dt_s = 0`` holds the scene still so any remaining
difference is the renderer disagreeing with itself.

Row numbers here are **OpenGL rows, counted from the bottom**, because that is
how the targets come back and flipping them would misplace the depth against
the colour. The capture tools report image rows counted from the top, so the
same pixels appear at ``height - 1 - row`` there. Mixing the two silently
points any follow-up analysis at the wrong geometry.

Example::

    python -m tools.probe_render_determinism --view water_reflection
    python -m tools.probe_render_determinism --view water_reflection --disable scene
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np

from simulator import shaders
from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)


#: Main-pass draws that can be removed without the renderer failing. The
#: reflection pre-pass is driven separately and always runs.
DISABLEABLE = (
    "sky",
    "land",
    "scene",
    "water",
    "ambient_occlusion",
    "haze",
    "particles",
)

# The scene-owned residual isolated in V0-7. Coordinates are OpenGL rows,
# counted from the bottom, and half-open like a NumPy slice.
DEFAULT_REGION = (598, 380, 600, 384)

# Each expression replaces only the final facade output. The rest of the
# shipped shader still runs, which keeps its interpolants, derivatives,
# uniforms, and control flow available to the probe. This is deliberately a
# diagnostic source transform rather than an edit to scene.frag.
FINAL_SCENE_TERMS = {
    "coordinates": "vec4(fract(grid), aa)",
    "occupancy": "vec4(occupancy_sample, occupied, room_dimmer * floor_blackout, 1.0)",
    "window_mask": "vec4(pane, blinds, curtains, 1.0)",
    "base_normal": "vec4(world_normal * .5 + .5, 1.0)",
    "normal": "vec4(n * .5 + .5, 1.0)",
    "view_direction": "vec4(normalize(camera_position - world_position) * .5 + .5, 1.0)",
    "world_position": "vec4(world_position / 4096.0 + .5, 1.0)",
    "reflectance": "reflectance",
    "facade": "vec4(facade, 1.0)",
    "facade_detail": "vec4(facade_detail)",
    "dynamic_count": "vec4(float(dynamic_light_count) / 8.0)",
    "emission": "vec4(emission * landmark_emission_scale, 1.0)",
    "reflected": "vec4(reflected_radiance(n, facade, reflectance), 1.0)",
}

EARLY_SCENE_TERMS = {
    "raw_normal": "vec4(world_normal * .5 + .5, 1.0)",
    "raw_position": "vec4(world_position / 4096.0 + .5, 1.0)",
    "attributes": "vec4(surface / 20.0, facade_style / 20.0, 0.0, 1.0)",
}

_FINAL_FACADE_OUTPUT = """    frag_color = vec4(
        reflected_radiance(n, facade, reflectance)
        + emission * landmark_emission_scale, 1.0
    );"""

CHECKPOINT_SCENE_TERMS = {
    "initial_normal": (
        "    vec3 n = normalize(world_normal);",
        "vec4(n * .5 + .5, 1.0)",
    ),
    "material_kind": (
        "    vec2 relief = material_relief[material];",
        "vec4(pattern.x / 10.0, surface / 20.0, facade_style / 20.0, 1.0)",
    ),
    "pre_grid_normal": (
        "    vec2 grid = surface_uv / vec2(bay_width, floor_height);",
        "vec4(n * .5 + .5, 1.0)",
    ),
    "post_scan_normal": (
        "    reflectance.x = mix(\n        max(reflectance.x, .42), .105, glass_coverage\n    );",
        "vec4(n * .5 + .5, 1.0)",
    ),
    "after_reflectance_normal": (
        "    float landmark_emission_scale = 1.0;",
        "vec4(n * .5 + .5, 1.0)",
    ),
    "pre_final_normal": (
        _FINAL_FACADE_OUTPUT,
        "vec4(n * .5 + .5, 1.0)",
    ),
}

RADIANCE_CHECKPOINT_TERMS = {
    "radiance_input_normal": (
        "    vec3 diffuse_albedo = albedo * (1.0 - metallic);\n    vec3 view_direction = normalize(camera_position - world_position);",
        "n * .5 + .5",
    ),
    "radiance_input_view": (
        "    vec3 diffuse_albedo = albedo * (1.0 - metallic);\n    vec3 view_direction = normalize(camera_position - world_position);",
        "view_direction * .5 + .5",
    ),
    "radiance_ambient": (
        "    vec3 result = diffuse_albedo * ambient_irradiance_w_m2\n                * reflectance.z / PI;",
        "result",
    ),
    "radiance_n_dot_v": (
        "    float n_dot_v = max(dot(n, view_direction), 0.0);",
        "vec3(n_dot_v)",
    ),
    "radiance_grazing": (
        "    vec3 grazing_fresnel = normal_reflectance\n        + (vec3(1.0) - normal_reflectance) * pow(1.0 - n_dot_v, 5.0);",
        "grazing_fresnel",
    ),
    "radiance_sky_visibility": (
        "    float sky_visibility = mix(.18, 1.0, clamp(n.y * .5 + .5, 0.0, 1.0));",
        "vec3(sky_visibility)",
    ),
    "radiance_environment": (
        "    result += grazing_fresnel * ambient_irradiance_w_m2 / PI\n            * sky_visibility * (1.0 - roughness * .72);",
        "result",
    ),
    "radiance_after_static": (
        "        result += static_light_color * irradiance * brdf * n_dot_l;\n    }",
        "result",
    ),
    "radiance_after_dynamic": (
        "        result += dynamic_light_color[i] * irradiance * brdf * n_dot_l;\n    }",
        "result",
    ),
}

SCENE_TERMS = {
    **FINAL_SCENE_TERMS,
    **EARLY_SCENE_TERMS,
    **{name: expression for name, (_, expression) in CHECKPOINT_SCENE_TERMS.items()},
    **{name: expression for name, (_, expression) in RADIANCE_CHECKPOINT_TERMS.items()},
}

def _instrument_scene_fragment(source: str, term: str) -> str:
    """Return a temporary scene fragment shader exposing one facade term."""

    expression = SCENE_TERMS[term]
    if term in EARLY_SCENE_TERMS:
        main = "void main() {"
        if source.count(main) != 1:
            raise RuntimeError("scene.frag main entry no longer has the expected form")
        # The uniform-dependent guard prevents the compiler from discarding
        # the real shader and its uniforms, while always selecting the raw
        # varying output for the event's non-negative render time.
        return source.replace(
            main,
            (
                f"{main}\n"
                f"    if (time_s > -1e20) {{\n"
                f"        frag_color = {expression};\n"
                f"        return;\n"
                f"    }}"
            ),
            1,
        )
    if term in CHECKPOINT_SCENE_TERMS:
        anchor, expression = CHECKPOINT_SCENE_TERMS[term]
        if source.count(anchor) != 1:
            raise RuntimeError(
                f"scene.frag checkpoint for {term} no longer has the expected form"
            )
        return source.replace(
            anchor,
            (
                f"{anchor}\n"
                f"    if (time_s > -1e20) {{\n"
                f"        frag_color = {expression};\n"
                f"        return;\n"
                f"    }}"
            ),
            1,
        )
    if term in RADIANCE_CHECKPOINT_TERMS:
        anchor, expression = RADIANCE_CHECKPOINT_TERMS[term]
        if source.count(anchor) != 1:
            raise RuntimeError(
                f"scene.frag radiance checkpoint for {term} no longer has the expected form"
            )
        instrumented = source.replace(
            anchor,
            (
                f"{anchor}\n"
                f"    if (time_s > -1e20) return {expression};"
            ),
            1,
        )
        return instrumented.replace(
            _FINAL_FACADE_OUTPUT,
            "    frag_color = vec4(reflected_radiance(n, facade, reflectance), 1.0);",
            1,
        )
    if source.count(_FINAL_FACADE_OUTPUT) != 1:
        raise RuntimeError("scene.frag final facade output no longer has the expected form")
    return source.replace(
        _FINAL_FACADE_OUTPUT,
        f"    frag_color = {expression};",
        1,
    )


def _diagnostic_shader_source(original_source, name: str, *, term: str) -> str:
    source = original_source(name)
    if name == "scene.frag":
        return _instrument_scene_fragment(source, term)
    return source


def _read_colour(texture: Any) -> np.ndarray:
    width, height = texture.size
    raw = np.frombuffer(texture.read(), dtype=np.float16)
    return raw.reshape(height, width, texture.components).astype(np.float32)


def _read_depth(texture: Any) -> np.ndarray:
    width, height = texture.size
    raw = np.frombuffer(texture.read(), dtype=np.float32)
    return raw.reshape(height, width, 1)


def _targets(
    renderer: Any,
    region: tuple[int, int, int, int] | None = None,
) -> dict[str, np.ndarray]:
    targets = renderer.targets
    frames = {
        "airlight": _read_colour(targets.airlight_texture),
        "ambient_occlusion": _read_colour(targets.ambient_occlusion_texture),
        "scene_depth": _read_depth(targets.scene_depth_texture),
        "reflection_depth": _read_depth(targets.reflection_depth),
        "reflection": _read_colour(targets.reflection_texture),
        "hdr": _read_colour(targets.hdr_texture),
    }
    if region is None:
        return frames
    left, bottom, right, top = region
    return {
        name: frame[bottom:top, left:right]
        for name, frame in frames.items()
    }


def _signature(reference: np.ndarray, candidate: np.ndarray) -> str | None:
    channels = min(3, candidate.shape[2])
    delta = np.abs(candidate[:, :, :channels] - reference[:, :, :channels])
    mask = delta.max(axis=2) > 0
    if not mask.any():
        return None
    rows, columns = np.nonzero(mask)
    return (
        f"{int(mask.sum())}px max{delta.max():.3e} "
        f"rows{rows.min()}-{rows.max()} cols{columns.min()}-{columns.max()}"
    )


def probe(
    view_id: str,
    iterations: int,
    disable: str | None,
    *,
    scene_term: str | None = None,
    region: tuple[int, int, int, int] | None = None,
    static_light_count: int | None = None,
    refresh_reflection: bool = True,
) -> dict[str, Any]:
    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    view = next(item for item in suite.views if item.view_id == view_id)
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    original_source = shaders.source
    if scene_term is not None:
        shaders.source = partial(
            _diagnostic_shader_source,
            original_source,
            term=scene_term,
        )
    try:
        app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    finally:
        # Never let a failed compile or application initialisation leak the
        # diagnostic source transform into another probe in this process.
        shaders.source = original_source
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        app.renderer.post.set_mode(DisplayMode(suite.display_mode))
        view.apply(app.camera)
        renderer = app.renderer
        if disable:
            getattr(renderer, disable).draw = lambda *args, **kwargs: None
        if scene_term is not None:
            # Preserve the instrumented scene colour in the HDR attachment.
            # These later passes are not part of the quantity being measured.
            for pass_name in (
                "water",
                "ambient_occlusion",
                "haze",
                "particles",
            ):
                getattr(renderer, pass_name).draw = lambda *args, **kwargs: None
        if static_light_count is not None:
            update_static_lights = renderer._update_static_lights

            def update_static_lights_with_override(camera) -> None:
                update_static_lights(camera)
                renderer.scene.program["static_light_count"] = static_light_count

            renderer._update_static_lights = update_static_lights_with_override

        first: dict[str, np.ndarray] | None = None
        signatures: dict[str, dict[str, int]] = {}
        differing: dict[str, int] = {}
        scene_states: dict[bytes, dict[str, Any]] = {}
        for _ in range(iterations):
            if refresh_reflection:
                # Force the pre-pass every iteration; a static camera would
                # otherwise refresh it once and reuse the result thereafter.
                renderer.reflection_ready = False
                renderer.reflection_accumulator_s = 1e9
                renderer.reflection_sky_accumulator_s = 1e9
            renderer.render(app.world, app.camera, app.celestial, 0.0, None)
            app.ctx.finish()
            frames = _targets(renderer, region)
            if scene_term is not None and region is not None:
                rgb = frames["hdr"][:, :, :3]
                key = rgb.tobytes()
                state = scene_states.get(key)
                if state is None:
                    state = {
                        "count": 0,
                        "rgb_min": rgb.min(axis=(0, 1)).tolist(),
                        "rgb_max": rgb.max(axis=(0, 1)).tolist(),
                    }
                    scene_states[key] = state
                state["count"] += 1
            if first is None:
                first = frames
                continue
            for name, frame in frames.items():
                signature = _signature(first[name], frame)
                if signature is None:
                    continue
                differing[name] = differing.get(name, 0) + 1
                bucket = signatures.setdefault(name, {})
                bucket[signature] = bucket.get(signature, 0) + 1
        return {
            "view_id": view_id,
            "iterations": iterations,
            "comparisons": max(iterations - 1, 0),
            "disabled_pass": disable,
            "scene_term": scene_term,
            "region_gl": list(region) if region is not None else None,
            "static_light_count": static_light_count,
            "refresh_reflection": refresh_reflection,
            "differing_counts": differing,
            "signatures": signatures,
            "scene_states": sorted(
                scene_states.values(),
                key=lambda state: -state["count"],
            ),
        }
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="water_reflection")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--disable",
        choices=DISABLEABLE,
        default=None,
        help="remove one main-pass draw to see whether the difference goes with it",
    )
    parser.add_argument(
        "--reuse-reflection",
        action="store_true",
        help="render the planar reflection once, then reuse it on frozen frames",
    )
    parser.add_argument(
        "--static-light-count",
        type=int,
        choices=range(5),
        default=None,
        help="override the scene shader's cumulative static-light count (0..4)",
    )
    parser.add_argument(
        "--scene-term",
        choices=tuple(SCENE_TERMS),
        default=None,
        help="temporarily expose one facade intermediate in the scene colour",
    )
    parser.add_argument(
        "--region",
        metavar="LEFT,BOTTOM,RIGHT,TOP",
        default=None,
        help=(
            "compare only this half-open OpenGL pixel rectangle; defaults to "
            "the known 2x4 scene residual when --scene-term is used"
        ),
    )
    arguments = parser.parse_args()
    region = None
    if arguments.region is not None:
        try:
            region = tuple(int(value) for value in arguments.region.split(","))
        except ValueError as error:
            parser.error(f"--region must contain four integers: {error}")
        if len(region) != 4:
            parser.error("--region must contain LEFT,BOTTOM,RIGHT,TOP")
        left, bottom, right, top = region
        if min(left, bottom) < 0 or right <= left or top <= bottom:
            parser.error("--region must be a non-empty rectangle with non-negative bounds")
    elif arguments.scene_term is not None:
        region = DEFAULT_REGION
    report = probe(
        arguments.view,
        arguments.iterations,
        arguments.disable,
        scene_term=arguments.scene_term,
        region=region,
        static_light_count=arguments.static_light_count,
        refresh_reflection=not arguments.reuse_reflection,
    )
    disabled = report["disabled_pass"] or "none"
    term = report["scene_term"] or "final"
    reflection = "refresh" if report["refresh_reflection"] else "reuse"
    print(
        f"{report['view_id']}  disabled={disabled}  term={term}  "
        f"reflection={reflection}  comparisons={report['comparisons']}"
    )
    if report["region_gl"] is not None:
        print(f"  region_gl={report['region_gl']}")
    if report["static_light_count"] is not None:
        print(f"  static_light_count={report['static_light_count']}")
    for index, state in enumerate(report["scene_states"], start=1):
        print(
            f"  state{index}: {state['count']} frames  "
            f"rgb_min={np.asarray(state['rgb_min'])}  "
            f"rgb_max={np.asarray(state['rgb_max'])}"
        )
    if not report["differing_counts"]:
        print("  every target bit-identical")
        return
    for name, count in sorted(
        report["differing_counts"].items(), key=lambda item: -item[1]
    ):
        print(f"  {name}: {count} differing")
        for signature, hits in sorted(
            report["signatures"][name].items(), key=lambda item: -item[1]
        ):
            print(f"      {hits:3d}x  {signature}")


if __name__ == "__main__":
    main()
