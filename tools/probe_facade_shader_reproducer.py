"""Run the residual facade maths in an independent 2x4 RGBA16F target.

V0-10 found an eight-pixel, optimisation-sensitive residual in ``scene.frag``
on one Intel driver.  This probe removes the simulator, city mesh, textures,
depth buffer, and every other render pass while retaining representative input
ranges captured from those exact pixels.  It then adds the shading operations
back in stages and checks every half-float output bit over repeated draws.

Example::

    python -m tools.probe_facade_shader_reproducer --iterations 4096
    python -m tools.probe_facade_shader_reproducer --iterations 4096 \
        --output docs/validation/render_determinism_v0/standalone_report.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import moderngl  # noqa: E402
import pygame  # noqa: E402


TARGET_SIZE = (2, 4)
STAGES = (
    "raw_normal",
    "view_direction",
    "n_dot_v",
    "grazing_fresnel",
    "environment_inline",
    "environment_helper",
    "full_helper_zero_lights",
    "final_facade",
)

# Bounds observed by the runtime-only V0-10 instrumentation in the known GL
# rectangle (598, 380)-(600, 384). Position was encoded as p / 4096 + .5;
# normal as n * .5 + .5. Values below are decoded and rounded well inside one
# fp16 output interval, rather than presented as new scene measurements.
CAPTURED_INPUT_BOUNDS = {
    "world_normal": [[-0.98124694, 0.0, -0.19287110]] * 2,
    "world_position_m": [
        [814.0, 20.0, -749.0],
        [814.0, 26.0, -746.0],
    ],
    "reflectance": [
        [0.36328125, 0.0, 0.84619141, 0.0],
        [0.41992188, 0.0, 0.85937500, 0.0],
    ],
    "facade_linear_rgb": [
        [0.15295410, 0.17248535, 0.20336914],
        [0.18359375, 0.20397949, 0.23474121],
    ],
}

_VERTEX_SHADER = """#version 330
const vec2 positions[3] = vec2[3](
    vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0)
);
void main() {
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}
"""

_FRAGMENT_PREAMBLE = """#version 330
uniform vec3 camera_position;
uniform float ambient_irradiance_w_m2;
uniform int static_light_count;
uniform vec3 static_light_position[4];
uniform vec3 static_light_color;
uniform float static_light_power_w;
uniform int dynamic_light_count;
uniform vec3 dynamic_light_position[8];
uniform vec3 dynamic_light_color[8];
uniform float dynamic_light_power_w[8];
out vec4 frag_color;
const float PI = 3.14159265359;
const float DIELECTRIC_NORMAL_REFLECTANCE = 0.04;

float pixel_index() {
    return (gl_FragCoord.x - .5) + (gl_FragCoord.y - .5) * 2.0;
}
float pixel_mix() { return pixel_index() / 7.0; }
vec3 captured_normal() {
    return vec3(-.98124694, 0.0, -.19287110);
}
vec3 captured_position() {
    return mix(vec3(814.0, 20.0, -749.0),
               vec3(814.0, 26.0, -746.0), pixel_mix());
}
vec4 captured_reflectance() {
    return mix(vec4(.36328125, 0.0, .84619141, 0.0),
               vec4(.41992188, 0.0, .85937500, 0.0), pixel_mix());
}
vec3 captured_facade() {
    return mix(vec3(.15295410, .17248535, .20336914),
               vec3(.18359375, .20397949, .23474121), pixel_mix());
}

vec3 environment_radiance(
    vec3 n, vec3 albedo, vec4 reflectance, vec3 world_position
) {
    float roughness = clamp(reflectance.x, .03, 1.0);
    float metallic = reflectance.y;
    vec3 normal_reflectance = mix(
        vec3(DIELECTRIC_NORMAL_REFLECTANCE), albedo, metallic
    );
    vec3 diffuse_albedo = albedo * (1.0 - metallic);
    vec3 view_direction = normalize(camera_position - world_position);
    vec3 result = diffuse_albedo * ambient_irradiance_w_m2
                * reflectance.z / PI;
    float n_dot_v = max(dot(n, view_direction), 0.0);
    vec3 grazing_fresnel = normal_reflectance
        + (vec3(1.0) - normal_reflectance) * pow(1.0 - n_dot_v, 5.0);
    float sky_visibility = mix(.18, 1.0, clamp(n.y * .5 + .5, 0.0, 1.0));
    result += grazing_fresnel * ambient_irradiance_w_m2 / PI
            * sky_visibility * (1.0 - roughness * .72);
    return result;
}
"""

_FULL_HELPER = """
vec3 specular_brdf(
    vec3 n, vec3 view_direction, vec3 light_direction,
    float roughness, vec3 normal_reflectance
) {
    float n_dot_l = max(dot(n, light_direction), 0.0);
    float n_dot_v = max(dot(n, view_direction), 1e-4);
    if (n_dot_l <= 0.0) return vec3(0.0);
    float alpha = roughness * roughness;
    float alpha_squared = alpha * alpha;
    vec3 half_vector = normalize(view_direction + light_direction);
    float n_dot_h = max(dot(n, half_vector), 0.0);
    float v_dot_h = max(dot(view_direction, half_vector), 0.0);
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    float distribution = alpha_squared
                       / max(PI * denominator * denominator, 1e-5);
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float geometry_v = n_dot_v / mix(n_dot_v, 1.0, k);
    float geometry_l = n_dot_l / mix(n_dot_l, 1.0, k);
    vec3 fresnel = normal_reflectance
        + (vec3(1.0) - normal_reflectance) * pow(1.0 - v_dot_h, 5.0);
    return fresnel * distribution * geometry_v * geometry_l
         / max(4.0 * n_dot_v * n_dot_l, 1e-5);
}

vec3 full_reflected_radiance(
    vec3 n, vec3 albedo, vec4 reflectance, vec3 world_position
) {
    float roughness = clamp(reflectance.x, .03, 1.0);
    float metallic = reflectance.y;
    vec3 normal_reflectance = mix(
        vec3(DIELECTRIC_NORMAL_REFLECTANCE), albedo, metallic
    );
    vec3 diffuse_albedo = albedo * (1.0 - metallic);
    vec3 view_direction = normalize(camera_position - world_position);
    vec3 result = diffuse_albedo * ambient_irradiance_w_m2
                * reflectance.z / PI;
    float n_dot_v = max(dot(n, view_direction), 0.0);
    vec3 grazing_fresnel = normal_reflectance
        + (vec3(1.0) - normal_reflectance) * pow(1.0 - n_dot_v, 5.0);
    float sky_visibility = mix(.18, 1.0, clamp(n.y * .5 + .5, 0.0, 1.0));
    result += grazing_fresnel * ambient_irradiance_w_m2 / PI
            * sky_visibility * (1.0 - roughness * .72);
    for (int i = 0; i < 4; ++i) {
        if (i >= static_light_count) break;
        vec3 displacement = static_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), .5);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float down_lobe = pow(max(light_direction.y, 0.0), 3.0);
        vec3 irradiance = vec3(
            static_light_power_w * 2.0 * down_lobe / (PI * distance_squared)
        );
        float n_dot_l = max(dot(n, light_direction), 0.0);
        vec3 brdf = diffuse_albedo / PI + specular_brdf(
            n, view_direction, light_direction, roughness, normal_reflectance
        );
        result += static_light_color * irradiance * brdf * n_dot_l;
    }
    for (int i = 0; i < 8; ++i) {
        if (i >= dynamic_light_count) break;
        vec3 displacement = dynamic_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), 1.0);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        vec3 irradiance = vec3(
            dynamic_light_power_w[i] / (4.0 * PI * distance_squared)
        );
        float n_dot_l = max(dot(n, light_direction), 0.0);
        vec3 brdf = diffuse_albedo / PI + specular_brdf(
            n, view_direction, light_direction, roughness, normal_reflectance
        );
        result += dynamic_light_color[i] * irradiance * brdf * n_dot_l;
    }
    return result;
}
"""

_STAGE_BODIES = {
    "raw_normal": """frag_color = vec4(raw_normal * .5 + .5, 1.0);""",
    "view_direction": """
        vec3 view_direction = normalize(camera_position - world_position);
        frag_color = vec4(view_direction * .5 + .5, 1.0);""",
    "n_dot_v": """
        vec3 view_direction = normalize(camera_position - world_position);
        float n_dot_v = max(dot(n, view_direction), 0.0);
        frag_color = vec4(vec3(n_dot_v), 1.0);""",
    "grazing_fresnel": """
        vec3 view_direction = normalize(camera_position - world_position);
        float n_dot_v = max(dot(n, view_direction), 0.0);
        vec3 grazing_fresnel = vec3(DIELECTRIC_NORMAL_REFLECTANCE)
            + vec3(1.0 - DIELECTRIC_NORMAL_REFLECTANCE)
            * pow(1.0 - n_dot_v, 5.0);
        frag_color = vec4(grazing_fresnel, 1.0);""",
    "environment_inline": """
        float roughness = clamp(reflectance.x, .03, 1.0);
        vec3 diffuse_albedo = albedo * (1.0 - reflectance.y);
        vec3 view_direction = normalize(camera_position - world_position);
        vec3 result = diffuse_albedo * ambient_irradiance_w_m2
                    * reflectance.z / PI;
        float n_dot_v = max(dot(n, view_direction), 0.0);
        vec3 grazing_fresnel = vec3(DIELECTRIC_NORMAL_REFLECTANCE)
            + vec3(1.0 - DIELECTRIC_NORMAL_REFLECTANCE)
            * pow(1.0 - n_dot_v, 5.0);
        float sky_visibility = mix(.18, 1.0, clamp(n.y * .5 + .5, 0.0, 1.0));
        result += grazing_fresnel * ambient_irradiance_w_m2 / PI
                * sky_visibility * (1.0 - roughness * .72);
        frag_color = vec4(result, 1.0);""",
    "environment_helper": """
        frag_color = vec4(environment_radiance(
            n, albedo, reflectance, world_position
        ), 1.0);""",
    "full_helper_zero_lights": """
        frag_color = vec4(full_reflected_radiance(
            n, albedo, reflectance, world_position
        ), 1.0);""",
    "final_facade": """
        vec3 emission = mix(vec3(.0003, .0002, .0001),
                            vec3(.0012, .0007, .0003), pixel_mix());
        frag_color = vec4(full_reflected_radiance(
            n, albedo, reflectance, world_position
        ) + emission, 1.0);""",
}


def fragment_source(stage: str) -> str:
    """Build one cumulative standalone shader, failing closed on bad stages."""

    if stage not in STAGES:
        raise ValueError(f"unknown standalone facade stage: {stage}")
    helpers = _FULL_HELPER if stage in {
        "full_helper_zero_lights",
        "final_facade",
    } else ""
    return (
        _FRAGMENT_PREAMBLE
        + helpers
        + "\nvoid main() {\n"
        + "    vec3 raw_normal = captured_normal();\n"
        + "    vec3 n = normalize(raw_normal);\n"
        + "    vec3 world_position = captured_position();\n"
        + "    vec4 reflectance = captured_reflectance();\n"
        + "    vec3 albedo = captured_facade();\n"
        + "    "
        + _STAGE_BODIES[stage].strip()
        + "\n}\n"
    )


def _set_uniform(program: moderngl.Program, name: str, value: Any) -> None:
    if name in program:
        program[name].value = value


def _summarize_states(states: Iterable[bytes], iterations: int) -> dict[str, Any]:
    counts = Counter(states)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    most_common = ordered[0][1] if ordered else 0
    return {
        "bit_deterministic": len(counts) <= 1,
        "unique_states": len(counts),
        "differing_iterations": iterations - most_common,
        "states": [
            {"sha256": sha256(raw).hexdigest(), "count": count}
            for raw, count in ordered
        ],
    }


def _run_stage(
    ctx: moderngl.Context, stage: str, iterations: int
) -> dict[str, Any]:
    program = ctx.program(
        vertex_shader=_VERTEX_SHADER,
        fragment_shader=fragment_source(stage),
    )
    vao = ctx.vertex_array(program, [])
    texture = ctx.texture(TARGET_SIZE, components=4, dtype="f2")
    framebuffer = ctx.framebuffer([texture])
    try:
        _set_uniform(program, "camera_position", (-100.0, 12.3056, 260.0))
        _set_uniform(program, "ambient_irradiance_w_m2", 0.012)
        _set_uniform(program, "static_light_count", 0)
        _set_uniform(program, "static_light_color", (1.0, 0.82, 0.62))
        _set_uniform(program, "static_light_power_w", 0.0)
        _set_uniform(program, "dynamic_light_count", 0)
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.CULL_FACE)
        framebuffer.use()
        ctx.viewport = (0, 0, *TARGET_SIZE)
        states: list[bytes] = []
        first_pixels: np.ndarray | None = None
        for _ in range(iterations):
            vao.render(moderngl.TRIANGLES, vertices=3)
            ctx.finish()
            raw = texture.read(alignment=1)
            states.append(raw)
            if first_pixels is None:
                first_pixels = np.frombuffer(raw, dtype=np.float16).reshape(
                    TARGET_SIZE[1], TARGET_SIZE[0], 4
                )
        summary = _summarize_states(states, iterations)
        assert first_pixels is not None
        summary["first_rgb_min"] = (
            first_pixels[:, :, :3].min(axis=(0, 1)).astype(float).tolist()
        )
        summary["first_rgb_max"] = (
            first_pixels[:, :, :3].max(axis=(0, 1)).astype(float).tolist()
        )
        return summary
    finally:
        framebuffer.release()
        texture.release()
        vao.release()
        program.release()


def _context_metadata(ctx: moderngl.Context) -> dict[str, Any]:
    keys = ("GL_VENDOR", "GL_RENDERER", "GL_VERSION", "GL_SHADING_LANGUAGE_VERSION")
    return {key: ctx.info.get(key, "unknown") for key in keys}


def probe(iterations: int) -> dict[str, Any]:
    """Create a hidden GL context and execute every cumulative stage."""

    if iterations < 2:
        raise ValueError("iterations must be at least 2")
    try:
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.set_mode(
            (64, 64), pygame.OPENGL | pygame.HIDDEN, vsync=0
        )
        ctx = moderngl.create_context()
        stage_results = {
            stage: _run_stage(ctx, stage, iterations) for stage in STAGES
        }
        varying = [
            stage
            for stage in STAGES
            if not stage_results[stage]["bit_deterministic"]
        ]
        return {
            "schema_version": 1,
            "probe": "standalone_facade_shader_v0_11",
            "target": {"size": list(TARGET_SIZE), "format": "RGBA16F"},
            "iterations_per_stage": iterations,
            "gpu": _context_metadata(ctx),
            "captured_input_bounds": CAPTURED_INPUT_BOUNDS,
            "zero_light_counts": {"static": 0, "dynamic": 0},
            "stages": stage_results,
            "verdict": {
                "all_stages_bit_deterministic": not varying,
                "first_varying_stage": varying[0] if varying else None,
            },
        }
    finally:
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        report = probe(arguments.iterations)
    except ValueError as error:
        parser.error(str(error))

    gpu = report["gpu"]
    print(f"{gpu['GL_VENDOR']} / {gpu['GL_RENDERER']}")
    print(gpu["GL_VERSION"])
    for stage, result in report["stages"].items():
        state = "stable" if result["bit_deterministic"] else "VARYING"
        print(
            f"{stage:26s} {state:7s}  "
            f"states={result['unique_states']}  "
            f"different={result['differing_iterations']}"
        )
    verdict = report["verdict"]
    if verdict["all_stages_bit_deterministic"]:
        print("verdict: standalone 2x4 path did not reproduce the residual")
    else:
        print(f"verdict: first varying stage={verdict['first_varying_stage']}")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
