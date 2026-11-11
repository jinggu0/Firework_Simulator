"""Evaluate deterministic bridge/road candidates without replacing old views."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pygame

from simulator import shaders
from simulator.app import SimulatorApp
from simulator.bridge_semantics import DEFAULT_BRIDGE_SEMANTICS_PATH, load_bridge_plan_semantics
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.scene import SURFACE_BRIDGE, SURFACE_ROAD, SURFACE_ROOF, SURFACE_WALL, load_scene
from simulator.validation.capture import read_display_sdr
from simulator.validation.subject_visibility import GATES, NEAR_OCCLUDER_M, measure_visibility, visibility_gates
from simulator.validation.view_placement import validate_view_placement
from simulator.validation.views import DEFAULT_VISUAL_VIEWS_PATH, VisualViewError, load_visual_regression_suite
from tools.capture_visual_baselines import _validate_camera_surface
from tools.shimmer_regions import ASSIGNMENT, SemanticReplay


BRIDGE_REGION = (-1145.0, -1250.0, -600.0, -470.0)  # min x/z, max x/z
ROAD_TARGET_XZ = (-1151.4, -363.9)
ROAD_REGION = (-1281.4, -493.9, -1021.4, -233.9)
BASE_CONFIG = SimulationConfig()
PREVIEW_FRAMES = 8


def visibility_fragment(source, kind):
    """Keep geometry/discard behavior, encode per-fragment ownership instead of light."""
    expected = {"scene": 5, "land": 1, "water": 1}
    if kind not in expected or len(ASSIGNMENT.findall(source)) != expected[kind]:
        raise ValueError("subject visibility fragment contract changed")
    if source.count("void main() {") != 1:
        raise ValueError("subject visibility requires a single fragment entry point")
    uniforms = """
uniform vec3 audit_eye;
uniform bool audit_isolated;
"""
    if kind == "scene":
        uniforms += """
uniform float audit_material;
uniform vec4 audit_bounds;
bool audit_subject() {
    vec2 p = world_position.xz;
    return abs(surface - audit_material) < 0.25
        && all(greaterThanEqual(p, audit_bounds.xy))
        && all(lessThanEqual(p, audit_bounds.zw));
}
"""
        begin = "bool subject_hit = audit_subject();\n    if (audit_isolated && !subject_hit) discard;"
        building = f"(abs(surface - {SURFACE_WALL}) < .25 || abs(surface - {SURFACE_ROOF}) < .25)"
    else:
        begin = "bool subject_hit = false;\n    if (audit_isolated) discard;"
        building = "false"
    near = f"(distance(world_position, audit_eye) < {NEAR_OCCLUDER_M})"
    colour = f"frag_color = vec4(float(subject_hit), float(!subject_hit && {near}), float(!subject_hit && !{near} && {building}), 1.0);"
    return ASSIGNMENT.sub(colour, source).replace("void main() {", uniforms + "\nvoid main() {\n    " + begin)


def candidates(reference, terrain, bridge):
    def positioned(name, xz, target, height):
        x, z = map(float, xz)
        return replace(reference, view_id=name, position_eus_m=(x, terrain.height_at(x, z) + height, z),
                       target_eus_m=target, notes="Non-historical subject-visibility engineering candidate.")

    if reference.subject == "bridge_structure":
        sections = bridge.cross_sections_xz_m.mean(axis=1)
        tx, tz = sections[np.argmin(abs(sections[:, 1] + 700.0))]
        target = (float(tx), bridge.local_y_m - .65, float(tz))
        for ix, x in enumerate((-1325., -1175., -1025.)):
            for iz, z in enumerate((-470., -420., -370.)):
                yield positioned(f"bridge_subject_{ix}{iz}", (x, z), target, 30.)
    else:
        tx, tz = ROAD_TARGET_XZ
        target = (tx, terrain.height_at(tx, tz) + .10, tz)
        forward = np.array((.45, -.89))
        forward /= np.linalg.norm(forward)
        side = np.array((-forward[1], forward[0]))
        for ix, distance in enumerate((15., 28., 42.)):
            for iz, offset in enumerate((-3., 0., 3.)):
                xz = np.array(ROAD_TARGET_XZ) - distance * forward + offset * side
                yield positioned(f"road_subject_{ix}{iz}", xz, target, 1.75)


def new_app(display_mode):
    config = replace(BASE_CONFIG, render=replace(BASE_CONFIG.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    app.world.shells.clear()
    app.world.stars.count = 0
    app.renderer.post.set_mode(DisplayMode(display_mode))
    if not app.renderer.post.distortion.is_identity:
        close_app(app)
        raise ValueError("subject-visibility masks require identity lens distortion")
    return app


def close_app(app):
    app.audio_executor.shutdown(wait=True, cancel_futures=True)
    pygame.quit()


def subject_uniforms(subject, eye, *, isolated=False):
    material, bounds = {
        "bridge_structure": (SURFACE_BRIDGE, BRIDGE_REGION),
        "road_ground": (SURFACE_ROAD, ROAD_REGION),
    }[subject]
    return {
        "audit_eye": tuple(float(value) for value in eye),
        "audit_material": material,
        "audit_bounds": bounds,
        "audit_isolated": isolated,
    }


def capture(app, replay, view, frames, *, check_display=False):
    view.apply(app.camera)
    app.renderer.reflection_ready = False
    for _ in range(frames):
        app.renderer.render(app.world, app.camera, app.celestial, 1. / 60., None)
    rgb = read_display_sdr(app.ctx) if check_display else None
    uniforms = subject_uniforms(view.subject, app.camera.position_m)
    visible = replay.capture(uniform_overrides=uniforms)
    isolated = replay.capture(uniform_overrides={**uniforms, "audit_isolated": True})
    if check_display and not np.array_equal(rgb, read_display_sdr(app.ctx)):
        raise RuntimeError("private visibility replay changed the display framebuffer")
    metrics = measure_visibility(visible, isolated)
    return metrics, visible, rgb


def placement(app, view, building_vertices):
    return {
        **_validate_camera_surface(app, view),
        **validate_view_placement(view, building_vertices,
                                  BASE_CONFIG.camera.camera_collision_radius_m),
    }


def measure_candidates(suite, building_vertices):
    app = new_app(suite.display_mode)
    replay = None
    try:
        replay = SemanticReplay(app.renderer, fragment_builder=visibility_fragment)
        bridge = load_bridge_plan_semantics()
        results = {}
        for subject in GATES:
            reference = suite.view(subject)
            views = (reference, *candidates(reference, app.terrain_surface, bridge))
            rows = []
            for view in views:
                print(f"screening {view.view_id}", flush=True)
                row = {"view": asdict(view), "canonical": view is reference}
                try:
                    row["placement"] = placement(app, view, building_vertices)
                except VisualViewError as error:
                    row.update(passed=False, placement_failure=str(error))
                else:
                    row["metrics"], _, _ = capture(app, replay, view, 1)
                    row["gates"] = visibility_gates(subject, row["metrics"])
                    row["passed"] = all(row["gates"].values())
                rows.append(row)
            passing = [i for i in range(1, len(rows)) if rows[i]["passed"]]
            best = min(passing, key=lambda i: (-rows[i]["metrics"]["visible_screen_fraction"], views[i].view_id)) if passing else None
            results[subject] = (rows, reference, views[best] if best is not None else None)
        gl = {key: app.ctx.info[key] for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
        return results, gl, app.renderer.targets.samples
    finally:
        if replay is not None:
            replay.release()
        close_app(app)


def preview(suite, view, output):
    app = new_app(suite.display_mode)
    replay = None
    try:
        replay = SemanticReplay(app.renderer, fragment_builder=visibility_fragment)
        metrics, masks, rgb = capture(app, replay, view, PREVIEW_FRAMES, check_display=True)
        Image.fromarray(rgb).save(output / f"{view.view_id}.png")
        overlay = rgb.astype(np.float32)
        for channel, colour in ((0, (40, 235, 230)), (1, (255, 90, 45))):
            weight = masks[..., channel:channel + 1].astype(np.float32) * .7
            overlay = overlay * (1 - weight) + np.array(colour) * weight
        Image.fromarray(overlay.astype(np.uint8)).save(output / f"{view.view_id}_mask.png")
        return metrics
    finally:
        if replay is not None:
            replay.release()
        close_app(app)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/validation/subject_visibility_v0"))
    args = parser.parse_args()
    suite = load_visual_regression_suite()
    scene = load_scene(suite.verify_scene_asset())
    results, gl, msaa = measure_candidates(suite, scene.building_vertices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects = {}
    for subject, (rows, reference, winner) in results.items():
        subjects[subject] = {"screening": rows, "selected_view": None, "fresh_captures": {}}
        for view in (reference, winner):
            if view is None:
                continue
            print(f"fresh capture {view.view_id}", flush=True)
            metrics = preview(suite, view, args.output_dir)
            gates = visibility_gates(subject, metrics)
            subjects[subject]["fresh_captures"][view.view_id] = {"metrics": metrics, "gates": gates}
            if view is winner and all(gates.values()):
                subjects[subject]["selected_view"] = asdict(view)
    report = {
        "schema_version": 1,
        "canonical_views_sha256": sha256(DEFAULT_VISUAL_VIEWS_PATH.read_bytes()).hexdigest(),
        "scene_asset_sha256": suite.scene_asset_sha256,
        "bridge_plan_sha256": sha256(DEFAULT_BRIDGE_SEMANTICS_PATH.read_bytes()).hexdigest(),
        "shader_sha256": {name: sha256(shaders.source(name).encode()).hexdigest()
                          for name in ("scene.vert", "scene.frag", "land.vert", "land.frag", "water.vert", "water.frag")},
        "subject_bounds_xz": {"bridge_structure": BRIDGE_REGION, "road_ground": ROAD_REGION},
        "gate_order": ["min_screen", "min_centre", "min_visible_projected", "max_near_non_subject"],
        "gates": GATES, "near_occluder_m": NEAR_OCCLUDER_M, "preview_frames": PREVIEW_FRAMES,
        "msaa_samples": msaa, "gl": gl, "subjects": subjects,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print({subject: item["selected_view"] is not None for subject, item in subjects.items()}, flush=True)


if __name__ == "__main__":
    main()
