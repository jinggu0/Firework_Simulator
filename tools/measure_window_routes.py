"""V0-20 diagnostic: suppress distant window light in one draw route at a time."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from simulator.passes.scene import ScenePass
from simulator.scene import SURFACE_WALL
from simulator.validation.capture import load_known_residuals, select_residuals
from simulator.validation.temporal_shimmer import INTERPOLATION_ORDER, pan_source_coordinates
from simulator.validation.views import REPOSITORY_ROOT
from tools.ab_shader_candidate import score_run
from tools.measure_background_shimmer import (
    DISTANT_M, OUTPUT as BACKGROUND_OUTPUT, REGIONS, background_fragment, background_pair_masks, baseline_identity,
)
from tools.measure_subject_shimmer import capture_summary, replay_control, verified_views
from tools.measure_temporal_shimmer import FRAMES, PIXELS_PER_FRAME, render_pan
from tools.shader_candidate import ShaderReplacement, shader_variant


OUTPUT = REPOSITORY_ROOT / "docs/validation/window_routes_v0"
BASELINE = BACKGROUND_OUTPUT / "report.json"
RADIUS = 1
ROUTES = (None, "direct", "reflection")
REPLACEMENTS = (
    ShaderReplacement("scene.frag", "uniform vec3 camera_position;",
                      "uniform vec3 camera_position;\nuniform vec3 audit_real_eye;\n"
                      "uniform bool audit_suppress_windows;"),
    ShaderReplacement(
        "scene.frag",
        "    frag_color = vec4(\n        reflected_radiance(n, facade, reflectance)\n"
        "        + emission * landmark_emission_scale, 1.0\n    );",
        f"    if (audit_suppress_windows && abs(surface - {SURFACE_WALL}) < .25\n"
        f"            && distance(world_position, audit_real_eye) >= {DISTANT_M}) {{\n"
        "        emission = vec3(0.0);\n    }\n"
        "    frag_color = vec4(\n        reflected_radiance(n, facade, reflectance)\n"
        "        + emission * landmark_emission_scale, 1.0\n    );",
    ),
)


@contextmanager
def window_route(route, eye):
    """Use explicit draw entry points; restore hooks even after a failed pan."""
    if route not in ROUTES:
        raise ValueError(f"unknown window route: {route}")
    eye = tuple(float(value) for value in np.asarray(eye, dtype=np.float32))
    original_draw, original_reflection = ScenePass.draw, ScenePass.draw_reflection
    calls = {name: {"draws": 0, "suppressed": 0} for name in ROUTES[1:]}

    def wrap(name, method):
        def draw(scene):
            suppressed = name == route
            scene.program["audit_real_eye"].value = eye
            scene.program["audit_suppress_windows"].value = suppressed
            calls[name]["draws"] += 1
            calls[name]["suppressed"] += int(suppressed)
            return method(scene)
        return draw

    ScenePass.draw = wrap("direct", original_draw)
    ScenePass.draw_reflection = wrap("reflection", original_reflection)
    try:
        yield calls
    finally:
        ScenePass.draw, ScenePass.draw_reflection = original_draw, original_reflection


def receiver_statistics(pan, scored, hdr_scale, labels=None):
    """Keep receiver membership and HDR normalization fixed across interventions."""
    display, hdr = pan["display_luma"], pan["linear_hdr"]
    pairs, (height, width) = len(display) - 1, scored.shape
    if labels is None:
        fixed = np.empty((pairs, int(scored.sum())), dtype=np.uint8)
    else:
        fixed = labels
    counts = np.zeros(len(REGIONS), dtype=np.int64)
    sums = np.zeros((4, len(REGIONS)), dtype=np.float64)
    for pair in range(pairs):
        coords = pan_source_coordinates(pan["matrices"][pair], pan["matrices"][pair + 1], width, height)
        if labels is None:
            masks = background_pair_masks(*pan["semantic_fractions"][pair:pair + 2], coords, RADIUS)
            fixed[pair] = np.argmax(np.stack([masks[name][scored] for name in REGIONS]), axis=0)
        codes = fixed[pair]
        counts += np.bincount(codes, minlength=len(REGIONS))
        for index, values in enumerate((display, hdr)):
            predicted = map_coordinates(values[pair], coords, order=INTERPOLATION_ORDER, mode="nearest")
            actual = values[pair + 1][scored]
            error = actual - predicted[scored]
            if index == 1:
                error = error / hdr_scale
            sums[index] += np.bincount(codes, weights=error ** 2, minlength=len(REGIONS))
            sums[index + 2] += np.bincount(codes, weights=actual, minlength=len(REGIONS))
    if counts.sum() != pairs * scored.sum():
        raise ValueError("receiver samples did not partition the common domain")
    return {
        name: {"samples": int(count), "display_energy": float(sums[0, i]),
               "hdr_energy": float(sums[1, i]),
               "mean_display_code": float(sums[2, i] / count) if count else None,
               "mean_linear_hdr": float(sums[3, i] / count) if count else None}
        for i, (name, count) in enumerate(zip(REGIONS, counts, strict=True))
    }, fixed


def verify_receiver_baseline(current, old):
    for name in REGIONS:
        row, expected = current[name], old[name]
        if row["samples"] != expected["samples"]:
            raise ValueError(f"baseline sample count changed: {name}")
        for key in ("display_energy", "hdr_energy"):
            if abs(row[key] - expected[key]) > max(abs(expected[key]) * .01, 1e-6):
                raise ValueError(f"baseline receiver {name} {key} changed")


def reflection_difference(control, variant):
    if control.shape != variant.shape or not np.isfinite(control).all() or not np.isfinite(variant).all():
        raise ValueError("invalid reflection readback")
    delta = np.abs(control.astype(np.float32) - variant.astype(np.float32))
    return {
        "size_wh": [control.shape[1], control.shape[0]],
        "changed_pixels": int(np.any(delta > 0, axis=2).sum()),
        "mean_absolute_rgb_delta": float(delta.mean(dtype=np.float64)),
        "max_absolute_rgb_delta": float(delta.max()),
    }


def measure_view(view, display_mode, residual_record, baseline):
    if json.loads(json.dumps(asdict(view))) != baseline["view"]:
        raise ValueError(f"baseline camera changed: {view.view_id}")
    cases = {}
    labels = None
    for route in ROUTES:
        name = "control" if route is None else f"{route}_off"
        print(f"{view.view_id}: {name}", flush=True)
        with shader_variant(REPLACEMENTS) as sources, window_route(route, view.position_eus_m) as calls:
            pan = render_pan(view, display_mode, FRAMES, PIXELS_PER_FRAME, subject_capture=route is None,
                             subject_fragment_builder=background_fragment if route is None else None,
                             capture_first_reflection=True)
        if any(item["draws"] < FRAMES or item["suppressed"] != (item["draws"] if key == route else 0)
               for key, item in calls.items()):
            raise ValueError(f"incomplete draw routing: {calls}")
        residuals = select_residuals(residual_record, view.view_id, pan["gl"])
        run = score_run(pan, residuals)
        identity = {"gl": pan["gl"], "render_config": pan["render_config"],
                    "yaw_step_deg": pan["yaw_step_deg"], "display_pixels_wh": list(reversed(run["scored"].shape)),
                    "excluded_residual_ids": [entry["id"] for entry in residuals]}
        if any(baseline[key] != value for key, value in identity.items()):
            raise ValueError(f"baseline capture environment changed: {view.view_id}")
        summary = capture_summary(pan, run, run["scored"])
        reflection = pan.pop("first_reflection_rgb")
        if route is None:
            check = replay_control(baseline["masked_moving"], summary)
            if not check["passed"] or summary["scored_pixels"] != baseline["masked_moving"]["scored_pixels"]:
                raise ValueError(f"diagnostic control did not reproduce V0-19: {check}")
            domain, matrices = run["scored"], pan["matrices"]
            hdr_scale = np.maximum(np.abs(pan["linear_hdr"][1:]).mean(axis=0)[domain], 1e-6)
            reflection_control = reflection
        elif not np.array_equal(matrices, pan["matrices"]) or not np.array_equal(domain, run["scored"]):
            raise ValueError("intervention changed the camera or scoring domain")
        print(f"{view.view_id}: {name} receiver analysis", flush=True)
        regions, labels = receiver_statistics(pan, domain, hdr_scale, labels)
        if route is None:
            old = next(part for part in baseline["partitions"] if part["source_support_radius"] == RADIUS)
            verify_receiver_baseline(regions, old["regions"])
        cases[name] = {
            "whole_view": summary, "regions": regions, "draw_routing": calls,
            "reflection_first_frame_delta": reflection_difference(reflection_control, reflection),
            "diagnostic_shader_sha256": sources,
        }
        del pan, run
    direct = cases["direct_off"]["reflection_first_frame_delta"]["mean_absolute_rgb_delta"]
    reflected = cases["reflection_off"]["reflection_first_frame_delta"]["mean_absolute_rgb_delta"]
    if reflected <= 1e-8 or direct > max(1e-8, reflected * .01):
        raise ValueError(f"reflection readback did not separate the routes: {direct=} {reflected=}")
    return {"view": asdict(view), **identity, "baseline_control": check, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    suite, views = verified_views()
    baseline_bytes = BASELINE.read_bytes()
    baseline = json.loads(baseline_bytes)
    identity = baseline_identity(suite)
    if any(baseline[key] != value for key, value in identity.items()) or baseline["distant_facade_min_distance_m"] != DISTANT_M:
        raise ValueError("V0-19 measurement identity changed")
    previous = {row["view"]["view_id"]: row for row in baseline["views"]}
    record = load_known_residuals()
    rows = [measure_view(view, suite.display_mode, record, previous[view.view_id]) for view in views]
    report = {
        "schema_version": 1, "probe": "window_routes_v0_20", **identity,
        "v0_19_report_sha256": sha256(baseline_bytes).hexdigest(),
        "distant_window_min_distance_m": DISTANT_M, "regional_source_support_radius": RADIUS,
        "views": rows,
        "limitations": [
            "Only final facade emission for wall fragments 250 m or more from the real eye is suppressed.",
            "Direct and reflected draws use the same diagnostic shader; the control leaves emission enabled in both.",
            "Reflection readback verifies routing only at the first pose; direct post-processing can still redistribute light.",
            "Receiver membership and regional HDR normalization are frozen to the unmodified control.",
            "Ablation may dim the scene; score reductions are not causal fractions, additive shares or a candidate adoption verdict.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
