"""Measure separate temporal baselines on the V0-17 engineering cameras."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from simulator import shaders
from simulator.bridge_semantics import DEFAULT_BRIDGE_SEMANTICS_PATH
from simulator.validation.capture import load_known_residuals, select_residuals
from simulator.validation.subject_visibility import visibility_gates
from simulator.validation.temporal_shimmer import pan_source_coordinates
from simulator.validation.views import DEFAULT_VISUAL_VIEWS_PATH, REPOSITORY_ROOT, load_visual_regression_suite
from tools.ab_shader_candidate import score_run, stability
from tools.audit_subject_visibility import BRIDGE_REGION, ROAD_REGION
from tools.measure_temporal_shimmer import (
    DISPLAY_THRESHOLD_CODE_VALUES, FRAMES, PIXELS_PER_FRAME, RELATIVE_THRESHOLD, WARMUP_FRAMES, render_pan,
)
from tools.pair_shimmer_regions import pair_statistics, source_support


VISIBILITY_DIR = REPOSITORY_ROOT / "docs/validation/subject_visibility_v0"
OUTPUT = REPOSITORY_ROOT / "docs/validation/subject_shimmer_v0"
RADII = (1, 7)
REGIONS = ("subject_interior", "subject_transition", "non_subject")
CONTROL_KEYS = ("display_excess", "hdr_excess", "mean_display_code")
CONTROL_RELATIVE_TOLERANCE = .01
CONTROL_ABSOLUTE_FLOOR = 1e-6


def subject_pair_masks(previous, current, coordinates, radius):
    """Classify receiver pixels with matching subject labels at the source."""
    previous, current = np.asarray(previous), np.asarray(current)
    if previous.shape != current.shape or current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("expected matching height x width x 3 visibility masks")
    p, c = previous[..., 0], current[..., 0]
    for values in (p, c):
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("invalid subject coverage")
    subject = (c >= .999) & source_support(p >= .999, coordinates, radius)
    other = (c <= .001) & source_support(p <= .001, coordinates, radius)
    return dict(zip(REGIONS, (subject, ~(subject | other), other), strict=True))


def capture_summary(pan, run, domain):
    if not domain.any():
        raise ValueError("empty common scoring domain")
    return {
        **stability(run, domain),
        "mean_display_code": float(pan["display_luma"][1:].mean(axis=0)[domain].mean()),
        "scored_pixels": int(domain.sum()),
    }


def replay_control(plain, masked):
    differences = {key: abs(plain[key] - masked[key]) for key in CONTROL_KEYS}
    tolerances = {key: max(abs(plain[key]) * CONTROL_RELATIVE_TOLERANCE, CONTROL_ABSOLUTE_FLOOR)
                  for key in CONTROL_KEYS}
    return {"passed": all(differences[key] <= tolerances[key] for key in CONTROL_KEYS),
            "absolute_difference": differences, "tolerance": tolerances}


def verified_views():
    suite = load_visual_regression_suite(VISIBILITY_DIR / "candidate_views.json")
    suite.verify_scene_asset()
    visibility = json.loads((VISIBILITY_DIR / "report.json").read_text(encoding="utf-8"))
    if visibility["scene_asset_sha256"] != suite.scene_asset_sha256:
        raise ValueError("visibility audit and candidate suite disagree on the scene")
    if visibility["bridge_plan_sha256"] != sha256(DEFAULT_BRIDGE_SEMANTICS_PATH.read_bytes()).hexdigest():
        raise ValueError("bridge plan changed since the subject visibility audit")
    if visibility["subject_bounds_xz"] != {
        "bridge_structure": list(BRIDGE_REGION), "road_ground": list(ROAD_REGION),
    }:
        raise ValueError("subject bounds changed since the visibility audit")
    views = []
    for subject in ("bridge_structure", "road_ground"):
        view = next(v for v in suite.views if v.subject == subject)
        record = visibility["subjects"][subject]
        fresh = record["fresh_captures"].get(view.view_id)
        if (view.view_id == subject or json.loads(json.dumps(asdict(view))) != record["selected_view"]
                or fresh is None or not all(visibility_gates(subject, fresh["metrics"]).values())):
            raise ValueError(f"camera {view.view_id} lacks a matching visibility pass")
        views.append(view)
    return suite, views


def save_preview(path, pan, run):
    pair = (len(pan["display_luma"]) - 1) // 2
    display = pan["display_luma"][pair + 1]
    height, width = display.shape
    coordinates = pan_source_coordinates(pan["matrices"][pair], pan["matrices"][pair + 1], width, height)
    masks = subject_pair_masks(*pan["semantic_fractions"][pair:pair + 2], coordinates, RADII[0])
    gray = np.repeat(np.clip(display, 0, 255)[..., None], 3, axis=2)
    labels = gray.copy()
    for name, colour in (("subject_interior", (35, 223, 210)), ("subject_transition", (255, 175, 55))):
        chosen = masks[name] & run["scored"]
        labels[chosen] = .35 * gray[chosen] + .65 * np.array(colour)
    weight = np.where(run["scored"], np.clip((run["display_score"] - DISPLAY_THRESHOLD_CODE_VALUES) / 8, 0, 1), 0)
    heat = (1 - weight[..., None]) * gray + weight[..., None] * np.array([255, 75, 35])
    canvas = Image.new("RGB", (2 * width, height + 34), (20, 23, 29))
    canvas.paste(Image.fromarray(labels.astype(np.uint8)), (0, 34))
    canvas.paste(Image.fromarray(heat.astype(np.uint8)), (width, 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Middle pair: subject interior (cyan), transition (amber), 4x4 support", fill="white")
    draw.text((width + 10, 10), "Whole-path display excess: orange intensity from 0 to 8 code values", fill="white")
    canvas.save(path)


def measure_view(view, display_mode, residual_record, output):
    print(f"{view.view_id}: plain moving control", flush=True)
    plain_pan = render_pan(view, display_mode, FRAMES, PIXELS_PER_FRAME)
    gl, config = plain_pan["gl"], plain_pan["render_config"]
    residuals = select_residuals(residual_record, view.view_id, gl)
    plain_run = score_run(plain_pan, residuals)
    plain = capture_summary(plain_pan, plain_run, plain_run["scored"])
    matrices = plain_pan["matrices"]
    domain = plain_run["scored"]
    del plain_pan, plain_run

    print(f"{view.view_id}: frozen static control", flush=True)
    static_pan = render_pan(view, display_mode, FRAMES, 0)
    static_run = score_run(static_pan, residuals)
    static_domain = static_run["scored"] & domain
    static = capture_summary(static_pan, static_run, static_domain)
    if static_pan["gl"] != gl or static_pan["render_config"] != config:
        raise ValueError("static and moving capture environments differ")
    del static_pan, static_run

    print(f"{view.view_id}: moving subject masks", flush=True)
    pan = render_pan(view, display_mode, FRAMES, PIXELS_PER_FRAME, subject_capture=True)
    run = score_run(pan, residuals)
    if (pan["gl"] != gl or pan["render_config"] != config
            or not np.array_equal(matrices, pan["matrices"]) or not np.array_equal(domain, run["scored"])):
        raise ValueError("plain and masked pan do not match")
    measured = capture_summary(pan, run, domain)
    control = replay_control(plain, measured)
    if not control["passed"]:
        raise ValueError(f"subject mask replay changed the moving baseline: {control}")
    print(f"{view.view_id}: replay control passed; analysing pairs", flush=True)
    partitions = pair_statistics(pan, run, RADII, partition=subject_pair_masks, region_names=REGIONS)
    if not all(part["regions"]["subject_interior"]["samples"] for part in partitions):
        raise ValueError(f"{view.view_id} has no subject interior support")
    save_preview(output / f"{view.view_id}.png", pan, run)
    coverage = pan["semantic_fractions"][..., 0].mean(axis=(1, 2), dtype=np.float64)
    return {
        "view": asdict(view), "gl": gl, "render_config": config,
        "yaw_step_deg": pan["yaw_step_deg"],
        "display_pixels_wh": list(reversed(domain.shape)),
        "excluded_residual_ids": [entry["id"] for entry in residuals],
        "plain_moving": plain, "masked_moving": measured, "static": static, "replay_control": control,
        "moving_on_static_domain": capture_summary(pan, run, static_domain),
        "subject_frame_coverage_min_max": [float(coverage.min()), float(coverage.max())],
        "partitions": partitions,
        "semantic_fragment_sha256": {key: sha256(value.encode()).hexdigest()
                                     for key, value in pan["semantic_sources"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    suite, views = verified_views()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = load_known_residuals()
    rows = [measure_view(view, suite.display_mode, record, args.output_dir) for view in views]
    report = {
        "schema_version": 1, "probe": "subject_shimmer_v0_18", "suite_id": suite.suite_id,
        "display_mode": suite.display_mode, "scene_asset_sha256": suite.scene_asset_sha256,
        "canonical_views_sha256": sha256(DEFAULT_VISUAL_VIEWS_PATH.read_bytes()).hexdigest(),
        "candidate_views_sha256": sha256((VISIBILITY_DIR / "candidate_views.json").read_bytes()).hexdigest(),
        "visibility_report_sha256": sha256((VISIBILITY_DIR / "report.json").read_bytes()).hexdigest(),
        "shipped_shader_sha256": {name: sha256(shaders.source(name).encode()).hexdigest() for name in shaders.available()},
        "subject_bounds_xz": {"bridge_structure": BRIDGE_REGION, "road_ground": ROAD_REGION},
        "frames": FRAMES, "warmup_frames": WARMUP_FRAMES, "pixels_per_frame_at_centre": PIXELS_PER_FRAME,
        "display_excess_threshold": DISPLAY_THRESHOLD_CODE_VALUES, "relative_hdr_excess_threshold": RELATIVE_THRESHOLD,
        "source_support_radii": RADII, "views": rows,
        "limitations": [
            "Yaw only, at frozen simulation time; does not measure walking parallax, time-varying waves, or motion during fireworks.",
            "Regional energy is squared pair prediction residual, not thresholded RMS excess or causal attribution.",
            "Cubic prefilter tails extend beyond either raw-label support window; bloom, AO, and haze can also cross labels.",
            "Subject masks label receivers, not reflected lights or the cause of residuals; bridge piers and unrelated road materials are not subjects.",
            "These two cameras are a new baseline; no old-view comparison or quality adoption verdict is implied.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['view']['view_id']}: moving display excess {row['plain_moving']['display_excess']:.6f}; "
              f"static {row['static']['display_excess']:.6f}", flush=True)


if __name__ == "__main__":
    main()
