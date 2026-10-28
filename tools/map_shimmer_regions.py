"""V2-3k: localise unchanged shipped shimmer with separate semantic replays.

Reports spatial support, not causal fractions. A source-pixel classification
cannot identify the origin of light spread across boundaries by bloom or AO.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulator import shaders
from simulator.validation.capture import load_known_residuals, select_residuals
from simulator.validation.views import load_visual_regression_suite
from tools.ab_shader_candidate import score_run, stability
from tools.measure_temporal_shimmer import FRAMES, PIXELS_PER_FRAME, render_pan
from tools.shimmer_regions import region_masks, region_statistics


COLOURS = {"facade_boundary_band": (255, 125, 40), "pane_interior": (50, 235, 215),
           "wall_interior": (160, 160, 160), "pane_transition": (245, 220, 60),
           "water_interior": (50, 120, 240), "other_or_moving_support": (45, 45, 60)}


def save_visual(path, display, run, fractions):
    masks = region_masks(fractions, 2)
    height, width = display.shape
    labels = np.zeros((height, width, 3), np.uint8)
    for name, mask in masks.items():
        labels[mask] = COLOURS[name]
    background = np.repeat(np.clip(display, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    left = (background * .5 + labels * .5).astype(np.uint8)
    excess = np.maximum(run["display_score"] - 2, 0)
    scale = max(float(np.percentile(excess[run["scored"]], 99.5)), 1e-6)
    weight = np.where(run["scored"], np.clip(excess / scale, 0, 1), 0)
    right = (background * (1 - weight[..., None]) + np.array([255, 50, 25]) * weight[..., None]).astype(np.uint8)
    canvas = Image.new("RGB", (width * 2, height + 130), (15, 18, 25))
    canvas.paste(Image.fromarray(left), (0, 45))
    canvas.paste(Image.fromarray(right), (width, 45))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((15, 12), "Whole-pan semantic support (boundary radius: 2px)", font=font, fill="white")
    draw.text((width + 15, 12), "Unchanged display excess flicker (red)", font=font, fill="white")
    for index, (name, colour) in enumerate(COLOURS.items()):
        x, y = 15 + (index % 3) * (width * 2 // 3), height + 62 + (index // 3) * 28
        draw.rectangle((x, y, x + 15, y + 15), fill=colour)
        draw.text((x + 23, y - 2), name, fill="white", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="bridge_structure")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--output", type=Path, default=Path("docs/validation/shimmer_ab_v2/diagnostics/results/spatial_bridge_attribution.json"))
    parser.add_argument("--cache", type=Path, default=Path("local_reference/shimmer_regions_v2/bridge_structure.npz"))
    args = parser.parse_args()
    if args.frames < 21:
        parser.error("at least 21 frames are required")
    suite = load_visual_regression_suite()
    view = suite.view(args.view)
    print("control pan", flush=True)
    baseline = render_pan(view, suite.display_mode, args.frames, PIXELS_PER_FRAME)
    residuals = select_residuals(load_known_residuals(), view.view_id, baseline["gl"])
    control_run = score_run(baseline, residuals)
    control = stability(control_run)
    control["mean_display_code"] = float(baseline["display_luma"][1:].mean(axis=0)[control_run["scored"]].mean())
    # Score arrays are sufficient for the control; release frame sequences before
    # allocating the instrumented run's private targets and semantic sequence.
    del baseline, control_run
    print("pan with private semantic replay", flush=True)
    pan = render_pan(view, suite.display_mode, args.frames, PIXELS_PER_FRAME, semantic_capture=True)
    run = score_run(pan, residuals)
    measured = stability(run)
    measured["mean_display_code"] = float(pan["display_luma"][1:].mean(axis=0)[run["scored"]].mean())
    delta = {key: abs(measured[key] - control[key]) for key in control}
    valid = all(delta[key] <= max(abs(control[key]) * .01, 1e-6)
                for key in ("display_excess", "hdr_excess", "mean_display_code"))
    pane_counts = (pan["semantic_fractions"][..., 1] >= .999).sum(axis=(1, 2))
    report = {"schema_version": 1, "probe": "spatial_shimmer_v2_3k", "view_id": view.view_id,
              "frames": args.frames, "pixels_per_frame_at_centre": PIXELS_PER_FRAME,
              "semantic_support": {
                  "fully_covered_pane_pixel_frames": int(pane_counts.sum()),
                  "fully_covered_pane_pixels_per_frame_min": int(pane_counts.min()),
                  "fully_covered_pane_pixels_per_frame_max": int(pane_counts.max())},
              "gl": pan["gl"], "render_config": pan["render_config"],
              "control": control, "with_semantic_replay": measured,
              "control_absolute_difference": delta, "control_within_one_percent": valid,
              "shipped_shader_sha256": {name: sha256(shaders.source(name).encode()).hexdigest() for name in shaders.available()},
              "semantic_fragment_sha256": {name: sha256(source.encode()).hexdigest() for name, source in pan["semantic_sources"].items()},
              "partitions": [region_statistics(run, pan["semantic_fractions"], radius) for radius in (2, 4)],
              "limitations": [
                  "Spatial receiver-pixel support, not causal attribution; original HDR/display colours are read before replay.",
                  "Pane is the single-fragment analytic mask, not the 16-sample emitted-light average or occupancy.",
                  "Boundary is the whole-path union of window-bearing facade coverage contours, not every depth edge between buildings.",
                  "Display bloom, haze and AO can move light across these receiver categories.",
                  "All-pose interior membership is conservative; transition/moving-support buckets are retained.",
                  "Two and four pixel boundary bands were fixed before measurement; control tolerance is one percent."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, fractions=pan["semantic_fractions"], display_score=run["display_score"],
                        hdr_score=run["hdr_score"], scored=run["scored"], display=pan["display_luma"][0])
    save_visual(args.output.with_suffix(".png"), pan["display_luma"][0], run, pan["semantic_fractions"])
    print(f"control valid={valid}; wrote {args.output}", flush=True)
    if not valid:
        raise SystemExit("semantic replay changed the measured result beyond the predeclared tolerance")


if __name__ == "__main__":
    main()
