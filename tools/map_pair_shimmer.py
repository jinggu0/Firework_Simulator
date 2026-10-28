"""V2-3l: reproject frame-pair semantic labels and check prediction support."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulator import shaders
from simulator.validation.capture import load_known_residuals, select_residuals
from simulator.validation.temporal_shimmer import pan_source_coordinates
from simulator.validation.views import load_visual_regression_suite
from tools.ab_shader_candidate import score_run, stability
from tools.measure_temporal_shimmer import FRAMES, PIXELS_PER_FRAME, render_pan
from tools.pair_shimmer_regions import REGIONS, SUPPORT_RADII, pair_masks, pair_statistics


COLOURS = dict(zip(REGIONS, ((255, 125, 40), (50, 235, 215), (230, 80, 220),
                            (160, 160, 160), (245, 220, 60), (50, 120, 240), (45, 45, 60)), strict=True))


def save_visual(path, pan):
    pair = (len(pan["display_luma"]) - 1) // 2
    display = pan["display_luma"][pair + 1]
    height, width = display.shape
    coordinates = pan_source_coordinates(pan["matrices"][pair], pan["matrices"][pair + 1], width, height)
    canvas = Image.new("RGB", (width * 2, height + 155), (15, 18, 25))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    background = np.repeat(np.clip(display, 0, 255)[..., None], 3, axis=2)
    for index, radius in enumerate((0, 1)):
        labels = np.zeros((height, width, 3), np.uint8)
        masks = pair_masks(pan["semantic_fractions"][pair], pan["semantic_fractions"][pair + 1], coordinates, radius)
        for name, mask in masks.items():
            labels[mask] = COLOURS[name]
        panel = (background * .5 + labels * .5).astype(np.uint8)
        canvas.paste(Image.fromarray(panel), (index * width, 45))
        draw.text((index * width + 15, 12), f"Pair {pair}->{pair + 1}: {2 * radius + 2}x{2 * radius + 2} source support", fill="white", font=font)
    for index, (name, colour) in enumerate(COLOURS.items()):
        x, y = 15 + (index % 3) * (width * 2 // 3), height + 62 + (index // 3) * 28
        draw.rectangle((x, y, x + 15, y + 15), fill=colour)
        draw.text((x + 23, y - 2), name, fill="white", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="bridge_structure")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.frames < 21:
        parser.error("at least 21 frames are required")
    output = args.output or Path(f"docs/validation/shimmer_ab_v2/diagnostics/results/pair_{args.view}_attribution.json")
    cache = args.cache or Path(f"local_reference/shimmer_regions_v2/pair_{args.view}.npz")
    suite = load_visual_regression_suite()
    view = suite.view(args.view)
    print("independent control pan", flush=True)
    control_pan = render_pan(view, suite.display_mode, args.frames, PIXELS_PER_FRAME)
    residuals = select_residuals(load_known_residuals(), view.view_id, control_pan["gl"])
    control_run = score_run(control_pan, residuals)
    control = stability(control_run)
    control["mean_display_code"] = float(control_pan["display_luma"][1:].mean(axis=0)[control_run["scored"]].mean())
    del control_pan, control_run
    print("pan with private semantic replay", flush=True)
    pan = render_pan(view, suite.display_mode, args.frames, PIXELS_PER_FRAME, semantic_capture=True)
    run = score_run(pan, residuals)
    measured = stability(run)
    measured["mean_display_code"] = float(pan["display_luma"][1:].mean(axis=0)[run["scored"]].mean())
    delta = {key: abs(measured[key] - control[key]) for key in control}
    valid = all(delta[key] <= max(abs(control[key]) * .01, 1e-6)
                for key in ("display_excess", "hdr_excess", "mean_display_code"))
    # Full sequences and matrices are retained locally, not merely score maps,
    # so later support analysis can reuse exactly this capture without a GPU.
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, linear_hdr=pan["linear_hdr"], display_luma=pan["display_luma"],
                        semantic_fractions=pan["semantic_fractions"], matrices=np.asarray(pan["matrices"]),
                        scored=run["scored"], display_score=run["display_score"], hdr_score=run["hdr_score"])
    print(f"control valid={valid}; cached complete sequence; analysing pairs", flush=True)
    partitions = pair_statistics(pan, run, progress=lambda done, total: print(f"pair {done}/{total}", flush=True))
    report = {"schema_version": 1, "probe": "pair_shimmer_v2_3l", "view_id": view.view_id,
              "frames": args.frames, "pixels_per_frame_at_centre": PIXELS_PER_FRAME,
              "gl": pan["gl"], "render_config": pan["render_config"],
              "control": control, "with_semantic_replay": measured,
              "control_absolute_difference": delta, "control_within_one_percent": valid,
              "shipped_shader_sha256": {name: sha256(shaders.source(name).encode()).hexdigest() for name in shaders.available()},
              "semantic_fragment_sha256": {name: sha256(source.encode()).hexdigest() for name, source in pan["semantic_sources"].items()},
              "support_radii_declared_before_measurement": list(SUPPORT_RADII),
              "metric": "sum squared pair prediction residual; HDR divided by the original per-pixel path mean magnitude (floor 1e-6)",
              "partitions": partitions,
              "limitations": [
                  "Energy shares are not the V2-3k thresholded RMS-excess shares and are not causal removal effects.",
                  "Cubic interpolation uses local 4x4 prefiltered coefficients, not just raw pixels; prefilter tails have infinite support.",
                  "16x16 raw-pixel homogeneity is a conservative sensitivity check, not a proof of no outside influence.",
                  "Clean refers to analytic pane/facade/water labels, not occupancy, emitted-light filter support or same-building identity.",
                  "Bloom, AO and haze may carry light across receiver labels; pane centre agreement alone does not isolate interior shading.",
                  "The common original scored domain and known-residual exclusions are unchanged; conditional RMS with zero samples is null."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save_visual(output.with_suffix(".png"), pan)
    print(f"wrote {output}", flush=True)
    if not valid:
        raise SystemExit("semantic replay exceeded the predeclared one-percent control tolerance")


if __name__ == "__main__":
    main()
