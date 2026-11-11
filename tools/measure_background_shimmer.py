"""Split V0-18 background residuals while preserving subject and boundary support."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from simulator import shaders
from simulator.scene import SURFACE_WALL
from simulator.validation.capture import load_known_residuals, select_residuals
from simulator.validation.views import DEFAULT_VISUAL_VIEWS_PATH, REPOSITORY_ROOT
from tools.ab_shader_candidate import score_run
from tools.audit_subject_visibility import BRIDGE_REGION, ROAD_REGION, visibility_fragment
from tools.measure_subject_shimmer import (
    OUTPUT as SUBJECT_OUTPUT,
    RADII, VISIBILITY_DIR, capture_summary, replay_control, save_preview, subject_pair_masks, verified_views,
)
from tools.measure_temporal_shimmer import (
    DISPLAY_THRESHOLD_CODE_VALUES, FRAMES, PIXELS_PER_FRAME, RELATIVE_THRESHOLD, WARMUP_FRAMES, render_pan,
)
from tools.pair_shimmer_regions import pair_statistics, source_support
from tools.shimmer_regions import ASSIGNMENT


OUTPUT = REPOSITORY_ROOT / "docs/validation/background_shimmer_v0"
BASELINE = SUBJECT_OUTPUT / "report.json"
DISTANT_M = 250.0
BACKGROUND = ("facade_interior", "facade_mixed", "water_interior", "water_mixed", "facade_water_mixed", "other")
REGIONS = ("subject_interior", "subject_transition", *BACKGROUND)
PARENT_REGIONS = {
    "subject_interior": ("subject_interior",),
    "subject_transition": ("subject_transition",),
    "non_subject": BACKGROUND,
}
COLOURS = (
    ("subject_interior", (35, 223, 210)), ("subject_transition", (255, 175, 55)),
    ("facade_interior", (241, 72, 160)), ("facade_mixed", (241, 72, 160)),
    ("water_interior", (65, 145, 255)), ("water_mixed", (65, 145, 255)),
    ("facade_water_mixed", (177, 110, 240)),
)


def background_fragment(source, kind):
    """Reuse the exact subject test and depth, replacing only mask outputs."""
    source = visibility_fragment(source, kind)
    distant_wall = (
        f"(!subject_hit && abs(surface - {SURFACE_WALL}) < .25 "
        f"&& distance(world_position, audit_eye) >= {DISTANT_M})"
    ) if kind == "scene" else "false"
    water = "1.0" if kind == "water" else "0.0"
    return ASSIGNMENT.sub(
        f"frag_color = vec4(float(subject_hit), float({distant_wall}), {water}, 1.0);",
        source,
    )


def background_pair_masks(previous, current, coordinates, radius):
    """Refine only the prior non-subject mask, leaving all subject edges intact."""
    previous, current = np.asarray(previous), np.asarray(current)
    masks = subject_pair_masks(previous, current, coordinates, radius)
    non_subject = masks.pop("non_subject")
    for values in (previous, current):
        background = values[..., 1:]
        if (not np.isfinite(background).all() or background.min() < 0 or background.max() > 1
                or np.any(values.sum(axis=2, dtype=np.float32) > 1.001)):
            raise ValueError("invalid or overlapping receiver coverage")
    pg, pb = previous[..., 1], previous[..., 2]
    cg, cb = current[..., 1], current[..., 2]
    no_facade = (cg <= .001) & source_support(pg <= .001, coordinates, radius)
    no_water = (cb <= .001) & source_support(pb <= .001, coordinates, radius)
    facade = non_subject & ~no_facade & no_water
    water = non_subject & no_facade & ~no_water
    facade_interior = facade & (cg >= .999) & source_support(pg >= .999, coordinates, radius)
    water_interior = water & (cb >= .999) & source_support(pb >= .999, coordinates, radius)
    return {
        **masks,
        "facade_interior": facade_interior,
        "facade_mixed": facade & ~facade_interior,
        "water_interior": water_interior,
        "water_mixed": water & ~water_interior,
        "facade_water_mixed": non_subject & ~no_facade & ~no_water,
        "other": non_subject & no_facade & no_water,
    }


def compare_parent_partition(partition, baseline):
    if (partition["source_support_radius"] != baseline["source_support_radius"]
            or partition["scored_pixel_pairs"] != baseline["scored_pixel_pairs"]):
        raise ValueError("baseline pair domain changed")
    rows = {}
    for name, children in PARENT_REGIONS.items():
        observed = partition["regions"]
        expected = baseline["regions"][name]
        samples = sum(observed[child]["samples"] for child in children)
        if samples != expected["samples"]:
            raise ValueError(f"baseline {name} sample count changed: {samples} != {expected['samples']}")
        rows[name] = {"samples": samples}
        for key in ("display_energy", "hdr_energy"):
            value = math.fsum(observed[child][key] for child in children)
            tolerance = max(abs(expected[key]) * .01, 1e-6)
            delta = abs(value - expected[key])
            if delta > tolerance:
                raise ValueError(f"baseline {name} {key} changed: {value} != {expected[key]}")
            rows[name][key] = {"collapsed": value, "baseline": expected[key], "difference": delta,
                               "tolerance": tolerance}
    return rows


def baseline_identity(suite):
    return {
        "suite_id": suite.suite_id, "display_mode": suite.display_mode,
        "scene_asset_sha256": suite.scene_asset_sha256,
        "canonical_views_sha256": sha256(DEFAULT_VISUAL_VIEWS_PATH.read_bytes()).hexdigest(),
        "candidate_views_sha256": sha256((VISIBILITY_DIR / "candidate_views.json").read_bytes()).hexdigest(),
        "visibility_report_sha256": sha256((VISIBILITY_DIR / "report.json").read_bytes()).hexdigest(),
        "shipped_shader_sha256": {name: sha256(shaders.source(name).encode()).hexdigest() for name in shaders.available()},
        "subject_bounds_xz": {"bridge_structure": list(BRIDGE_REGION), "road_ground": list(ROAD_REGION)},
        "frames": FRAMES, "warmup_frames": WARMUP_FRAMES, "pixels_per_frame_at_centre": PIXELS_PER_FRAME,
        "display_excess_threshold": DISPLAY_THRESHOLD_CODE_VALUES, "relative_hdr_excess_threshold": RELATIVE_THRESHOLD,
        "source_support_radii": list(RADII),
    }


def measure_view(view, display_mode, residual_record, baseline, output):
    if json.loads(json.dumps(asdict(view))) != baseline["view"]:
        raise ValueError(f"baseline camera changed: {view.view_id}")
    print(f"{view.view_id}: background receiver capture", flush=True)
    pan = render_pan(view, display_mode, FRAMES, PIXELS_PER_FRAME, subject_capture=True,
                     subject_fragment_builder=background_fragment)
    residuals = select_residuals(residual_record, view.view_id, pan["gl"])
    run = score_run(pan, residuals)
    identity = {"gl": pan["gl"], "render_config": pan["render_config"],
                "yaw_step_deg": pan["yaw_step_deg"], "display_pixels_wh": list(reversed(run["scored"].shape)),
                "excluded_residual_ids": [entry["id"] for entry in residuals]}
    if any(baseline[key] != value for key, value in identity.items()):
        raise ValueError(f"baseline capture environment changed: {view.view_id}")
    measured = capture_summary(pan, run, run["scored"])
    control = replay_control(baseline["plain_moving"], measured)
    if not control["passed"] or measured["scored_pixels"] != baseline["plain_moving"]["scored_pixels"]:
        raise ValueError(f"background replay did not reproduce the V0-18 moving baseline: {control}")
    print(f"{view.view_id}: baseline reproduced; splitting pairs", flush=True)
    partitions = pair_statistics(pan, run, RADII, partition=background_pair_masks, region_names=REGIONS)
    for partition, old in zip(partitions, baseline["partitions"], strict=True):
        partition["collapsed_baseline_check"] = compare_parent_partition(partition, old)
    save_preview(output / f"{view.view_id}.png", pan, run, partition=background_pair_masks, colours=COLOURS,
                 label_title="4x4 receiver support: subject cyan / edge amber / facade pink / water blue / joint purple")
    return {
        "view": asdict(view), **identity, "masked_moving": measured, "baseline_control": control,
        "partitions": partitions,
        "semantic_fragment_sha256": {key: sha256(value.encode()).hexdigest()
                                     for key, value in pan["semantic_sources"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    suite, views = verified_views()
    baseline_bytes = BASELINE.read_bytes()
    baseline = json.loads(baseline_bytes)
    identity = baseline_identity(suite)
    for key, value in identity.items():
        if baseline[key] != value:
            raise ValueError(f"V0-18 measurement identity changed: {key}")
    previous = {row["view"]["view_id"]: row for row in baseline["views"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = load_known_residuals()
    rows = [measure_view(view, suite.display_mode, record, previous[view.view_id], args.output_dir)
            for view in views]
    report = {
        "schema_version": 1, "probe": "background_shimmer_v0_19", **identity,
        "v0_18_report_sha256": sha256(baseline_bytes).hexdigest(),
        "distant_facade_min_distance_m": DISTANT_M,
        "views": rows,
        "limitations": [
            "Receiver labels are not causal attribution. Water may reflect a distant facade, a bridge lamp or another light source.",
            "Facade means wall material at 250 m or farther; roofs and nearer walls remain other.",
            "Mixed means at least one matching label and a heterogeneous raw support; it does not isolate a specific edge.",
            "Support windows cannot bound the full cubic prefilter or post-processing influence.",
            "Frozen-time yaw only; this comparison reproduces a baseline, not an appearance or performance improvement.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
