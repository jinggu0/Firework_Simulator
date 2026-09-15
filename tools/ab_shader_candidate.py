"""A/B a shader candidate for temporal shimmer, guarded against what it removes.

For each view three pans are rendered: the shipped shader twice, in separate
applications, and the candidate once with its replacements served in memory
(`tools.shader_candidate`). The two shipped runs differ only by the renderer
disagreeing with itself, so their difference is the noise every comparison has
to clear. A supersampled reference of the first pose, rendered by the shipped
shader, anchors the detail guard (`simulator.validation.detail_guard`).

The rule is fixed before anything is measured. A candidate is accepted only if
in every view its display excess flicker is no higher than the shipped shader's
and it is no further from the reference at any guard scale, each beyond a
tolerance of measured noise or a one percent margin — and if its display excess
flicker is lower beyond that tolerance in at least one view. Revised in V2-3g,
before re-judging, because the share of pixels over the threshold saturated.

Example::

    python -m tools.ab_shader_candidate \\
        --candidate docs/validation/shimmer_ab_v2/candidates/window_grid_expectation_fade.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from simulator import shaders
from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import (
    load_known_residuals,
    read_linear_hdr,
    select_residuals,
)
from simulator.validation.detail_guard import GUARD_SCALES_PX, detail_guard
from simulator.validation.frame_comparison import LUMINANCE_WEIGHTS
from simulator.validation.temporal_shimmer import (
    BORDER_MARGIN_PX,
    excess_flicker,
    pan_source_coordinates,
    shimmer_scores,
)
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)
from tools.measure_temporal_shimmer import (
    DISPLAY_THRESHOLD_CODE_VALUES,
    FRAMES,
    PIXELS_PER_FRAME,
    RELATIVE_THRESHOLD,
    WARMUP_FRAMES,
    render_pan,
    residual_exclusion_mask,
)
from tools.shader_candidate import (
    DEFAULT_CANDIDATE_DIRECTORY,
    load_candidate,
    shader_variant,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = DEFAULT_CANDIDATE_DIRECTORY / "window_grid_expectation_fade.json"
#: The same supersample the static aliasing baseline used.
SUPERSAMPLE = 4
#: Stability is decided in the display frame by excess flicker: how far past
#: the visibility threshold each pixel flickers (V2-3g). The share of pixels
#: over the threshold saturated where fixes act (V2-3f), and linear HDR's
#: relative score has no ceiling on near-black pixels, so both are reported
#: beside it without deciding.
DECIDING_KEY = "display_excess"
REPORTED_KEYS = ("display_excess", "hdr_excess")
#: Non-inferiority margin, declared in V2-3g before any candidate was re-judged.
#: Renders are deterministic almost everywhere, so measured noise is often
#: exactly zero and a candidate that touches a dozen pixels of a view would fail
#: it on that alone. A difference under one percent of the shipped value counts
#: as no difference, for stability and for the guard. It is a policy choice,
#: not a measured quantity.
EQUIVALENCE_MARGIN = 0.01


def portable_path(path: Path) -> str:
    """Repository-relative when inside it, so reports do not carry a home path."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _tolerance(shipped: float, repeat: float) -> float:
    """The larger of the measured noise and the equivalence margin."""

    return max(abs(shipped - repeat), EQUIVALENCE_MARGIN * abs(shipped))


def view_verdict(
    shipped: dict[str, Any], repeat: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare one view's candidate with the shipped shader, beyond noise."""

    noise = {key: abs(shipped[key] - repeat[key]) for key in REPORTED_KEYS}
    guard_noise = {
        scale: abs(shipped["guard"][scale] - repeat["guard"][scale])
        for scale in shipped["guard"]
    }
    tolerance = _tolerance(shipped[DECIDING_KEY], repeat[DECIDING_KEY])
    return {
        "noise": {**noise, "guard": guard_noise},
        "tolerance": tolerance,
        "more_stable": bool(candidate[DECIDING_KEY] < shipped[DECIDING_KEY] - tolerance),
        "not_less_stable": bool(
            candidate[DECIDING_KEY] <= shipped[DECIDING_KEY] + tolerance
        ),
        "guard_held": {
            scale: bool(
                candidate["guard"][scale]
                <= shipped["guard"][scale]
                + _tolerance(shipped["guard"][scale], repeat["guard"][scale])
            )
            for scale in shipped["guard"]
        },
    }


def candidate_verdict(views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the fixed acceptance rule across every view."""

    reasons = []
    for name, verdict in views.items():
        if not verdict["not_less_stable"]:
            reasons.append(
                f"{name}: less stable than the shipped shader beyond noise and margin"
            )
        for scale, held in verdict["guard_held"].items():
            if not held:
                reasons.append(
                    f"{name}: further from the reference at {scale} px blocks "
                    "beyond noise and margin"
                )
    if not any(verdict["more_stable"] for verdict in views.values()):
        reasons.append(
            "no view is more stable in the display frame beyond noise and margin"
        )
    return {"accepted": not reasons, "reasons": reasons}


def changed_pixel_mask(shipped: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Pixels the candidate rendered differently, compared exactly."""

    first = np.asarray(shipped)
    second = np.asarray(candidate)
    if first.shape != second.shape:
        raise ValueError("frames must share one shape")
    return first != second


def render_reference(view, display_mode: str, factor: int) -> np.ndarray:
    """Linear HDR luminance of the first pan pose, supersampled then averaged."""

    base = SimulationConfig()
    width, height = base.render.width, base.render.height
    render = replace(
        base.render, vsync=False, target_fps=0,
        width=width * factor, height=height * factor,
    )
    app = SimulatorApp(replace(base, render=render), scenario_path=DEFAULT_SCENARIO_PATH)
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        app.renderer.post.set_mode(DisplayMode(display_mode))
        view.apply(app.camera)
        for _ in range(WARMUP_FRAMES):
            app.renderer.render(app.world, app.camera, app.celestial, 1.0 / 60.0, None)
        # The pan's first frame is drawn exactly like this: time frozen and the
        # reflection refreshed for the pose.
        app.renderer.reflection_ready = False
        app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
        app.ctx.finish()
        luminance = (
            read_linear_hdr(app.renderer)[:, :, :3].astype(np.float64) @ LUMINANCE_WEIGHTS
        )
        return luminance.reshape(height, factor, width, factor).mean(axis=(1, 3))
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def score_run(pan: dict[str, Any], residuals: list[dict]) -> dict[str, Any]:
    """Per-pixel scores in both units, with the path-grown residual exclusion."""

    frames, height, width = pan["linear_hdr"].shape
    matrices = pan["matrices"]
    coordinates = np.stack(
        [
            pan_source_coordinates(matrices[k], matrices[k + 1], width, height)
            for k in range(frames - 1)
        ]
    )
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    motion = np.hypot(coordinates[:, 0] - rows, coordinates[:, 1] - cols)
    dilation = math.ceil(float(motion.max(axis=(1, 2)).sum())) + BORDER_MARGIN_PX
    exclude = residual_exclusion_mask((height, width), residuals, dilation)
    display_score, scored = shimmer_scores(pan["display_luma"], coordinates, relative=False)
    hdr_score, _ = shimmer_scores(pan["linear_hdr"], coordinates, relative=True)
    scored = scored & ~exclude
    return {
        "display_score": display_score,
        "hdr_score": hdr_score,
        "display_unstable_map": (display_score > DISPLAY_THRESHOLD_CODE_VALUES) & scored,
        "hdr_unstable_map": (hdr_score > RELATIVE_THRESHOLD) & scored,
        "scored": scored,
        "first_frame": pan["linear_hdr"][0],
        "gl": pan["gl"],
        "yaw_step_deg": pan["yaw_step_deg"],
    }


def _stability(run: dict[str, Any], mask: np.ndarray | None = None) -> dict[str, Any]:
    scored = run["scored"] if mask is None else run["scored"] & mask
    if not scored.any():
        return {
            "display_excess": 0.0, "hdr_excess": 0.0,
            "display_unstable": 0.0, "hdr_unstable": 0.0,
        }
    return {
        "display_excess": excess_flicker(
            run["display_score"], scored, DISPLAY_THRESHOLD_CODE_VALUES
        ),
        "hdr_excess": excess_flicker(run["hdr_score"], scored, RELATIVE_THRESHOLD),
        "display_unstable": float(run["display_unstable_map"][scored].mean()),
        "hdr_unstable": float(run["hdr_unstable_map"][scored].mean()),
    }


def measure_view(view, display_mode, frames, record, candidate) -> dict[str, Any]:
    runs = {}
    digests: dict[str, str] = {}
    for label in ("shipped", "repeat", "candidate"):
        variant = (
            shader_variant(candidate.replacements) if label == "candidate" else nullcontext({})
        )
        with variant as served:
            pan = render_pan(view, display_mode, frames, PIXELS_PER_FRAME)
        if label == "candidate":
            digests = served
        residuals = select_residuals(record, view.view_id, pan["gl"])
        runs[label] = score_run(pan, residuals)
        del pan
    reference = render_reference(view, display_mode, SUPERSAMPLE)
    shipped_frame = runs["shipped"]["first_frame"]
    against_candidate = detail_guard(shipped_frame, runs["candidate"]["first_frame"], reference)
    against_repeat = detail_guard(shipped_frame, runs["repeat"]["first_frame"], reference)
    summaries = {label: _stability(run) for label, run in runs.items()}
    for scale in against_candidate["scales"]:
        summaries["shipped"].setdefault("guard", {})[scale] = against_candidate["scales"][scale]["baseline"]
        summaries["repeat"].setdefault("guard", {})[scale] = against_repeat["scales"][scale]["candidate"]
        summaries["candidate"].setdefault("guard", {})[scale] = against_candidate["scales"][scale]["candidate"]
    verdict = view_verdict(summaries["shipped"], summaries["repeat"], summaries["candidate"])

    changed = changed_pixel_mask(runs["shipped"]["first_frame"], runs["candidate"]["first_frame"])
    shipped, candidate_run = runs["shipped"], runs["candidate"]
    region = changed & shipped["scored"] & candidate_run["scored"]
    shipped_unstable = shipped["display_unstable_map"]
    within = {
        "pixels": int(region.sum()),
        "shipped": _stability(shipped, region),
        "candidate": _stability(candidate_run, region),
    }
    shipped_unstable_count = int(shipped_unstable.sum())
    return {
        "view_id": view.view_id,
        "gl": shipped["gl"],
        "yaw_step_deg": shipped["yaw_step_deg"],
        "shipped": summaries["shipped"],
        "repeat": summaries["repeat"],
        "candidate": summaries["candidate"],
        "verdict": verdict,
        "changed_pixel_fraction": float(changed.mean()),
        "share_of_shipped_display_unstable_pixels_the_candidate_changed": (
            float((shipped_unstable & changed).sum() / shipped_unstable_count)
            if shipped_unstable_count else 0.0
        ),
        "within_changed_pixels": within,
        "candidate_shader_sha256": digests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    candidate = load_candidate(arguments.candidate)
    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    views = [
        view for view in suite.views
        if not arguments.view or view.view_id in arguments.view
    ]
    record = load_known_residuals()
    shipped_sources = {
        name: sha256(shaders.source(name).encode("utf-8")).hexdigest()
        for name in sorted({item.file for item in candidate.replacements})
    }
    results = []
    for view in views:
        print(f"measuring {view.view_id}", flush=True)
        results.append(measure_view(view, suite.display_mode, arguments.frames, record, candidate))

    verdict = candidate_verdict({result["view_id"]: result["verdict"] for result in results})
    report = {
        "schema_version": 1,
        "probe": "shader_candidate_ab_for_temporal_shimmer",
        "candidate_id": candidate.candidate_id,
        "candidate_file": portable_path(arguments.candidate),
        "candidate_file_sha256": sha256(arguments.candidate.read_bytes()).hexdigest(),
        "hypothesis": candidate.hypothesis,
        "shipped_shader_sha256": shipped_sources,
        "frames": arguments.frames,
        "pixels_per_frame_at_centre": PIXELS_PER_FRAME,
        "supersample": SUPERSAMPLE,
        "guard_scales_px": list(GUARD_SCALES_PX),
        "rule": (
            "Accepted only if, in every view, the candidate's display excess flicker is not "
            "higher than the shipped shader's by more than the tolerance, and it is no further "
            "from the supersampled reference at any guard scale by more than the tolerance; "
            "and its display excess flicker is lower by more than the tolerance in at least "
            "one view. The tolerance is the larger of the difference between two separate "
            "runs of the shipped shader and one percent of the shipped value."
        ),
        "rule_revision": (
            "V2-3g: excess flicker replaced the share of pixels over two code values, which "
            "saturated where fixes act (V2-3f); linear HDR is reported without deciding; a "
            "one percent equivalence margin was added; frame cost is not part of acceptance "
            "while performance work is deferred. All declared before any candidate was re-judged."
        ),
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "views": results,
        "verdict": verdict,
    }
    output = arguments.output or (
        REPOSITORY_ROOT / "docs" / "validation" / "shimmer_ab_v2"
        / f"{candidate.candidate_id}_ab.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(json.dumps(verdict, indent=2))
    for result in results:
        print(
            f"  {result['view_id']:18} excess {result['shipped']['display_excess']:.4f} -> "
            f"{result['candidate']['display_excess']:.4f}  changed {result['changed_pixel_fraction'] * 100:5.2f}%  "
            f"guard4 {result['shipped']['guard']['4']:.4f} -> {result['candidate']['guard']['4']:.4f}  "
            f"guard16 {result['shipped']['guard']['16']:.4f} -> {result['candidate']['guard']['16']:.4f}"
        )


if __name__ == "__main__":
    main()
