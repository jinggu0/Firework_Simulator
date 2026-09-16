"""Attribute a view's visible shimmer to one term of the renderer.

A diagnostic removes or flattens one term — a shader variant
(`tools.shader_candidate`) or a `RenderConfig` override — and the pan is scored
again. What the removal takes away is that term's share of the shimmer.

A diagnostic is not a candidate. It may change brightness, so the detail guard
says nothing here and is not run. Brightness is not harmless either: the
display transform's ACES curve is steeper through the midtones than at either
end, so a diagnostic that dims a region lowers its code-value flicker without
having touched the cause. Mean display code is therefore reported beside every
number, and a diagnostic that keeps a term's area mean is preferred to one that
sets it to zero.

Example::

    python -m tools.attribute_shimmer --view facade_landmark \\
        --diagnostic docs/validation/shimmer_ab_v2/diagnostics/diagnostic_window_blinds_at_mean.json \\
        --render-override bloom_strength=0.0
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import numpy as np

from simulator.validation.capture import load_known_residuals, select_residuals
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)
from tools.ab_shader_candidate import (
    changed_pixel_mask,
    portable_path,
    score_run,
    stability,
)
from tools.measure_temporal_shimmer import FRAMES, PIXELS_PER_FRAME, render_pan
from tools.shader_candidate import load_candidate, shader_variant


DEFAULT_OUTPUT = Path("docs/validation/shimmer_ab_v2/diagnostics/results/attribution.json")
#: Fields a `--render-override` may set, with the parser that reads the value.
#: Restricted on purpose: a diagnostic that resizes the frame or moves the
#: camera would not be comparable with the shipped run it is subtracted from.
OVERRIDE_FIELDS = {
    "bloom_strength": float,
    "ambient_occlusion_strength": float,
    "simulate_fixed_gaze_peripheral_acuity": lambda text: {
        "true": True, "false": False
    }[text.strip().lower()],
}


def parse_override(text: str) -> tuple[str, Any]:
    """Read one ``field=value`` render override, refusing anything else."""

    field, separator, value = text.partition("=")
    field = field.strip()
    if not separator or field not in OVERRIDE_FIELDS:
        allowed = ", ".join(sorted(OVERRIDE_FIELDS))
        raise ValueError(f"render override must be one of {allowed}=value, got {text!r}")
    try:
        return field, OVERRIDE_FIELDS[field](value)
    except (KeyError, ValueError) as error:
        raise ValueError(f"bad value for {field}: {value!r}") from error


def carried_share(shipped: float, diagnostic: float) -> float:
    """Share of the shipped flicker the removed term carried.

    Negative when removing the term made the flicker worse, which is a result
    and not an error: a term can be stabilising the image it sits on.
    """

    if shipped <= 0.0:
        return 0.0
    return (shipped - diagnostic) / shipped


def display_mean_map(pan: dict[str, Any]) -> np.ndarray:
    """Each pixel's mean display luminance over the frames that are predicted.

    Frame zero is only ever a source, never a prediction, so it is left out to
    match the pixels the shimmer score is built from.
    """

    return np.asarray(pan["display_luma"])[1:].mean(axis=0)


def mean_display_code(display_mean: np.ndarray, scored: np.ndarray) -> float:
    """Mean display luminance over the scored pixels, in code values."""

    mask = np.asarray(scored, dtype=bool)
    if not mask.any():
        return 0.0
    return float(np.asarray(display_mean)[mask].mean())


def _summary(
    run: dict[str, Any], display_mean: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, Any]:
    scored = run["scored"] if mask is None else run["scored"] & mask
    return {
        **stability(run, mask),
        "mean_display_code": mean_display_code(display_mean, scored),
        "pixels": int(scored.sum()),
    }


def attribute_view(
    view,
    display_mode: str,
    frames: int,
    record,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score the shipped pan once, then each diagnostic against it."""

    pan = render_pan(view, display_mode, frames, PIXELS_PER_FRAME)
    residuals = select_residuals(record, view.view_id, pan["gl"])
    shipped = score_run(pan, residuals)
    shipped_display_mean = display_mean_map(pan)
    shipped_summary = _summary(shipped, shipped_display_mean)
    gl, yaw_step = pan["gl"], pan["yaw_step_deg"]
    del pan

    results = []
    for diagnostic in diagnostics:
        variant = (
            shader_variant(diagnostic["replacements"])
            if diagnostic["replacements"]
            else nullcontext({})
        )
        with variant as served:
            ablated_pan = render_pan(
                view, display_mode, frames, PIXELS_PER_FRAME,
                render_overrides=diagnostic["render_overrides"] or None,
                supersample=diagnostic["supersample"],
            )
        ablated = score_run(ablated_pan, residuals)
        ablated_display_mean = display_mean_map(ablated_pan)
        changed = changed_pixel_mask(shipped["first_frame"], ablated["first_frame"])
        region = changed & shipped["scored"] & ablated["scored"]
        unstable = int(shipped["display_unstable_map"].sum())
        results.append(
            {
                "diagnostic_id": diagnostic["diagnostic_id"],
                "source": diagnostic["source"],
                "hypothesis": diagnostic["hypothesis"],
                "render_overrides": diagnostic["render_overrides"],
                "supersample": diagnostic["supersample"],
                "display_frame_comparable": diagnostic["supersample"] == 1,
                "shader_sha256": dict(served),
                "whole_frame": _summary(ablated, ablated_display_mean),
                "carried_share_of_display_excess": carried_share(
                    shipped_summary["display_excess"],
                    stability(ablated)["display_excess"],
                ),
                "changed_pixel_fraction": float(changed.mean()),
                "share_of_shipped_display_unstable_pixels_changed": (
                    float((shipped["display_unstable_map"] & changed).sum() / unstable)
                    if unstable else 0.0
                ),
                "within_changed_pixels": {
                    "shipped": _summary(shipped, shipped_display_mean, region),
                    "diagnostic": _summary(ablated, ablated_display_mean, region),
                },
            }
        )
        del ablated_pan, ablated
    return {
        "view_id": view.view_id,
        "gl": gl,
        "yaw_step_deg": yaw_step,
        "shipped": shipped_summary,
        "diagnostics": results,
    }


def _load_diagnostics(
    paths: list[Path], overrides: list[str], supersamples: list[int]
) -> list[dict[str, Any]]:
    diagnostics = []
    for path in paths:
        candidate = load_candidate(path)
        diagnostics.append(
            {
                "diagnostic_id": candidate.candidate_id,
                "source": portable_path(path),
                "hypothesis": candidate.hypothesis,
                "replacements": candidate.replacements,
                "render_overrides": {},
                "supersample": 1,
            }
        )
    for text in overrides:
        field, value = parse_override(text)
        diagnostics.append(
            {
                "diagnostic_id": f"render:{field}={value}",
                "source": "render config override",
                "hypothesis": f"Removing {field} shows how much shimmer that stage carries.",
                "replacements": [],
                "render_overrides": {field: value},
                "supersample": 1,
            }
        )
    for factor in supersamples:
        if factor < 2:
            raise ValueError(f"supersample must be at least 2, got {factor}")
        diagnostics.append(
            {
                "diagnostic_id": f"supersample:{factor}x",
                "source": "supersampled render",
                "hypothesis": (
                    f"Rendering {factor} times finer in each axis and box averaging "
                    "back bounds what antialiasing the scene could remove. Read the "
                    "linear HDR only: bloom and the glare tail are measured in "
                    "rendered pixels, so a finer grid narrows them on the scene and "
                    "the display frame is no longer the same transform."
                ),
                "replacements": [],
                "render_overrides": {},
                "supersample": factor,
            }
        )
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--diagnostic", action="append", type=Path, default=[])
    parser.add_argument("--render-override", action="append", default=[])
    parser.add_argument("--supersample", action="append", type=int, default=[])
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if not (arguments.diagnostic or arguments.render_override or arguments.supersample):
        parser.error(
            "give at least one --diagnostic, --render-override or --supersample"
        )
    try:
        diagnostics = _load_diagnostics(
            arguments.diagnostic, arguments.render_override, arguments.supersample
        )
    except ValueError as error:
        parser.error(str(error))

    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    views = [
        view for view in suite.views
        if not arguments.view or view.view_id in arguments.view
    ]
    record = load_known_residuals()
    results = []
    for view in views:
        print(f"attributing {view.view_id}", flush=True)
        results.append(
            attribute_view(
                view, suite.display_mode, arguments.frames, record, diagnostics
            )
        )
    report = {
        "schema_version": 1,
        "probe": "shimmer_attribution",
        "frames": arguments.frames,
        "pixels_per_frame_at_centre": PIXELS_PER_FRAME,
        "note": (
            "Diagnostics may change brightness; the detail guard is not run and "
            "mean display code is reported so a dimming is not read as a fix."
        ),
        "views": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
