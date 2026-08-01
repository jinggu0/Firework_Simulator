"""V-22 aerial perspective, measured in a separate process.

``tools.measure_aerial_perspective`` opens a real GL context and a window.
Running it in a subprocess keeps the validation runner free of an OpenGL
dependency, matching how V-12 is handled, so the rest of the report still works
on a machine without a GPU.

Unlike V-12 this result is *not* machine specific: it compares a rendered frame
against a closed-form prediction of the same frame, so the tolerance is set by
the buffer's numeric precision rather than by the hardware's speed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import catalogue
from .report import MetricResult, MetricStatus

RELATIVE_TOLERANCE = 2.0e-3
"""Half-float quantum is 4.9e-4 and the prediction combines three stored values."""


def aerial_perspective(
    repository_root: Path | None = None, timeout_s: float = 600.0
) -> MetricResult:
    """Render with and without the atmosphere and gate the residual."""

    root = repository_root or Path(__file__).resolve().parent.parent.parent
    command = [sys.executable, "-m", "tools.measure_aerial_perspective"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return MetricResult(
            spec=catalogue.V22,
            status=MetricStatus.ERROR,
            message=f"measurement subprocess did not complete: {error}",
            detail={"command": command},
        )

    if completed.returncode != 0:
        return MetricResult(
            spec=catalogue.V22,
            status=MetricStatus.NO_REFERENCE,
            message=(
                f"measurement harness exited {completed.returncode}; "
                "an OpenGL context is required"
            ),
            detail={
                "command": command,
                "stderr_tail": completed.stderr.strip()[-2_000:],
            },
        )

    # The harness prints a pygame banner before its JSON payload.
    payload_start = completed.stdout.find("{")
    if payload_start < 0:
        return MetricResult(
            spec=catalogue.V22,
            status=MetricStatus.ERROR,
            message="measurement harness produced no JSON payload",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:]},
        )
    try:
        payload = json.loads(completed.stdout[payload_start:])
    except json.JSONDecodeError as error:
        return MetricResult(
            spec=catalogue.V22,
            status=MetricStatus.ERROR,
            message=f"could not parse measurement output: {error}",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:]},
        )

    error_max = float(payload.get("relative_error_max", float("nan")))
    sky_difference = float(
        payload.get("sky_absolute_difference_max", float("nan"))
    )
    transmittance_min = float(payload.get("transmittance_min", float("nan")))
    reflection_change = float(
        payload.get("reflection_mean_radiance_change", float("nan"))
    )
    reflection_sky_difference = float(
        payload.get("reflection_sky_absolute_difference_max", float("nan"))
    )
    water_residual = float(
        payload.get("bright_water_signed_residual", float("nan"))
    )
    # A vacuous pass is possible if nothing in frame is far enough to be hazed,
    # so the metric requires the frame to actually exercise the composite.
    exercised = transmittance_min < 0.98
    skies_untouched = sky_difference == 0.0 and reflection_sky_difference == 0.0
    # The reflected skyline reaches the eye by way of the water, so its path is
    # longer than the direct one and it must lose radiance. Signs, not
    # magnitudes: the mirrored path is where a sign error would hide, and it is
    # invisible in the residual because the CPU prediction models only the
    # direct path.
    reflection_dims = reflection_change < 0.0 and water_residual < 0.0
    passed = (
        error_max <= RELATIVE_TOLERANCE
        and skies_untouched
        and exercised
        and reflection_dims
    )
    if not exercised:
        message = (
            f"no geometry beyond {transmittance_min:.3f} transmittance; the "
            "frame does not exercise aerial perspective"
        )
    elif not reflection_dims:
        message = (
            f"the reflected skyline changed by {reflection_change:+.2%} and "
            f"bright water by {water_residual:+.2e}; hazing a longer path "
            "must remove radiance, so a non-negative change is a sign error"
        )
    else:
        message = (
            f"rendered frame within {error_max:.2e} of the predicted "
            f"composite at {payload.get('visibility_km', float('nan')):.1f} km "
            f"visibility; reflected skyline {reflection_change:+.2%}; both "
            f"skies unchanged"
        )
    return MetricResult(
        spec=catalogue.V22,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=message,
        residuals={
            "relative_error_max": error_max,
            "relative_error_p99": float(
                payload.get("relative_error_p99", float("nan"))
            ),
            "relative_error_mean": float(
                payload.get("relative_error_mean", float("nan"))
            ),
            "tolerance": RELATIVE_TOLERANCE,
            "sky_absolute_difference_max": sky_difference,
            "reflection_sky_absolute_difference_max": reflection_sky_difference,
            "reflection_mean_radiance_change": reflection_change,
            "reflection_worst_radiance_change": float(
                payload.get("reflection_worst_radiance_change", float("nan"))
            ),
            "bright_water_signed_residual": water_residual,
            "transmittance_min": transmittance_min,
            "transmittance_p50": float(
                payload.get("transmittance_p50", float("nan"))
            ),
            "visibility_km": float(payload.get("visibility_km", float("nan"))),
            "mean_radiance_change": float(
                payload.get("mean_radiance_change", float("nan"))
            ),
        },
        detail={
            "geometry_fraction": payload.get("geometry_fraction"),
            "water_fraction_of_geometry": payload.get(
                "water_fraction_of_geometry"
            ),
            "reflection_geometry_fraction": payload.get(
                "reflection_geometry_fraction"
            ),
            "gated_population": (
                "opaque non-water pixels; water carries a reflection with its "
                "own atmospheric path, so the two-render solve does not "
                "isolate its object radiance and it is checked by sign instead"
            ),
            "path_p50_m": payload.get("path_p50_m"),
            "path_p95_m": payload.get("path_p95_m"),
            "surface_extinction_per_m": payload.get("surface_extinction_per_m"),
            "airlight_rgb": payload.get("airlight_rgb"),
            "camera_position_eus_m": payload.get("camera_position_eus_m"),
            "visibility_is_modelled": True,
            "note": (
                "Visibility is what the aerosol model implies, not an "
                "observation; the event weather record carries none."
            ),
        },
    )
