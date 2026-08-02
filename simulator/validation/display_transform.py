"""V-23 display transform, measured in a separate process.

``tools.measure_display_transform`` opens a real GL context and a window.
Running it in a subprocess keeps the validation runner free of an OpenGL
dependency, matching V-12 and V-22.

This is the one metric in the suite that reads the **display-referred** image,
and deliberately so: the shader it checks is the stage that produces that
image, so there is nothing else to read. Every colour and brightness metric
still reads the linear buffer upstream of it, which is why a change to the
white balance or the tone curve cannot flatter any of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import catalogue
from .report import MetricResult, MetricStatus

DISPLAY_QUANTUM = 1.0 / 255.0
ABSOLUTE_TOLERANCE = 2.0 * DISPLAY_QUANTUM
"""Two 8-bit code values, for a texel-aligned sampling.

Rounding alone puts the mean absolute error at a quarter of a code value and
the maximum near a half. Measured worst case is 0.63, so this bound has about
three times the margin over rounding while sitting far below any real defect:
dropping the blue white-balance gain of 1.64 would move blue by tens of code
values.
"""

RESAMPLED_TOLERANCE = 4.0 * DISPLAY_QUANTUM
"""Four code values, once a lens calibration makes the sampling non-aligned.

With an identity lens the source coordinate equals the output coordinate, so
every fetch lands on a texel centre and the GPU's bilinear filter degenerates
to an exact read — hence the 0.63 floor. Any distortion turns it into a real
interpolation, and the CPU reference and the texture unit then disagree by up
to a measured 2.26 code values. That is filtering precision rather than a model
difference: a pincushion lens, which needs **no** overscan and no change of
resolution, shows the same rise. The bound is set at 1.8x the measured floor.

Revisit if a calibration ever exceeds it: the alternative is for the display
transform to do its own bilinear from four explicit fetches, the way the
peripheral blur now does its own mip interpolation.
"""

MINIMUM_LIT_FRACTION = 0.05
"""A black frame would agree trivially, so the frame has to contain an image."""


def display_transform(
    repository_root: Path | None = None, timeout_s: float = 600.0
) -> MetricResult:
    """Reproduce the rendered display transform on the CPU and gate the residual."""

    root = repository_root or Path(__file__).resolve().parent.parent.parent
    command = [sys.executable, "-m", "tools.measure_display_transform"]
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
            spec=catalogue.V23,
            status=MetricStatus.ERROR,
            message=f"measurement subprocess did not complete: {error}",
            detail={"command": command},
        )

    if completed.returncode != 0:
        return MetricResult(
            spec=catalogue.V23,
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

    payload_start = completed.stdout.find("{")
    if payload_start < 0:
        return MetricResult(
            spec=catalogue.V23,
            status=MetricStatus.ERROR,
            message="measurement harness produced no JSON payload",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:]},
        )
    try:
        payload = json.loads(completed.stdout[payload_start:])
    except json.JSONDecodeError as error:
        return MetricResult(
            spec=catalogue.V23,
            status=MetricStatus.ERROR,
            message=f"could not parse measurement output: {error}",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:]},
        )

    error_max = float(payload.get("absolute_error_max", float("nan")))
    lit_fraction = float(payload.get("lit_pixel_fraction", 0.0))
    # A distorted lens resamples, and resampling costs filtering precision.
    tolerance = (
        ABSOLUTE_TOLERANCE
        if payload.get("distortion_is_identity", True)
        else RESAMPLED_TOLERANCE
    )
    overscan = float(payload.get("overscan", 1.0))
    required_overscan = float(payload.get("required_overscan", 1.0))
    residual = float(payload.get("distortion_inverse_residual", float("nan")))
    coverage = float(payload.get("distortion_frame_coverage", 0.0))
    gains = [float(value) for value in payload.get("white_balance_gain", [])]

    exercised = lit_fraction >= MINIMUM_LIT_FRACTION
    # A calibration whose inversion has not converged, or one that samples
    # outside the rendered frame, would make the image wrong in a way the
    # residual against a CPU model using the same inversion cannot see.
    inversion_converged = residual <= 1e-4
    fully_covered = coverage >= 1.0 - 1e-9
    balanced = len(gains) == 3 and all(value > 0.0 for value in gains)
    # The renderer has to have actually widened the field by what the lens
    # needs; a coverage of 1.0 means nothing if it was measured against a
    # render that never happened.
    overscan_applied = abs(overscan - required_overscan) <= 1e-9
    passed = (
        error_max <= tolerance
        and overscan_applied
        and exercised
        and inversion_converged
        and fully_covered
        and balanced
    )

    if not exercised:
        message = (
            f"only {lit_fraction:.1%} of the frame is above the display floor; "
            "the frame does not exercise the transform"
        )
    elif not inversion_converged:
        message = (
            f"lens distortion inverse residual {residual:.2e} exceeds 1e-4; "
            "the fixed-point inversion has not converged for these coefficients"
        )
    elif not overscan_applied:
        message = (
            f"the lens needs {required_overscan:.4f} overscan but the scene "
            f"was rendered at {overscan:.4f}; the frame edges would be clamped"
        )
    elif not fully_covered:
        message = (
            f"only {coverage:.1%} of the output samples inside the rendered "
            "frame; this calibration needs overscan"
        )
    else:
        message = (
            f"rendered frame within {error_max / DISPLAY_QUANTUM:.2f} display "
            f"code values of the predicted transform at "
            f"{payload.get('white_balance_temperature_k', float('nan')):.0f} K "
            "white balance"
        )

    return MetricResult(
        spec=catalogue.V23,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=message,
        residuals={
            "absolute_error_max": error_max,
            "absolute_error_p999": float(
                payload.get("absolute_error_p999", float("nan"))
            ),
            "absolute_error_mean": float(
                payload.get("absolute_error_mean", float("nan"))
            ),
            "tolerance": tolerance,
            "overscan": overscan,
            "required_overscan": required_overscan,
            "coverage_without_overscan": float(
                payload.get("coverage_without_overscan", float("nan"))
            ),
            "error_in_code_values": error_max / DISPLAY_QUANTUM,
            "distortion_inverse_residual": residual,
            "distortion_frame_coverage": coverage,
            "lit_pixel_fraction": lit_fraction,
        },
        detail={
            "white_balance_temperature_k": payload.get(
                "white_balance_temperature_k"
            ),
            "white_balance_gain": gains,
            "distortion_is_identity": payload.get("distortion_is_identity"),
            "sensor_noise_enabled": payload.get("sensor_noise_enabled"),
            "display_referred": True,
            "note": (
                "The only metric that reads the 8-bit output, because the "
                "stage under test is what produces it. Sensor noise is "
                "disabled: its distribution is the claim, not its per-pixel "
                "value. White balance temperature is an operator setting, not "
                "a measurement."
            ),
        },
    )
