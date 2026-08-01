"""V-24 observer transform, measured in a separate process.

The sibling of V-23. Where that one checks the camera's display transform, this
checks the observer's — pupil gain, local adaptation, chromatic adaptation, the
mesopic mix, ACES — against the same kind of CPU prediction from the same
readable buffers.

Two of the observer stages are **excluded and reported as unverified**:
peripheral acuity samples a per-pixel mip level and the glare tail reads a
reduced mip of the bloom, so predicting either in NumPy would measure GPU mip
generation rather than the model. The harness switches both off, which isolates
the colour path — the part chromatic adaptation changed.
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
"""Two 8-bit code values, on the same reasoning as V-23."""

MINIMUM_LIT_FRACTION = 0.05
"""A black frame would agree trivially, so the frame has to contain an image."""

LUMINANCE_TOLERANCE = 2.0e-3
"""Half-float storage of a unit-luminance white admits about this much drift."""


def observer_transform(
    repository_root: Path | None = None, timeout_s: float = 900.0
) -> MetricResult:
    """Reproduce the rendered observer transform on the CPU and gate it."""

    root = repository_root or Path(__file__).resolve().parent.parent.parent
    command = [sys.executable, "-m", "tools.measure_observer_transform"]
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
            spec=catalogue.V24,
            status=MetricStatus.ERROR,
            message=f"measurement subprocess did not complete: {error}",
            detail={"command": command},
        )

    if completed.returncode != 0:
        return MetricResult(
            spec=catalogue.V24,
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
            spec=catalogue.V24,
            status=MetricStatus.ERROR,
            message="measurement harness produced no JSON payload",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:]},
        )
    try:
        payload = json.loads(completed.stdout[payload_start:])
    except json.JSONDecodeError as error:
        return MetricResult(
            spec=catalogue.V24,
            status=MetricStatus.ERROR,
            message=f"could not parse measurement output: {error}",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:]},
        )

    error_max = float(payload.get("absolute_error_max", float("nan")))
    lit_fraction = float(payload.get("lit_pixel_fraction", 0.0))
    degree = float(payload.get("chromatic_degree", float("nan")))
    spread = float(payload.get("adapting_white_spatial_spread", float("nan")))
    white_luminance = float(
        payload.get("adapting_white_luminance", float("nan"))
    )

    exercised = lit_fraction >= MINIMUM_LIT_FRACTION
    # CIECAM02 D is never 0 or 1 in a real viewing condition; either bound
    # means the relation collapsed rather than that the observer is unusual.
    degree_is_partial = 0.0 < degree < 1.0
    # The adapting white is global by construction — pooled from the top of the
    # mip chain — and any spatial structure means that pooling stopped covering
    # the field.
    white_is_global = spread == 0.0
    # Unit luminance is what makes the von Kries step luminance-preserving.
    white_is_normalised = abs(white_luminance - 1.0) <= LUMINANCE_TOLERANCE
    passed = (
        error_max <= ABSOLUTE_TOLERANCE
        and exercised
        and degree_is_partial
        and white_is_global
        and white_is_normalised
    )

    if not exercised:
        message = (
            f"only {lit_fraction:.1%} of the frame is above the display floor; "
            "the frame does not exercise the transform"
        )
    elif not degree_is_partial:
        message = (
            f"degree of adaptation {degree:.4f} is not strictly between 0 and "
            "1; the CIECAM02 relation has collapsed"
        )
    elif not white_is_global:
        message = (
            f"the adapting white varies across the frame by {spread:.2e}; it "
            "is pooled from the top of the mip chain and must be global"
        )
    elif not white_is_normalised:
        message = (
            f"the adapting white carries luminance {white_luminance:.4f} "
            "rather than 1, so the adaptation is not luminance preserving"
        )
    else:
        message = (
            f"rendered frame within {error_max / DISPLAY_QUANTUM:.2f} display "
            f"code values of the predicted transform; D {degree:.3f}, cone "
            f"fraction {payload.get('cone_fraction', float('nan')):.3f}, "
            f"effective chromatic shift "
            f"{payload.get('effective_chromatic_shift', float('nan')):.2%}"
        )

    return MetricResult(
        spec=catalogue.V24,
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
            "tolerance": ABSOLUTE_TOLERANCE,
            "error_in_code_values": error_max / DISPLAY_QUANTUM,
            "chromatic_degree": degree,
            "chromatic_gain_spread": float(
                payload.get("chromatic_gain_spread", float("nan"))
            ),
            "effective_chromatic_shift": float(
                payload.get("effective_chromatic_shift", float("nan"))
            ),
            "cone_fraction": float(payload.get("cone_fraction", float("nan"))),
            "adapting_white_luminance": white_luminance,
        },
        detail={
            "adapting_luminance_cd_m2": payload.get("adapting_luminance_cd_m2"),
            "adapting_white": payload.get("adapting_white"),
            "adapting_white_spatial_spread": spread,
            "display_referred": True,
            "unverified_stages": (
                "peripheral acuity and the glare tail are switched off for the "
                "measurement; both are spatial and predicting them would "
                "measure GPU mip generation rather than the model"
            ),
        },
    )
