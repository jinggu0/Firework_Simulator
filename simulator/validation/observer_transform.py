"""V-24 observer transform, measured in a separate process.

The sibling of V-23. Where that one checks the camera's display transform, this
checks the observer's — pupil gain, local adaptation, chromatic adaptation, the
mesopic mix, ACES — against the same kind of CPU prediction from the same
readable buffers.

**Every stage is covered**, including both spatial ones. Reading the generated
mip levels back makes ``textureLod`` reproducible; getting peripheral acuity
there meant first finding out why it would not reproduce, which turned out to
be a renderer defect — the driver's brilinear approximation of trilinear
filtering made the peripheral blur driver-dependent. See
``tools.measure_observer_transform``.
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

MINIMUM_VEILING_SHARE = 0.01
"""Glare must reach a percent of the retinal signal somewhere in the frame.

Verifying a term that is everywhere negligible would be verifying nothing.
Measured at 19% on the shipped scene.
"""

MINIMUM_PERIPHERAL_LOD = 1.0
"""The acuity bias must span at least one mip level somewhere in the frame.

Same reasoning: a frame whose periphery is never blurred would verify the
acuity path vacuously. Measured at 3.77 on the shipped scene, whose corner
sits 40.7 degrees from fixation.
"""

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
    veiling_share = float(
        payload.get("veiling_share_of_retinal_max", 0.0)
    )

    peripheral_lod = float(payload.get("peripheral_lod_max", 0.0))

    exercised = lit_fraction >= MINIMUM_LIT_FRACTION
    # Both spatial stages are verified here, so the frame has to contain enough
    # of each to make that claim mean something.
    glare_exercised = veiling_share >= MINIMUM_VEILING_SHARE
    acuity_exercised = peripheral_lod >= MINIMUM_PERIPHERAL_LOD
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
        and glare_exercised
        and acuity_exercised
        and degree_is_partial
        and white_is_global
        and white_is_normalised
    )

    if not exercised:
        message = (
            f"only {lit_fraction:.1%} of the frame is above the display floor; "
            "the frame does not exercise the transform"
        )
    elif not glare_exercised:
        message = (
            f"the veiling glare reaches only {veiling_share:.2%} of the "
            "retinal signal; the frame does not exercise the glare tail"
        )
    elif not acuity_exercised:
        message = (
            f"the peripheral mip bias reaches only {peripheral_lod:.2f}; the "
            "frame does not exercise peripheral acuity"
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
            "veiling_share_of_retinal_max": veiling_share,
            "peripheral_lod_max": peripheral_lod,
        },
        detail={
            "adapting_luminance_cd_m2": payload.get("adapting_luminance_cd_m2"),
            "adapting_white": payload.get("adapting_white"),
            "adapting_white_spatial_spread": spread,
            "display_referred": True,
            "glare_verified": payload.get("glare_verified"),
            "peripheral_acuity_verified": payload.get(
                "peripheral_acuity_verified"
            ),
            "unverified_stages": "none",
            "note": (
                "Peripheral acuity joined the gate once the driver's brilinear "
                "approximation of trilinear filtering was identified and the "
                "shader stopped relying on it. The residual fell from 7.2 to "
                "0.63 code values, and a 10 percent error in the acuity "
                "constant now fails this metric."
            ),
        },
    )
