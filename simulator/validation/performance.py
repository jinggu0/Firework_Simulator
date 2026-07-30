"""V-12 frame budget, measured in a separate process.

``tools.profile_runtime`` opens a real GL context and a window. Running it in a
subprocess keeps the validation runner free of an OpenGL dependency, so the rest
of the report still works on a machine or CI agent without a GPU.

The result records machine, platform, backend, frame count, and date. A frame
time without that context is a historical log entry, not a specification, and
must not be used as a regression baseline across machines.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import catalogue
from .report import MetricResult, MetricStatus

FRAME_BUDGET_MS = 1_000.0 / 60.0
"""16.67 ms, the 60 Hz display period."""


def _machine_context(backend: str, frames: int) -> dict[str, object]:
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "node": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "fluid_backend_requested": backend,
        "frames": frames,
    }


def frame_budget(
    frames: int = 240,
    fluid_backend: str = "3d",
    repository_root: Path | None = None,
    timeout_s: float = 900.0,
) -> MetricResult:
    """Run the profiling harness and gate the integrated frame p95."""

    root = repository_root or Path(__file__).resolve().parent.parent.parent
    command = [
        sys.executable,
        "-m",
        "tools.profile_runtime",
        "--frames",
        str(frames),
        "--fluid-backend",
        fluid_backend,
        "--integrated-only",
    ]
    context = _machine_context(fluid_backend, frames)
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
            spec=catalogue.V12,
            status=MetricStatus.ERROR,
            message=f"profiling subprocess did not complete: {error}",
            detail={"command": command, **context},
        )

    if completed.returncode != 0:
        # A GL context failure is a missing capability, not a regression: the
        # metric declares that it requires OpenGL.
        return MetricResult(
            spec=catalogue.V12,
            status=MetricStatus.NO_REFERENCE,
            message=(
                "profiling harness exited "
                f"{completed.returncode}; an OpenGL context is required"
            ),
            detail={
                "command": command,
                "stderr_tail": completed.stderr.strip()[-2_000:],
                **context,
            },
        )

    # The harness prints a pygame banner before its JSON payload.
    payload_start = completed.stdout.find("{")
    if payload_start < 0:
        return MetricResult(
            spec=catalogue.V12,
            status=MetricStatus.ERROR,
            message="profiling harness produced no JSON payload",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:], **context},
        )
    try:
        payload = json.loads(completed.stdout[payload_start:])
    except json.JSONDecodeError as error:
        return MetricResult(
            spec=catalogue.V12,
            status=MetricStatus.ERROR,
            message=f"could not parse profiling output: {error}",
            detail={"stdout_tail": completed.stdout.strip()[-2_000:], **context},
        )

    integrated = payload.get("integrated", {})
    frame_p95_ms = float(integrated.get("frame_p95_ms", float("nan")))
    passed = frame_p95_ms < FRAME_BUDGET_MS
    return MetricResult(
        spec=catalogue.V12,
        status=MetricStatus.PASS if passed else MetricStatus.FAIL,
        message=(
            f"frame p95 {frame_p95_ms:.3f} ms against a "
            f"{FRAME_BUDGET_MS:.2f} ms budget on {platform.node()} "
            f"({integrated.get('fluid_backend', 'unknown')} backend)"
        ),
        residuals={
            "frame_mean_ms": float(integrated.get("frame_mean_ms", float("nan"))),
            "frame_p95_ms": frame_p95_ms,
            "frame_p99_ms": float(integrated.get("frame_p99_ms", float("nan"))),
            "physics_p95_ms": float(
                integrated.get("physics_p95_ms", float("nan"))
            ),
            "visual_p95_ms": float(integrated.get("visual_p95_ms", float("nan"))),
            "budget_ms": FRAME_BUDGET_MS,
            "margin_ms": FRAME_BUDGET_MS - frame_p95_ms,
        },
        detail={
            "fluid_backend_selected": integrated.get("fluid_backend", ""),
            "machine_specific": True,
            "note": (
                "Comparable only against a run on this machine with the same "
                "backend, resolution, and frame count."
            ),
            **context,
        },
    )
