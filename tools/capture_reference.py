"""Capture the linear HDR frame, for proving a refactor changed nothing.

A refactor that preserves behaviour must produce the same pixels. This renders a
fixed scene deterministically and writes the linear RGBA16F buffer — before
exposure and tone mapping — so a before/after comparison measures the renderer
rather than the tone mapper or the sensor noise model.

    python -m tools.capture_reference --output before.npy
    python -m tools.capture_reference --output after.npy
    python -m tools.capture_reference --compare before.npy after.npy

The scene is populated the same way ``tools/profile_runtime.py`` populates it:
a shell is flown to its burst, star lifetimes are pinned so the field does not
decay between runs, and the fluid is stepped a fixed number of times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import (
    linear_hdr_statistics,
    read_linear_hdr,
    save_linear_hdr,
)

from dataclasses import replace


def _prepare(scenario_path: Path) -> SimulatorApp:
    base = SimulationConfig()
    config = replace(
        base,
        render=replace(base.render, vsync=False, target_fps=0),
    )
    app = SimulatorApp(config, scenario_path=scenario_path)
    physics_dt = 1.0 / config.render.physics_hz
    for _ in range(int(3.1 / physics_dt)):
        if app.environment is not None:
            app.world.atmosphere = app.environment.sample(app.event_timestamp)
        app.clock.advance_steps(1)
        app.world.update(physics_dt)
        for burst in app.world.consume_burst_events():
            app.smoke.inject_burst(
                burst.position_m,
                burst.smoke_mass_kg,
                burst.post_blast_thermal_energy_j,
            )
    count = app.world.stars.count
    if count:
        # Pin the burn so the field is identical at every capture.
        app.world.stars.lifetime_s[:count] = 100.0
        app.world.stars.age_s[:count] = 0.2
    for _ in range(4):
        app.smoke.step(1.0 / app.smoke.update_hz)
    return app


def capture(scenario_path: Path, frames: int) -> tuple[np.ndarray, dict]:
    app = _prepare(scenario_path)
    frame_dt = 1.0 / 60.0
    for _ in range(frames):
        app.renderer.render(
            app.world, app.camera, app.celestial, frame_dt, app.smoke
        )
    app.ctx.finish()
    frame = read_linear_hdr(app.renderer)
    context = {
        "frames": frames,
        "stars": int(app.world.stars.count),
        "fluid_backend": getattr(app.smoke, "backend_name", "cpu"),
        "star_catalogue_measured": bool(
            getattr(app.renderer, "star_catalogue_is_measured", False)
        ),
        "resolution": [int(frame.shape[1]), int(frame.shape[0])],
    }
    app.audio_executor.shutdown(wait=True, cancel_futures=True)
    pygame.quit()
    return frame, context


def compare(first: Path, second: Path) -> dict:
    a = np.load(first).astype(np.float64)
    b = np.load(second).astype(np.float64)
    if a.shape != b.shape:
        return {"identical": False, "reason": f"shape {a.shape} vs {b.shape}"}
    difference = np.abs(a - b)
    finite = np.isfinite(a) & np.isfinite(b)
    scale = np.maximum(np.abs(a), np.abs(b))
    relative = np.where(scale > 0.0, difference / np.maximum(scale, 1e-30), 0.0)
    changed = int(np.count_nonzero(difference[finite]))
    return {
        "identical": changed == 0,
        "differing_components": changed,
        "total_components": int(a.size),
        "max_absolute_difference": float(difference[finite].max())
        if finite.any()
        else float("nan"),
        "max_relative_difference": float(relative[finite].max())
        if finite.any()
        else float("nan"),
        "mean_absolute_difference": float(difference[finite].mean())
        if finite.any()
        else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture or compare a linear HDR reference frame."
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs=2,
        default=None,
        metavar=("BEFORE", "AFTER"),
        help="Compare two captures and exit non-zero if they differ.",
    )
    args = parser.parse_args()

    if args.compare is not None:
        result = compare(*args.compare)
        print(json.dumps(result, indent=2))
        return 0 if result["identical"] else 1

    if args.output is None:
        parser.error("--output is required unless --compare is given")
    frame, context = capture(args.scenario, args.frames)
    path = save_linear_hdr(frame, args.output)
    print(
        json.dumps(
            {
                "output": str(path),
                **context,
                "statistics": linear_hdr_statistics(frame),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
