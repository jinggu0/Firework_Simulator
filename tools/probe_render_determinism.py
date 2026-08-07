"""Find which render target and which pass a frame-to-frame difference comes from.

A capture that is not bit-reproducible tells you only that something moved.
This repeats one representative view with time frozen and reads every
intermediate target, so the first target that disagrees with its own earlier
output names the stage; then `--disable` removes one main-pass draw at a time,
so the configuration where the difference stops appearing names the pass.

Freezing time is the part that matters. With a live timestep the water surface
animates and every target drifts legitimately, which buries the thing being
looked for. Passing ``frame_dt_s = 0`` holds the scene still so any remaining
difference is the renderer disagreeing with itself.

Example::

    python -m tools.probe_render_determinism --view water_reflection
    python -m tools.probe_render_determinism --view water_reflection --disable scene
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Any

import numpy as np

from simulator.app import SimulatorApp
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)


#: Main-pass draws that can be removed without the renderer failing. The
#: reflection pre-pass is driven separately and always runs.
DISABLEABLE = (
    "sky",
    "land",
    "scene",
    "water",
    "ambient_occlusion",
    "haze",
    "particles",
)


def _read_colour(texture: Any) -> np.ndarray:
    width, height = texture.size
    raw = np.frombuffer(texture.read(), dtype=np.float16)
    return raw.reshape(height, width, texture.components).astype(np.float32)


def _read_depth(texture: Any) -> np.ndarray:
    width, height = texture.size
    raw = np.frombuffer(texture.read(), dtype=np.float32)
    return raw.reshape(height, width, 1)


def _targets(renderer: Any) -> dict[str, np.ndarray]:
    targets = renderer.targets
    return {
        "airlight": _read_colour(targets.airlight_texture),
        "ambient_occlusion": _read_colour(targets.ambient_occlusion_texture),
        "scene_depth": _read_depth(targets.scene_depth_texture),
        "reflection_depth": _read_depth(targets.reflection_depth),
        "reflection": _read_colour(targets.reflection_texture),
        "hdr": _read_colour(targets.hdr_texture),
    }


def _signature(reference: np.ndarray, candidate: np.ndarray) -> str | None:
    channels = min(3, candidate.shape[2])
    delta = np.abs(candidate[:, :, :channels] - reference[:, :, :channels])
    mask = delta.max(axis=2) > 0
    if not mask.any():
        return None
    rows, columns = np.nonzero(mask)
    return (
        f"{int(mask.sum())}px max{delta.max():.3e} "
        f"rows{rows.min()}-{rows.max()} cols{columns.min()}-{columns.max()}"
    )


def probe(view_id: str, iterations: int, disable: str | None) -> dict[str, Any]:
    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    view = next(item for item in suite.views if item.view_id == view_id)
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config, scenario_path=DEFAULT_SCENARIO_PATH)
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        app.renderer.post.set_mode(DisplayMode(suite.display_mode))
        view.apply(app.camera)
        renderer = app.renderer
        if disable:
            getattr(renderer, disable).draw = lambda *args, **kwargs: None

        first: dict[str, np.ndarray] | None = None
        signatures: dict[str, dict[str, int]] = {}
        differing: dict[str, int] = {}
        for _ in range(iterations):
            # Force the pre-pass every iteration; a static camera would
            # otherwise refresh it once and reuse the result thereafter.
            renderer.reflection_ready = False
            renderer.reflection_accumulator_s = 1e9
            renderer.reflection_sky_accumulator_s = 1e9
            renderer.render(app.world, app.camera, app.celestial, 0.0, None)
            app.ctx.finish()
            frames = _targets(renderer)
            if first is None:
                first = frames
                continue
            for name, frame in frames.items():
                signature = _signature(first[name], frame)
                if signature is None:
                    continue
                differing[name] = differing.get(name, 0) + 1
                bucket = signatures.setdefault(name, {})
                bucket[signature] = bucket.get(signature, 0) + 1
        return {
            "view_id": view_id,
            "iterations": iterations,
            "comparisons": max(iterations - 1, 0),
            "disabled_pass": disable,
            "differing_counts": differing,
            "signatures": signatures,
        }
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", default="water_reflection")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--disable",
        choices=DISABLEABLE,
        default=None,
        help="remove one main-pass draw to see whether the difference goes with it",
    )
    arguments = parser.parse_args()
    report = probe(arguments.view, arguments.iterations, arguments.disable)
    disabled = report["disabled_pass"] or "none"
    print(
        f"{report['view_id']}  disabled={disabled}  "
        f"comparisons={report['comparisons']}"
    )
    if not report["differing_counts"]:
        print("  every target bit-identical")
        return
    for name, count in sorted(
        report["differing_counts"].items(), key=lambda item: -item[1]
    ):
        print(f"  {name}: {count} differing")
        for signature, hits in sorted(
            report["signatures"][name].items(), key=lambda item: -item[1]
        ):
            print(f"      {hits:3d}x  {signature}")


if __name__ == "__main__":
    main()
