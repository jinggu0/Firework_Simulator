"""Measure the rendered observer transform against its CPU reference (V-24).

``human_vision.frag`` is the display transform for the observer, and chromatic
adaptation made it a colour pipeline rather than only a brightness one: cone
gains taking the field's average chromaticity to the display white, applied to
the cone signal before the mesopic mix. A cone matrix transposed, or the gains
applied to the rod path as well, would produce a plausible image and no error.

The check is the same one V-23 makes of the camera: the linear HDR, bloom, and
adaptation buffers are the shader's only image inputs, so running the same
chain in NumPy predicts the frame the GPU should have produced.

**The spatial path is switched off for the measurement.** Peripheral acuity
samples a per-pixel mip level and the glare tail reads a reduced mip of the
bloom; reproducing GPU mip generation in NumPy would measure the filter rather
than the model. Setting the maximum blur to zero and the glare constant to zero
removes both and isolates the colour path, which is what changed. Those two
stages remain unverified and are reported as such.

Prints a JSON payload; ``simulator.validation.display_transform`` and its
sibling run these harnesses in a subprocess so the rest of the report still
works without a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.color import chromatic_adaptation_gains
from simulator.config import SimulationConfig
from simulator.human_vision import HumanVisionState
from simulator.passes.post import DisplayMode
from tools.measure_aerial_perspective import _bilinear, _read_rgb

PHOTOPIC_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

ADAPTATION_CONVERGENCE_S = 90.0
"""Simulated seconds run before measuring.

Chromatic adaptation has a 26 s time constant, so a frame measured at startup
would be testing the initial condition rather than the model.
"""


def _flatten_spatial_stages() -> None:
    """Remove peripheral blur and glare from the shader's uniforms.

    A harness-only override. Both stages are spatial, both need GPU mip
    generation to predict, and neither is what chromatic adaptation changed.
    """

    original = HumanVisionState.uniforms

    def uniforms(self):
        values = original(self)
        values["maximum_blur_lod"] = 0.0
        values["glare_constant"] = 0.0
        return values

    HumanVisionState.uniforms = uniforms


def _aces(x: np.ndarray) -> np.ndarray:
    return np.clip(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0
    )


def _predict(
    hdr: np.ndarray,
    bloom: np.ndarray,
    adaptation: np.ndarray,
    vision: HumanVisionState,
    bloom_strength: float,
) -> np.ndarray:
    height, width = hdr.shape[:2]
    u = np.broadcast_to((np.arange(width) + 0.5) / width, (height, width))
    v = np.broadcast_to(
        ((np.arange(height) + 0.5) / height)[:, None], (height, width)
    )
    glare = _bilinear(bloom, u, v) * bloom_strength
    state = _bilinear(adaptation, u, v)
    adapting_white = state[..., :3]
    local_luminance = state[..., 3]

    retinal = (hdr + glare) * vision.pupil_gain
    reference = np.maximum(
        local_luminance * 0.65 + vision.adapting_luminance_cd_m2 * 0.35, 1e-6
    )
    normalized = retinal / (reference[..., None] * 6.0 + 1e-5)

    # The gains vary per pixel only through the adapting white, which is
    # global, so this is one 3x3 per frame in practice; it is evaluated per
    # pixel anyway so a future local white would need no change here.
    gains = np.stack(
        [
            chromatic_adaptation_gains(white, vision.chromatic_degree)
            for white in adapting_white.reshape(-1, 3)
        ]
    ).reshape(adapting_white.shape)
    from simulator.color import (
        CAT02_LMS_TO_LINEAR_SRGB,
        LINEAR_SRGB_TO_CAT02_LMS,
    )

    cone = normalized @ LINEAR_SRGB_TO_CAT02_LMS.T
    adapted = (cone * gains) @ CAT02_LMS_TO_LINEAR_SRGB.T

    luminance = normalized @ PHOTOPIC_WEIGHTS
    mesopic = luminance[..., None] + (
        adapted - luminance[..., None]
    ) * vision.cone_fraction
    return _aces(np.maximum(mesopic, 0.0)) ** (1.0 / 2.2)


def measure(convergence_s: float = ADAPTATION_CONVERGENCE_S) -> dict[str, object]:
    _flatten_spatial_stages()
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config)
    renderer = app.renderer
    renderer.post.set_mode(DisplayMode.HUMAN_VISION)
    step_s = 1.5
    for _ in range(int(convergence_s / step_s)):
        renderer.render(app.world, app.camera, app.celestial, step_s)

    renderer.render(app.world, app.camera, app.celestial, 0.0)
    app.ctx.finish()
    width, height = app.ctx.screen.size
    displayed = np.frombuffer(
        app.ctx.screen.read(components=3), dtype=np.uint8
    ).reshape(height, width, 3).astype(np.float64) / 255.0
    hdr = _read_rgb(renderer.targets.hdr_texture)
    bloom = _read_rgb(renderer.targets.bloom_textures[0])
    adaptation_texture = renderer.targets.current_adaptation
    adaptation = np.frombuffer(
        adaptation_texture.read(), dtype=np.float16
    ).reshape(
        adaptation_texture.size[1], adaptation_texture.size[0], 4
    ).astype(np.float64)
    pygame.display.flip()

    vision = renderer.post.vision
    predicted = _predict(
        hdr, bloom, adaptation, vision, config.render.bloom_strength
    )
    error = np.abs(predicted - displayed)

    white = adaptation[..., :3].reshape(-1, 3)
    mean_white = white.mean(axis=0)
    gains = chromatic_adaptation_gains(mean_white, vision.chromatic_degree)
    lit = displayed.max(axis=-1) > 4.0 / 255.0
    result = {
        "adapting_luminance_cd_m2": vision.adapting_luminance_cd_m2,
        "cone_fraction": vision.cone_fraction,
        "chromatic_degree": vision.chromatic_degree,
        "adapting_white": [float(value) for value in mean_white],
        "adapting_white_spatial_spread": float(white.std(axis=0).max()),
        # Unit luminance is what makes the adaptation luminance-preserving, so
        # it is checked rather than assumed.
        "adapting_white_luminance": float(mean_white @ PHOTOPIC_WEIGHTS),
        "chromatic_gains": [float(value) for value in gains],
        "chromatic_gain_spread": float(gains.max() / max(gains.min(), 1e-12)),
        "effective_chromatic_shift": float(
            (gains.max() / max(gains.min(), 1e-12) - 1.0) * vision.cone_fraction
        ),
        "absolute_error_max": float(error.max()),
        "absolute_error_mean": float(error.mean()),
        "absolute_error_p999": float(np.percentile(error, 99.9)),
        "display_quantum": 1.0 / 255.0,
        "lit_pixel_fraction": float(lit.mean()),
        "spatial_stages_disabled": True,
    }
    app.audio_executor.shutdown(wait=True, cancel_futures=True)
    pygame.quit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the rendered observer transform with its CPU reference."
    )
    parser.add_argument(
        "--convergence-seconds", type=float, default=ADAPTATION_CONVERGENCE_S
    )
    args = parser.parse_args()
    print(json.dumps(measure(args.convergence_seconds), indent=2))


if __name__ == "__main__":
    main()
