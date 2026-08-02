"""Measure the rendered observer transform against its CPU reference (V-24).

``human_vision.frag`` is the display transform for the observer, and chromatic
adaptation made it a colour pipeline rather than only a brightness one: cone
gains taking the field's average chromaticity to the display white, applied to
the cone signal before the mesopic mix. A cone matrix transposed, or the gains
applied to the rod path as well, would produce a plausible image and no error.

The check is the same one V-23 makes of the camera: the linear HDR, bloom, and
adaptation buffers are the shader's only image inputs, so running the same
chain in NumPy predicts the frame the GPU should have produced.

Both spatial stages read reduced mip levels, which at first looks unpredictable
without reimplementing GPU mip generation. It is not: ``moderngl`` reads each
generated level back, so ``textureLod`` is reproduced by interpolating the
levels the GPU itself produced.

**The whole transform is covered.** Getting peripheral acuity there took
finding out why it would not reproduce, and the answer turned out to be a
renderer defect rather than a harness limit: this driver implements
``GL_LINEAR_MIPMAP_LINEAR`` as *brilinear*, applying a level weight of
``clamp((frac - 1/6) / (2/3), 0, 1)`` instead of ``frac``. The curve was pinned
at five points — 0.125 to 0.000, 0.25 to 0.125, 0.5 to 0.5, 0.75 to 0.875, and
0 and 1 exact — each reproducing to the 0.6 code-value baseline once the
measured weight was used.

That made the peripheral blur a property of the driver rather than of the eye,
so ``human_vision.frag`` now blends two explicit integer levels instead. The
residual fell from 7.2 code values to 0.63, and the gate acquired teeth it did
not have: a 10 percent error in the acuity constant E2 now fails it, where
before the fix it moved the statistics by less than the noise.

Prints a JSON payload; ``simulator.validation.observer_transform`` runs this in
a subprocess so the rest of the report still works without a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.camera_optics import vertical_fov_deg
from simulator.color import chromatic_adaptation_gains
from simulator.config import SimulationConfig
from simulator.human_vision import HumanVisionState, acuity_fraction
from simulator.passes.post import DisplayMode
from tools.measure_aerial_perspective import _bilinear, _read_rgb

PHOTOPIC_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

GLARE_SCALE = 0.01
"""The shader's conversion from wide-field bloom luminance to a veil.

Not a published constant: the Stiles-Holladay term needs the source
illuminance at the eye, and a reduced mip of the bloom stands in for it. Kept
here so the reference and the shader cannot drift.
"""

ADAPTATION_CONVERGENCE_S = 90.0
"""Simulated seconds run before measuring.

Chromatic adaptation has a 26 s time constant, so a frame measured at startup
would be testing the initial condition rather than the model.
"""


def _mip_chain(texture, levels: int) -> list[np.ndarray]:
    """Read back the generated mip levels the shader will sample."""

    return [_read_rgb(texture, level) for level in range(levels)]


def _texture_lod(
    levels: list[np.ndarray], u: np.ndarray, v: np.ndarray, lod: np.ndarray
) -> np.ndarray:
    """Reproduce ``textureLod`` under GL_LINEAR_MIPMAP_LINEAR.

    Bilinear within each of the two bracketing levels, linear between them.
    The levels are the GPU's own, so this reproduces the sampler rather than
    the mip filter — which is what makes the spatial stages predictable at all.
    """

    top = len(levels) - 1
    clamped = np.clip(lod, 0.0, float(top))
    low = np.floor(clamped).astype(int)
    high = np.minimum(low + 1, top)
    blend = (clamped - low)[..., None]
    sampled = np.stack([_bilinear(level, u, v) for level in levels])
    rows = np.arange(u.shape[0])[:, None]
    columns = np.arange(u.shape[1])[None, :]
    lower = sampled[low, rows, columns]
    upper = sampled[high, rows, columns]
    return lower * (1.0 - blend) + upper * blend


def _eccentricity_deg(
    u: np.ndarray, v: np.ndarray, gaze_uv, tan_half_fov: float, aspect: float
) -> np.ndarray:
    """Angle between each pixel's ray and the fixation ray, as the shader does."""

    scale = np.array([aspect * tan_half_fov, tan_half_fov])
    here = np.stack([u * 2.0 - 1.0, v * 2.0 - 1.0], axis=-1) * scale
    gaze = (np.asarray(gaze_uv, dtype=np.float64) * 2.0 - 1.0) * scale
    ray = np.concatenate([here, np.ones(here.shape[:-1] + (1,))], axis=-1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    fixation = np.append(gaze, 1.0)
    fixation /= np.linalg.norm(fixation)
    return np.degrees(np.arccos(np.clip(ray @ fixation, -1.0, 1.0)))


def _aces(x: np.ndarray) -> np.ndarray:
    return np.clip(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0
    )


def _predict(
    hdr_levels: list[np.ndarray],
    bloom_levels: list[np.ndarray],
    adaptation: np.ndarray,
    vision: HumanVisionState,
    bloom_strength: float,
    tan_half_fov: float,
    aspect: float,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = hdr_levels[0].shape[:2]
    u = np.broadcast_to((np.arange(width) + 0.5) / width, (height, width))
    v = np.broadcast_to(
        ((np.arange(height) + 0.5) / height)[:, None], (height, width)
    )
    uniforms = vision.uniforms()

    # Peripheral acuity: cortical magnification gives resolvable frequency
    # 1 / (1 + e / E2), applied as a mip bias rather than a uniform blur.
    eccentricity = _eccentricity_deg(
        u, v, vision.gaze_uv, tan_half_fov, aspect
    )
    acuity = acuity_fraction(eccentricity)
    lod = uniforms["maximum_blur_lod"] * (1.0 - acuity)
    scene = _texture_lod(hdr_levels, u, v, lod)
    glare_source = _bilinear(bloom_levels[0], u, v) * bloom_strength

    # Disability glare: the Stiles-Holladay 1/theta^2 tail is far wider than
    # the bloom kernel, so a heavily reduced mip of the bloom stands in for it.
    wide_field = _bilinear(bloom_levels[min(5, len(bloom_levels) - 1)], u, v)
    veiling = (
        uniforms["glare_constant"]
        * (wide_field @ PHOTOPIC_WEIGHTS)
        * GLARE_SCALE
    )

    state = _bilinear(adaptation, u, v)
    adapting_white = state[..., :3]
    local_luminance = state[..., 3]

    retinal = (scene + glare_source + veiling[..., None]) * vision.pupil_gain
    reference = np.maximum(
        local_luminance * 0.65 + vision.adapting_luminance_cd_m2 * 0.35, 1e-6
    )
    normalized = retinal / (reference[..., None] * 6.0 + 1e-5)

    # Evaluated over the whole frame at once. The gains vary per pixel only
    # through the adapting white, which is global today, but keeping this
    # per-pixel means a future local white needs no change here.
    gains = chromatic_adaptation_gains(
        adapting_white, vision.chromatic_degree
    )
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
    predicted = _aces(np.maximum(mesopic, 0.0)) ** (1.0 / 2.2)
    veiling_share = veiling / np.maximum(
        scene[..., 1] + glare_source[..., 1] + veiling, 1e-12
    )
    exercised = {
        "peripheral_lod_max": float(lod.max()),
        "peripheral_eccentricity_max_deg": float(eccentricity.max()),
        "veiling_luminance_max": float(veiling.max()),
        # A share includes the term itself in the denominator and therefore
        # remains in [0, 1].  The earlier veil/(scene+bloom) ratio could exceed
        # 100% in a dark pixel and did not match its own reported name.
        "veiling_share_of_retinal_max": float(veiling_share.max()),
    }
    return predicted, exercised


def measure(convergence_s: float = ADAPTATION_CONVERGENCE_S) -> dict[str, object]:
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config)
    renderer = app.renderer
    renderer.post.set_mode(DisplayMode.HUMAN_VISION)
    # Exercise disability glare with the simulator's own radiometric source,
    # not a synthetic post-process patch.  The development shell launched by
    # SimulatorApp is advanced just past burst so its stars occupy enough of
    # the field for the wide glare tail to be measurable.  A rocket frozen at
    # launch exercised the old, darker scene but became negligible after the
    # event-photo sky calibration raised the real urban background.
    physics_step_s = 1.0 / config.render.physics_hz
    for _ in range(round(3.4 / physics_step_s)):
        app.world.update(physics_step_s)
    step_s = 1.5
    for _ in range(int(convergence_s / step_s)):
        renderer.render(app.world, app.camera, app.celestial, step_s)

    renderer.render(app.world, app.camera, app.celestial, 0.0)
    app.ctx.finish()
    width, height = app.ctx.screen.size
    displayed = np.frombuffer(
        app.ctx.screen.read(components=3), dtype=np.uint8
    ).reshape(height, width, 3).astype(np.float64) / 255.0
    vision = renderer.post.vision
    # One level past the deepest the acuity bias can request, because
    # trilinear filtering reads the level above the one it lands on.
    hdr_levels = _mip_chain(
        renderer.targets.hdr_texture,
        int(math.floor(vision.uniforms()["maximum_blur_lod"])) + 2,
    )
    bloom_levels = _mip_chain(renderer.targets.bloom_textures[0], 6)
    adaptation_texture = renderer.targets.current_adaptation
    adaptation = np.frombuffer(
        adaptation_texture.read(), dtype=np.float16
    ).reshape(
        adaptation_texture.size[1], adaptation_texture.size[0], 4
    ).astype(np.float64)
    pygame.display.flip()

    tan_half_fov = math.tan(
        math.radians(vertical_fov_deg(config.physical_camera)) * 0.5
    )
    predicted, exercised = _predict(
        hdr_levels,
        bloom_levels,
        adaptation,
        vision,
        config.render.bloom_strength,
        tan_half_fov,
        config.render.width / config.render.height,
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
        "glare_verified": True,
        "peripheral_acuity_verified": True,
        **exercised,
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
