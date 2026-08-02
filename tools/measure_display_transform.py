"""Measure the rendered display transform against its CPU reference (V-23).

``tonemap.frag`` is the only stage between linear scene radiance and what a
viewer sees, and it now carries five claims: lens distortion, natural
falloff, the sensor's photon-to-electron conversion, full-well clipping, and a
white balance applied after that clip. None of those is checkable by looking at
the image.

They are checkable by prediction. The linear HDR and bloom buffers are the
shader's only image inputs, so reading them back and running the same chain in
NumPy produces the frame the GPU should have produced. Sensor noise is
disabled for the measurement — it is a stochastic stage whose *distribution*
is the claim, not its per-pixel value, and leaving it on would only measure the
hash function.

The comparison is necessarily made on the 8-bit display-referred output,
because that is what this shader emits; the tolerance is set by that
quantisation. Every other colour metric in the harness reads the linear buffer
instead.

Prints a JSON payload; ``simulator.validation.display_transform`` runs this in
a subprocess so the rest of the report still works without a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.camera_optics import (
    LensDistortion,
    analog_gain,
    frame_half_extent,
    photon_to_electron_scale,
    white_balance_gains,
)
from simulator.config import SimulationConfig
from tools.measure_aerial_perspective import _bilinear, _read_rgb


def _aces(x: np.ndarray) -> np.ndarray:
    return np.clip(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0
    )


def _predict(
    hdr: np.ndarray,
    bloom: np.ndarray,
    camera_config,
    bloom_strength: float,
    overscan: float = 1.0,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Run the same chain tonemap.frag runs, on the same two buffers.

    The grid is the *output* one. With a lens calibration loaded the scene is
    rendered over a wider field and at more pixels, so the buffers are larger
    than the frame they are sampled into.
    """

    width, height = output_size or (hdr.shape[1], hdr.shape[0])
    half_extent = np.asarray(frame_half_extent(camera_config))
    # Pixel centres in OpenGL row order, matching the buffers as read.
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    grid_u, grid_v = np.meshgrid(u, v)
    distorted = np.stack(
        [(grid_u * 2.0 - 1.0), (grid_v * 2.0 - 1.0)], axis=-1
    ) * half_extent
    sensor_position = LensDistortion.from_config(camera_config).undistort(
        distorted
    )
    source = sensor_position / (half_extent * overscan) * 0.5 + 0.5
    source_u = np.clip(source[..., 0], 0.0, 1.0)
    source_v = np.clip(source[..., 1], 0.0, 1.0)

    scene = _bilinear(hdr, source_u, source_v)
    glare = _bilinear(bloom, source_u, source_v) * bloom_strength
    cos_theta = 1.0 / np.sqrt(1.0 + (sensor_position**2).sum(axis=-1))
    vignetting = cos_theta[..., None] ** 4

    electrons = (
        np.maximum(scene + glare, 0.0)
        * photon_to_electron_scale(camera_config).astype(np.float64)
        * vignetting
    )
    electrons = np.clip(electrons, 0.0, camera_config.full_well_electrons)
    signal = (
        electrons
        / camera_config.full_well_electrons
        * white_balance_gains(camera_config).astype(np.float64)
        * analog_gain(camera_config)
    )
    return _aces(signal) ** (1.0 / 2.2)


def measure(frames_warmup: int = 4) -> dict[str, object]:
    base = SimulationConfig()
    camera_config = replace(base.physical_camera, enable_sensor_noise=False)
    config = replace(
        base,
        render=replace(base.render, vsync=False, target_fps=0),
        physical_camera=camera_config,
    )
    app = SimulatorApp(config)
    renderer = app.renderer
    for _ in range(frames_warmup):
        renderer.render(app.world, app.camera, app.celestial, 1.0 / 60.0)

    renderer.render(app.world, app.camera, app.celestial, 0.0)
    app.ctx.finish()
    # Both reads are in OpenGL row order — row 0 is the bottom — so they line
    # up without flipping either. The screen is the only 8-bit surface here.
    width, height = app.ctx.screen.size
    displayed = np.frombuffer(
        app.ctx.screen.read(components=3), dtype=np.uint8
    ).reshape(height, width, 3).astype(np.float64) / 255.0
    hdr = _read_rgb(renderer.targets.hdr_texture)
    bloom = _read_rgb(renderer.targets.bloom_textures[0])
    pygame.display.flip()

    overscan = renderer.overscan
    predicted = _predict(
        hdr,
        bloom,
        camera_config,
        config.render.bloom_strength,
        overscan,
        (width, height),
    )
    error = np.abs(predicted - displayed)
    gains = white_balance_gains(camera_config)
    distortion = LensDistortion.from_config(camera_config)
    half_extent = frame_half_extent(camera_config)
    rendered_extent = (half_extent[0] * overscan, half_extent[1] * overscan)
    lit = displayed.max(axis=-1) > 4.0 / 255.0
    result = {
        "white_balance_temperature_k": camera_config.white_balance_temperature_k,
        "white_balance_gain": [float(value) for value in gains],
        "distortion_is_identity": distortion.is_identity,
        # Coverage is measured against the *rendered* field, which is what the
        # display transform actually samples. Without overscan this is the
        # sensor's field and a barrel lens falls short of 1.
        "distortion_frame_coverage": distortion.frame_coverage(
            half_extent, rendered_extent
        ),
        "coverage_without_overscan": distortion.frame_coverage(half_extent),
        "distortion_inverse_residual": distortion.inverse_residual(half_extent),
        "overscan": overscan,
        "required_overscan": distortion.required_overscan(half_extent),
        "render_size": [renderer.render_config.width, renderer.render_config.height],
        "output_size": [width, height],
        "absolute_error_max": float(error.max()),
        "absolute_error_mean": float(error.mean()),
        "absolute_error_p999": float(np.percentile(error, 99.9)),
        "display_quantum": 1.0 / 255.0,
        "lit_pixel_fraction": float(lit.mean()),
        "sensor_noise_enabled": camera_config.enable_sensor_noise,
    }
    app.audio_executor.shutdown(wait=True, cancel_futures=True)
    pygame.quit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the rendered display transform with its CPU reference."
    )
    parser.add_argument("--warmup-frames", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(measure(args.warmup_frames), indent=2))


if __name__ == "__main__":
    main()
