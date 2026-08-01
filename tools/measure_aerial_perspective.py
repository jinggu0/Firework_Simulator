"""Measure the rendered haze against its CPU reference (V-22).

The deferred composite claims to evaluate

    L = L_object * T + L_air * (1 - T)

with ``T`` the per-channel transmittance over the view path through an
exponential atmosphere and ``L_air`` the horizon sky radiance. That is
checkable rather than merely plausible: render the same frame twice, once with
the modelled extinction and once with the atmosphere removed, read the depth
buffer for the path each pixel represents, and predict the first frame from the
second using :meth:`simulator.atmosphere.SurfaceExtinction.transmittance`.

The comparison reads the linear RGBA16F buffer. A tone-mapped screenshot would
have lost the quantity being checked.

Prints a JSON payload; ``simulator.validation.aerial_perspective`` runs this in
a subprocess so the rest of the report still works without a GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.atmosphere import SurfaceExtinction
from simulator.config import SimulationConfig
from simulator.renderer import _look_at

VACUUM = SurfaceExtinction(
    aerosol_per_m=(0.0, 0.0, 0.0), molecular_per_m=(0.0, 0.0, 0.0)
)
"""No atmosphere at all, so the second render isolates the object radiance."""


def _read_rgb(texture) -> np.ndarray:
    """Linear RGB in OpenGL row order — row 0 is the bottom of the frame."""

    width, height = texture.size
    raw = np.frombuffer(texture.read(), dtype=np.float16)
    return raw.reshape(height, width, texture.components).astype(np.float64)[
        :, :, :3
    ]


def _read_depth(texture) -> np.ndarray:
    width, height = texture.size
    return np.frombuffer(texture.read(), dtype=np.float32).reshape(
        height, width
    ).astype(np.float64)


def _bilinear(field: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reproduce a GL_LINEAR, GL_CLAMP_TO_EDGE sample of ``field``."""

    height, width = field.shape[:2]
    x = np.clip(u * width - 0.5, 0.0, width - 1.0)
    y = np.clip(v * height - 0.5, 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = field[y0, x0] * (1.0 - fx) + field[y0, x1] * fx
    bottom = field[y1, x0] * (1.0 - fx) + field[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _world_positions(
    depth: np.ndarray, inverse_view_projection: np.ndarray
) -> np.ndarray:
    """Invert the projection the haze shader inverts, on the same grid."""

    height, width = depth.shape
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    ndc_x, ndc_y = np.meshgrid(u * 2.0 - 1.0, v * 2.0 - 1.0)
    homogeneous = np.stack(
        [ndc_x, ndc_y, depth * 2.0 - 1.0, np.ones_like(depth)], axis=-1
    )
    transformed = homogeneous @ inverse_view_projection.T
    return transformed[..., :3] / transformed[..., 3:4]


def measure(frames_warmup: int = 4) -> dict[str, object]:
    base = SimulationConfig()
    config = replace(base, render=replace(base.render, vsync=False, target_fps=0))
    app = SimulatorApp(config)
    # Stars are additive on top of the composite, so a burst in flight would
    # break the two-render solve. The opaque scene is what this metric checks.
    app.world.stars.count = 0
    renderer = app.renderer
    for _ in range(frames_warmup):
        renderer.render(app.world, app.camera, app.celestial, 1.0 / 60.0)
    # Zero elapsed time from here: the water spectrum and the facade animation
    # are functions of renderer.time_s, and the two renders must differ only in
    # the atmosphere.
    renderer.render(app.world, app.camera, app.celestial, 0.0)
    hazed = _read_rgb(renderer.targets.hdr_texture)
    depth = _read_depth(renderer.targets.scene_depth_texture)
    airlight = _read_rgb(renderer.targets.airlight_texture)
    extinction = renderer.surface_extinction

    renderer.set_air_extinction_override(VACUUM)
    renderer.render(app.world, app.camera, app.celestial, 0.0)
    clear = _read_rgb(renderer.targets.hdr_texture)
    renderer.set_air_extinction_override(None)

    view_projection = renderer.projection @ _look_at(
        app.camera.position_m, app.camera.position_m + app.camera.forward
    )
    world = _world_positions(depth, np.linalg.inv(view_projection))
    camera_position = np.asarray(app.camera.position_m, dtype=np.float64)
    offset = world - camera_position
    path_m = np.linalg.norm(offset, axis=-1)

    height, width = depth.shape
    u = np.broadcast_to((np.arange(width) + 0.5) / width, (height, width))
    v = np.broadcast_to(
        ((np.arange(height) + 0.5) / height)[:, None], (height, width)
    )
    sampled_airlight = _bilinear(airlight, u, v)

    # The same closed form the shader evaluates, computed here from the CPU
    # reference so a drifted constant on either side shows up as a residual.
    aerosol = np.asarray(extinction.aerosol_per_m)
    molecular = np.asarray(extinction.molecular_per_m)

    def profile_mean(scale_height_m: float) -> np.ndarray:
        low = abs(camera_position[1])
        high = np.abs(world[..., 1])
        rise = high - low
        at_start = np.exp(-low / scale_height_m)
        at_end = np.exp(-high / scale_height_m)
        level = np.abs(rise) < 1e-3
        safe = np.where(level, 1.0, rise)
        return np.where(
            level,
            at_start,
            scale_height_m * (at_start - at_end) / safe,
        )

    optical_depth = path_m[..., None] * (
        aerosol * profile_mean(extinction.aerosol_scale_height_m)[..., None]
        + molecular
        * profile_mean(extinction.molecular_scale_height_m)[..., None]
    )
    transmittance = np.exp(-optical_depth)
    predicted = clear * transmittance + sampled_airlight * (1.0 - transmittance)

    geometry = depth < 1.0
    sky = ~geometry
    scale = max(float(np.abs(hazed[geometry]).max()), 1e-12)
    error = np.abs(predicted - hazed)[geometry] / scale
    sky_difference = float(np.abs(hazed[sky] - clear[sky]).max()) if sky.any() else 0.0

    lit = geometry & (clear.max(axis=-1) > 1e-6)
    green_transmittance = transmittance[..., 1]
    result = {
        "visibility_km": extinction.visibility_m / 1_000.0,
        "surface_extinction_per_m": list(extinction.total_per_m),
        "geometry_fraction": float(geometry.mean()),
        "path_p50_m": float(np.percentile(path_m[geometry], 50)),
        "path_p95_m": float(np.percentile(path_m[geometry], 95)),
        "transmittance_p50": float(np.percentile(green_transmittance[geometry], 50)),
        "transmittance_min": float(green_transmittance[geometry].min()),
        "relative_error_max": float(error.max()),
        "relative_error_mean": float(error.mean()),
        "relative_error_p99": float(np.percentile(error, 99)),
        "sky_absolute_difference_max": sky_difference,
        "mean_radiance_change": float(
            (hazed[lit].mean() - clear[lit].mean()) / max(clear[lit].mean(), 1e-12)
        )
        if lit.any()
        else 0.0,
        "airlight_rgb": [float(value) for value in airlight.reshape(-1, 3).mean(axis=0)],
        "camera_position_eus_m": [float(value) for value in camera_position],
        "half_float_quantum": 2.0 ** -11,
    }
    app.audio_executor.shutdown(wait=True, cancel_futures=True)
    pygame.quit()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the rendered aerial perspective with its CPU reference."
    )
    parser.add_argument("--warmup-frames", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(measure(args.warmup_frames), indent=2))


if __name__ == "__main__":
    main()
