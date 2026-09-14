"""Measure temporal shimmer on the canonical views under a slow camera pan.

Each view is rendered along a short turn about the camera's own centre with
simulation time frozen, so between frames the only change is where the camera
points. The image motion that turn causes is known exactly from the camera
matrices, so every frame can be predicted from the one before it; what the
prediction misses is shimmer (`simulator.validation.temporal_shimmer`).

The planar reflection is re-rendered for every frame. Its normal schedule only
refreshes after simulated time passes, and with time frozen it would keep the
first pose's reflection for the whole path — a lag, not shimmer.

Two frames are scored for each pose. The linear HDR target before the display
transform attributes instability to the render passes. The display frame is
what a viewer sees, but the human-vision transform blurs the periphery through
mip levels tied to the screen, so part of its residual belongs to the observer
model rather than the scene.

Example::

    python -m tools.measure_temporal_shimmer --ambient-occlusion-off --static-control
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from simulator.app import SimulatorApp
from simulator.camera import FreeCamera
from simulator.config import SimulationConfig
from simulator.passes.post import DisplayMode
from simulator.renderer import _look_at
from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import (
    load_known_residuals,
    read_display_sdr,
    read_linear_hdr,
    select_residuals,
)
from simulator.validation.frame_comparison import LUMINANCE_WEIGHTS
from simulator.validation.temporal_shimmer import (
    BORDER_MARGIN_PX,
    MINIMUM_FRAME_PAIRS,
    pan_source_coordinates,
    shimmer_statistics,
)
from simulator.validation.views import (
    DEFAULT_VISUAL_VIEWS_PATH,
    load_visual_regression_suite,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(
    "docs/validation/temporal_shimmer_v2/temporal_shimmer_report.json"
)
#: Twenty-four frames give twenty-three pairs, clear of the statistics' minimum.
FRAMES = MINIMUM_FRAME_PAIRS + 4
#: The golden-ratio conjugate. Its multiples are as far from whole numbers as
#: any step's can be, so no two frames on the path sit a whole number of pixels
#: apart at the image centre, where aliased detail could line up and hide.
PIXELS_PER_FRAME = (math.sqrt(5.0) - 1.0) / 2.0
#: Matches the static captures, which settle the same state before reading.
WARMUP_FRAMES = 8
#: Same relative threshold as `tools.measure_aliasing`.
RELATIVE_THRESHOLD = 0.05
#: Same threshold as `frame_comparison.temporal_stability`.
DISPLAY_THRESHOLD_CODE_VALUES = 2.0


def camera_view_projection(
    projection: np.ndarray,
    position_m: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """The matrix `Renderer._update_camera` builds for this camera state."""

    camera = FreeCamera()
    camera.position_m[:] = np.asarray(position_m, dtype=np.float32)
    camera.yaw_deg, camera.pitch_deg = yaw_deg, pitch_deg
    view = _look_at(camera.position_m, camera.position_m + camera.forward)
    return np.asarray(projection, dtype=np.float64) @ view.astype(np.float64)


def _centre_motion_px(
    projection, position_m, yaw_deg, pitch_deg, step_deg, width, height
) -> float:
    before = camera_view_projection(projection, position_m, yaw_deg, pitch_deg)
    after = camera_view_projection(
        projection, position_m, yaw_deg + step_deg, pitch_deg
    )
    coordinates = pan_source_coordinates(before, after, width, height)
    row, col = height // 2, width // 2
    return float(
        np.hypot(coordinates[0, row, col] - row, coordinates[1, row, col] - col)
    )


def yaw_step_for_centre_motion(
    projection: np.ndarray,
    position_m: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    pixels_per_frame: float,
) -> float:
    """Yaw per frame, in degrees, that moves the image centre this many pixels.

    Yaw turns about the world vertical, so a tilted camera sweeps less of its
    image per degree. The level-camera estimate is corrected by measuring the
    motion it actually produces through the renderer's own matrices.
    """

    focal_px = float(projection[1][1]) * height / 2.0
    estimate = math.degrees(math.atan(pixels_per_frame / focal_px))
    moved = _centre_motion_px(
        projection, position_m, yaw_deg, pitch_deg, estimate, width, height
    )
    if moved <= 0.0:
        raise ValueError("a yaw step does not move the image centre")
    return estimate * pixels_per_frame / moved


def residual_exclusion_mask(
    shape: tuple[int, int], residuals: list[dict], dilation_px: int
) -> np.ndarray:
    """Known driver residual regions, grown by how far the path moves them."""

    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for residual in residuals:
        row0, row1 = residual["region_rows"]
        col0, col1 = residual["region_cols"]
        mask[
            max(row0 - dilation_px, 0) : min(row1 + dilation_px, height - 1) + 1,
            max(col0 - dilation_px, 0) : min(col1 + dilation_px, width - 1) + 1,
        ] = True
    return mask


def render_pan(
    view,
    display_mode: str,
    frames: int,
    pixels_per_frame: float,
    ambient_occlusion_strength: float | None = None,
) -> dict[str, Any]:
    """Render one view along the pan and return both luminance sequences."""

    base = SimulationConfig()
    render = replace(base.render, vsync=False, target_fps=0)
    if ambient_occlusion_strength is not None:
        render = replace(
            render, ambient_occlusion_strength=ambient_occlusion_strength
        )
    app = SimulatorApp(
        replace(base, render=render), scenario_path=DEFAULT_SCENARIO_PATH
    )
    try:
        app.world.shells.clear()
        app.world.stars.count = 0
        app.renderer.post.set_mode(DisplayMode(display_mode))
        view.apply(app.camera)
        for _ in range(WARMUP_FRAMES):
            app.renderer.render(app.world, app.camera, app.celestial, 1.0 / 60.0, None)
        width, height = app.renderer.hdr_texture.size
        projection = app.renderer.projection
        yaw0 = app.camera.yaw_deg
        step_deg = (
            yaw_step_for_centre_motion(
                projection, app.camera.position_m, yaw0,
                app.camera.pitch_deg, width, height, pixels_per_frame,
            )
            if pixels_per_frame > 0.0
            else 0.0
        )
        hdr, display, matrices = [], [], []
        for frame in range(frames):
            app.camera.yaw_deg = yaw0 + step_deg * frame
            app.renderer.reflection_ready = False
            app.renderer.render(app.world, app.camera, app.celestial, 0.0, None)
            app.ctx.finish()
            hdr.append(
                read_linear_hdr(app.renderer)[:, :, :3].astype(np.float64)
                @ LUMINANCE_WEIGHTS
            )
            display.append(
                read_display_sdr(app.ctx).astype(np.float64) @ LUMINANCE_WEIGHTS
            )
            matrices.append(
                camera_view_projection(
                    projection, app.camera.position_m,
                    app.camera.yaw_deg, app.camera.pitch_deg,
                )
            )
        info = dict(app.ctx.info)
        return {
            "linear_hdr": np.stack(hdr),
            "display_luma": np.stack(display),
            "matrices": matrices,
            "yaw_step_deg": step_deg,
            "gl": {
                key: info.get(key)
                for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")
            },
        }
    finally:
        app.audio_executor.shutdown(wait=True, cancel_futures=True)
        pygame.quit()


def score_pan(pan: dict[str, Any], residuals: list[dict]) -> dict[str, Any]:
    """Score one rendered pan in both units, beside the uncompensated score."""

    frames, height, width = pan["linear_hdr"].shape
    matrices = pan["matrices"]
    coordinates = np.stack(
        [
            pan_source_coordinates(matrices[k], matrices[k + 1], width, height)
            for k in range(frames - 1)
        ]
    )
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    identity = np.stack([rows, cols])
    displacement = np.hypot(coordinates[:, 0] - rows, coordinates[:, 1] - cols)
    path_motion_px = float(displacement.max(axis=(1, 2)).sum())
    dilation = math.ceil(path_motion_px) + BORDER_MARGIN_PX
    exclude = residual_exclusion_mask((height, width), residuals, dilation)
    stationary = np.broadcast_to(identity, coordinates.shape)
    centre = displacement[:, height // 2, width // 2]
    return {
        "yaw_step_deg": pan["yaw_step_deg"],
        "centre_motion_px_per_pair": [float(centre.min()), float(centre.max())],
        "largest_motion_px_per_pair": float(displacement.max()),
        "excluded_residuals": [residual["id"] for residual in residuals],
        "exclusion_dilation_px": dilation if residuals else 0,
        "linear_hdr": shimmer_statistics(
            pan["linear_hdr"], coordinates,
            threshold=RELATIVE_THRESHOLD, relative=True, exclude=exclude,
        ),
        "display_luma": shimmer_statistics(
            pan["display_luma"], coordinates,
            threshold=DISPLAY_THRESHOLD_CODE_VALUES, relative=False, exclude=exclude,
        ),
        # The same frames scored as if nothing had moved. Where scene content
        # moves this is far worse than the compensated score. Where a pattern
        # stays on its pixels while the scene turns — ambient occlusion noise
        # on a surface a metre away — it can be better, because that pattern
        # only shows up once motion is accounted for. It is context, not a
        # check on the motion; motion_prediction_check.json is that check.
        "linear_hdr_without_motion_compensation": shimmer_statistics(
            pan["linear_hdr"], stationary,
            threshold=RELATIVE_THRESHOLD, relative=True, exclude=exclude,
        ),
    }


def measure_view(view, display_mode, frames, pixels_per_frame, record, *,
                 ambient_occlusion_off: bool, static_control: bool) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    shipped = render_pan(view, display_mode, frames, pixels_per_frame)
    gl = shipped["gl"]
    residuals = select_residuals(record, view.view_id, gl)
    runs["shipped"] = score_pan(shipped, residuals)
    del shipped
    if ambient_occlusion_off:
        ablated = render_pan(
            view, display_mode, frames, pixels_per_frame, ambient_occlusion_strength=0.0
        )
        runs["ambient_occlusion_off"] = score_pan(ablated, residuals)
        del ablated
    if static_control:
        still = render_pan(view, display_mode, frames, 0.0)
        runs["static_control"] = score_pan(still, residuals)
        del still
    return {"view_id": view.view_id, "gl": gl, **runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="append", default=[])
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--pixels-per-frame", type=float, default=PIXELS_PER_FRAME)
    parser.add_argument("--ambient-occlusion-off", action="store_true")
    parser.add_argument("--static-control", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.frames < MINIMUM_FRAME_PAIRS + 1:
        parser.error(f"--frames must give at least {MINIMUM_FRAME_PAIRS} pairs")
    if arguments.pixels_per_frame <= 0.0:
        parser.error("--pixels-per-frame must be positive; use --static-control")

    suite = load_visual_regression_suite(DEFAULT_VISUAL_VIEWS_PATH)
    views = [
        view for view in suite.views
        if not arguments.view or view.view_id in arguments.view
    ]
    record = load_known_residuals()
    results = []
    for view in views:
        print(f"measuring {view.view_id}", flush=True)
        results.append(
            measure_view(
                view, suite.display_mode, arguments.frames,
                arguments.pixels_per_frame, record,
                ambient_occlusion_off=arguments.ambient_occlusion_off,
                static_control=arguments.static_control,
            )
        )

    report = {
        "schema_version": 1,
        "probe": "motion_compensated_residual_under_camera_pan",
        "frames": arguments.frames,
        "pixels_per_frame_at_centre": arguments.pixels_per_frame,
        "relative_threshold_linear_hdr": RELATIVE_THRESHOLD,
        "threshold_display_code_values": DISPLAY_THRESHOLD_CODE_VALUES,
        "views": results,
        "method": [
            "Yaw about the camera centre with simulation time frozen; the "
            "reflection pre-pass is re-rendered every frame.",
            "Each frame is predicted from the previous one by cubic-spline "
            "interpolation along the motion the camera matrices imply; the "
            "residual is what did not move with the scene.",
            "A pixel's score is the RMS residual over all frame pairs, relative "
            "to its mean for linear HDR and in code values for the display frame.",
        ],
        "limitations": [
            "A turn about the camera centre has no parallax, so instability that "
            "only appears when the camera translates (reflections, disocclusion) "
            "is not measured.",
            "A change that blurs the frame also scores as more stable; a lower "
            "score must be read beside what the change removed.",
            "The display score includes the human-vision transform, whose "
            "peripheral blur samples mip levels tied to the screen.",
            "One GPU. Renders at a fixed pose repeat to within the thresholds "
            "but not always bit for bit: terrain_shoreline frames differ slightly, "
            "and runs with ambient occlusion differ by a few pixels between "
            "processes. Recorded driver residual regions are excluded with the "
            "path motion added.",
        ],
    }
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")
    for result in results:
        for run in ("shipped", "ambient_occlusion_off", "static_control"):
            if run not in result:
                continue
            scored = result[run]
            hdr = scored["linear_hdr"]["unstable_pixel_fraction"]
            display = scored["display_luma"]["unstable_pixel_fraction"]
            still = scored["linear_hdr_without_motion_compensation"][
                "unstable_pixel_fraction"
            ]
            print(
                f"  {result['view_id']:18} {run:22} "
                f"hdr unstable={hdr * 100:6.2f}% "
                f"display unstable={display * 100:6.2f}% "
                f"uncompensated={still * 100:6.2f}%"
            )


if __name__ == "__main__":
    main()
