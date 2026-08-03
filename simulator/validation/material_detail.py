"""Metric-scale and temporal diagnostics for scanned ground materials."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..camera_optics import vertical_fov_deg
from ..config import PhysicalCameraConfig, RenderConfig
from ..material_textures import (
    MAP_SUFFIXES,
    MATERIAL_TEXTURE_DIRECTORY,
    SCANNED_MATERIALS,
)


DISTANCE_SAMPLES_M = (2.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def ground_pixel_footprints(
    distance_m: float,
    eye_height_m: float,
    camera: PhysicalCameraConfig,
    render: RenderConfig,
) -> dict[str, float]:
    """Approximate one ground pixel's along/cross-view dimensions.

    The along-view derivative is exact for a level pinhole camera looking at a
    horizontal plane: ``d = h*cot(theta)``. The cross-view footprint uses the
    angular width of one pixel at the same distance. These numbers are a
    sampling diagnostic, not a reconstruction of a particular triangle.
    """

    if distance_m <= 0.0 or eye_height_m <= 0.0:
        raise ValueError("distance and eye height must be positive")
    vertical_fov_rad = math.radians(vertical_fov_deg(camera))
    aspect = render.width / render.height
    horizontal_fov_rad = 2.0 * math.atan(
        math.tan(vertical_fov_rad * 0.5) * aspect
    )
    vertical_angle_per_pixel = vertical_fov_rad / render.height
    horizontal_angle_per_pixel = horizontal_fov_rad / render.width
    along_m = (
        (distance_m * distance_m + eye_height_m * eye_height_m)
        / eye_height_m
        * vertical_angle_per_pixel
    )
    across_m = 2.0 * distance_m * math.tan(
        horizontal_angle_per_pixel * 0.5
    )
    return {
        "distance_m": distance_m,
        "along_view_m_per_px": along_m,
        "cross_view_m_per_px": across_m,
        "footprint_anisotropy": along_m / max(across_m, 1e-12),
    }


def projection_sampling_report(
    eye_height_m: float = 1.68,
    camera: PhysicalCameraConfig | None = None,
    render: RenderConfig | None = None,
) -> dict[str, Any]:
    camera = camera or PhysicalCameraConfig()
    render = render or RenderConfig()
    return {
        "eye_height_m": eye_height_m,
        "resolution_px": [render.width, render.height],
        "focal_length_mm": camera.focal_length_mm,
        "sensor_size_mm": [camera.sensor_width_mm, camera.sensor_height_mm],
        "vertical_fov_deg": vertical_fov_deg(camera),
        "samples": [
            ground_pixel_footprints(distance, eye_height_m, camera, render)
            for distance in DISTANCE_SAMPLES_M
        ],
    }


def _edge_discontinuity(image: np.ndarray) -> dict[str, float]:
    image = image.astype(np.float64) / 255.0
    internal_x = float(np.abs(np.diff(image, axis=1)).mean())
    internal_y = float(np.abs(np.diff(image, axis=0)).mean())
    seam_x = float(np.abs(image[:, 0] - image[:, -1]).mean())
    seam_y = float(np.abs(image[0] - image[-1]).mean())
    return {
        "x_edge_mean_absolute": seam_x,
        "y_edge_mean_absolute": seam_y,
        "x_edge_to_internal_ratio": seam_x / max(internal_x, 1e-12),
        "y_edge_to_internal_ratio": seam_y / max(internal_y, 1e-12),
    }


def scanned_material_report(
    directory: Path = MATERIAL_TEXTURE_DIRECTORY,
    projection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    projection = projection or projection_sampling_report()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for asset_id, runtime_width_m in SCANNED_MATERIALS:
        source = manifest["assets"][asset_id]
        width_m = float(source["tile_width_m"])
        if not math.isclose(width_m, runtime_width_m, abs_tol=1e-9):
            raise ValueError(
                f"{asset_id}: runtime width {runtime_width_m} does not match "
                f"manifest width {width_m}"
            )
        paths = {
            suffix: directory / f"{asset_id}_{suffix}_1k.jpg"
            for suffix in MAP_SUFFIXES
        }
        with Image.open(paths["diff"]) as source_image:
            diffuse = np.asarray(source_image.convert("RGB")).copy()
        height_px, width_px = diffuse.shape[:2]
        if width_px != height_px:
            raise ValueError(f"{asset_id}: scanned tile must be square")
        sampling = []
        texel_pitch_m = width_m / width_px
        for sample in projection["samples"]:
            major_texels_per_px = (
                sample["along_view_m_per_px"] / texel_pitch_m
            )
            sampling.append(
                {
                    "distance_m": sample["distance_m"],
                    "major_axis_texels_per_px": major_texels_per_px,
                    "implicit_major_axis_lod": max(
                        math.log2(max(major_texels_per_px, 1e-12)), 0.0
                    ),
                    "footprint_anisotropy": sample["footprint_anisotropy"],
                }
            )
        records.append(
            {
                "asset_id": asset_id,
                "tile_width_m": width_m,
                "resolution_px": [width_px, height_px],
                "texel_pitch_mm": texel_pitch_m * 1000.0,
                "tile_repetitions_across_2_to_80_m": 78.0 / width_m,
                "diffuse_edge_discontinuity": _edge_discontinuity(diffuse),
                "sampling": sampling,
                "files": {
                    suffix: {
                        "file_name": paths[suffix].name,
                        "sha256": _digest(paths[suffix]),
                    }
                    for suffix in MAP_SUFFIXES
                },
            }
        )
    return records


def temporal_delta_metrics(
    frames: np.ndarray,
    roi_fraction: tuple[float, float, float, float] = (0.1, 0.45, 0.9, 0.95),
) -> dict[str, Any]:
    """Measure motion-frame deltas in a fixed ground-heavy image region."""

    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] < 2 or values.shape[-1] < 3:
        raise ValueError("frames must have shape (n>=2, height, width, channels>=3)")
    x0, y0, x1, y1 = roi_fraction
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("ROI fractions must form a non-empty unit rectangle")
    height, width = values.shape[1:3]
    left, right = round(width * x0), round(width * x1)
    top, bottom = round(height * y0), round(height * y1)
    rgb = np.maximum(values[:, top:bottom, left:right, :3], 0.0)
    luma = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    )
    delta = np.diff(luma, axis=0)
    absolute = np.abs(delta)
    # Five-tap spatial low pass. The residual is a repeatable high-frequency
    # temporal proxy; it is not called a pass/fail shimmer score until a
    # registered real sequence supplies a threshold.
    low_pass = (
        delta
        + np.roll(delta, 1, axis=1)
        + np.roll(delta, -1, axis=1)
        + np.roll(delta, 1, axis=2)
        + np.roll(delta, -1, axis=2)
    ) / 5.0
    high_frequency = delta - low_pass
    signal_p95 = float(np.percentile(luma, 95.0))
    threshold = max(signal_p95 * 0.02, 1e-12)
    return {
        "frame_count": int(values.shape[0]),
        "roi_fraction": list(roi_fraction),
        "roi_size_px": [right - left, bottom - top],
        "signal_luminance_p95": signal_p95,
        "mean_absolute_frame_delta": float(absolute.mean()),
        "p95_absolute_frame_delta": float(np.percentile(absolute, 95.0)),
        "normalized_mean_delta_to_signal_p95": float(
            absolute.mean() / max(signal_p95, 1e-12)
        ),
        "two_percent_signal_flip_fraction": float(
            np.count_nonzero(absolute > threshold) / absolute.size
        ),
        "high_frequency_delta_fraction": float(
            np.abs(high_frequency).sum() / max(absolute.sum(), 1e-12)
        ),
    }
