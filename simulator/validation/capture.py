"""Linear HDR frame capture, for colour and brightness validation.

Colour and brightness comparisons must read the linear RGBA16F buffer before
exposure and tone mapping. A gamma-encoded screenshot has already lost the
quantity being validated, and an SDR display cannot represent firework
luminance in the first place, so a metric computed on the displayed image would
measure the tone mapper.

This module is the only part of the validation package that needs OpenGL. It is
imported lazily by the runner so a headless agent can still produce the rest of
the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_HDR_REGRESSION_LIMITS = {
    "changed_pixel_fraction": 1.0e-5,
    "maximum_absolute_error": 1.0e-3,
    "mean_absolute_error": 1.0e-7,
}
DEFAULT_SDR_REGRESSION_LIMITS = {
    "changed_pixel_fraction": 1.0e-5,
    "mean_absolute_error_rgb8": 5.0e-4,
}


def read_linear_hdr(renderer: Any) -> np.ndarray:
    """Read a renderer's HDR colour target as linear float32 RGBA.

    Returns an ``(height, width, 4)`` array in image order — row 0 is the top of
    the frame — with the OpenGL bottom-left origin already corrected. Values are
    scene-referred linear radiance, not display-referred.
    """

    texture = getattr(renderer, "hdr_texture", None)
    if texture is None:
        raise AttributeError(
            "renderer has no hdr_texture; linear capture requires the HDR "
            "colour target created by Renderer.__init__"
        )
    width, height = texture.size
    components = texture.components
    if texture.dtype != "f2":
        raise ValueError(
            f"expected a half-float HDR target, found dtype {texture.dtype!r}"
        )
    raw = np.frombuffer(texture.read(), dtype=np.float16)
    frame = raw.reshape(height, width, components).astype(np.float32)
    # OpenGL texture row 0 is the bottom of the image.
    return np.flipud(frame).copy()


def linear_hdr_statistics(frame: np.ndarray) -> dict[str, float]:
    """Summarise a linear HDR frame without applying any display transform."""

    colour = frame[:, :, :3]
    finite = np.isfinite(colour)
    return {
        "width": float(frame.shape[1]),
        "height": float(frame.shape[0]),
        "minimum": float(colour[finite].min()) if finite.any() else float("nan"),
        "maximum": float(colour[finite].max()) if finite.any() else float("nan"),
        "mean": float(colour[finite].mean()) if finite.any() else float("nan"),
        "percentile_99": (
            float(np.percentile(colour[finite], 99)) if finite.any() else float("nan")
        ),
        "non_finite_fraction": float(1.0 - finite.mean()),
        "negative_fraction": float((colour < 0.0).mean()),
    }


def save_linear_hdr(frame: np.ndarray, path: Path) -> Path:
    """Persist a linear HDR frame as ``.npy`` for later comparison.

    Deliberately not an image format: PNG or JPEG would quantise and gamma-encode
    the very values the comparison depends on.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    np.save(path, frame.astype(np.float32))
    return path


def read_display_sdr(context: Any) -> np.ndarray:
    """Read the final display-referred RGB8 framebuffer in image order."""

    screen = getattr(context, "screen", None)
    if screen is None:
        raise AttributeError("OpenGL context has no screen framebuffer")
    width, height = screen.size
    raw = np.frombuffer(
        screen.read(components=3, alignment=1), dtype=np.uint8
    )
    expected = width * height * 3
    if raw.size != expected:
        raise ValueError(
            f"expected {expected} display bytes, received {raw.size}"
        )
    return np.flipud(raw.reshape(height, width, 3)).copy()


def display_sdr_statistics(frame: np.ndarray) -> dict[str, object]:
    """Summarise display RGB without presenting code values as radiance."""

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("display frame must have shape (height, width, 3)")
    luminance = rgb.astype(np.float64) @ np.array(
        [0.2126, 0.7152, 0.0722], dtype=np.float64
    )
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "median_rgb8": np.median(rgb.reshape(-1, 3), axis=0).tolist(),
        "luminance_p05_p50_p95": np.percentile(
            luminance, [5, 50, 95]
        ).tolist(),
        "clipped_black_fraction": float(np.all(rgb == 0, axis=2).mean()),
        "clipped_white_fraction": float(np.all(rgb == 255, axis=2).mean()),
    }


def save_display_sdr(frame: np.ndarray, path: Path) -> Path:
    """Save a display-referred RGB8 frame losslessly as PNG."""

    rgb = np.asarray(frame, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("display frame must have shape (height, width, 3)")
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


def compare_linear_hdr(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limits: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare scene-linear frames with a bounded raster-edge tolerance."""

    first = np.asarray(reference, dtype=np.float32)
    second = np.asarray(candidate, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] < 3:
        raise ValueError(
            "HDR frames must have the same (height, width, channels) shape"
        )
    thresholds = dict(DEFAULT_HDR_REGRESSION_LIMITS)
    if limits:
        thresholds.update(limits)
    difference = np.abs(first[:, :, :3] - second[:, :, :3])
    finite = np.isfinite(difference)
    changed_pixels = np.any(difference > 0.0, axis=2)
    metrics = {
        "changed_pixel_count": int(changed_pixels.sum()),
        "changed_pixel_fraction": float(changed_pixels.mean()),
        "maximum_absolute_error": (
            float(difference[finite].max()) if finite.any() else float("inf")
        ),
        "mean_absolute_error": (
            float(difference[finite].mean()) if finite.any() else float("inf")
        ),
        "non_finite_error_count": int((~finite).sum()),
    }
    passed = metrics["non_finite_error_count"] == 0 and all(
        metrics[key] <= value for key, value in thresholds.items()
    )
    return {"passed": bool(passed), "limits": thresholds, "metrics": metrics}


def compare_display_sdr(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    limits: dict[str, float] | None = None,
) -> dict[str, object]:
    """Compare final RGB8 frames without interpreting code values as light."""

    first = np.asarray(reference)
    second = np.asarray(candidate)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("SDR frames must have the same (height, width, 3) shape")
    thresholds = dict(DEFAULT_SDR_REGRESSION_LIMITS)
    if limits:
        thresholds.update(limits)
    difference = np.abs(first.astype(np.int16) - second.astype(np.int16))
    changed_pixels = np.any(difference > 0, axis=2)
    metrics = {
        "changed_pixel_count": int(changed_pixels.sum()),
        "changed_pixel_fraction": float(changed_pixels.mean()),
        "maximum_absolute_error_rgb8": int(difference.max()),
        "mean_absolute_error_rgb8": float(difference.mean()),
    }
    passed = all(metrics[key] <= value for key, value in thresholds.items())
    return {"passed": bool(passed), "limits": thresholds, "metrics": metrics}
