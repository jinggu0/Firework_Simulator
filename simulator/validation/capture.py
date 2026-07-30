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
