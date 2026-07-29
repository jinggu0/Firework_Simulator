from __future__ import annotations

import numpy as np


def blackbody_rgb(temperature_k: float) -> np.ndarray:
    """Approximate a black-body chromaticity as a linear RGB triplet.

    This is used as a physically meaningful baseline. Spectral emission curves
    for individual pyrotechnic compositions can replace it without changing
    the renderer interface.
    """

    t = float(np.clip(temperature_k, 1_000.0, 40_000.0)) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19.0 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0

    srgb = np.clip(np.array([r, g, b], dtype=np.float32) / 255.0, 0.0, 1.0)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)

