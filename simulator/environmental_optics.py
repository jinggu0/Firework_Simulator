from __future__ import annotations

import math

import numpy as np


def periodic_cloud_noise(
    width: int = 512, height: int = 256, seed: int = 20241005
) -> np.ndarray:
    """Deterministic, tileable multi-scale density field for GPU advection."""

    generator = np.random.default_rng(seed)
    white = generator.standard_normal((height, width))
    spectrum = np.fft.rfft2(white)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    radial = np.sqrt(fx * fx + fy * fy)
    field = np.zeros((height, width), dtype=np.float64)
    for cutoff, weight in ((0.012, 0.58), (0.032, 0.29), (0.075, 0.13)):
        low_pass = np.exp(-((radial / cutoff) ** 2))
        octave = np.fft.irfft2(spectrum * low_pass, s=(height, width))
        octave -= octave.min()
        octave /= max(float(np.ptp(octave)), 1e-9)
        field += octave * weight
    field -= field.min()
    field /= max(float(np.ptp(field)), 1e-9)
    return np.round(field * 255.0).astype(np.uint8)


def procedural_star_catalogue(
    width: int = 1024,
    height: int = 512,
    count: int = 3_500,
    seed: int = 20241005,
) -> np.ndarray:
    """Stable equirectangular flux field with subpixel reconstruction.

    .. warning::

        These positions and magnitudes are **invented**, not astrometry. This
        is confidence grade D and exists only so the sky is populated before
        ``python -m tools.import_star_catalogue`` has been run. Deterministic
        is not the same as real. :class:`simulator.starcatalogue.StarCatalogue`
        supersedes it whenever measured data is present.
    """

    generator = np.random.default_rng(seed)
    catalogue = np.zeros((height, width, 3), dtype=np.float32)
    longitude = generator.integers(0, width, count)
    sphere_y = generator.uniform(-1.0, 1.0, count)
    latitude = np.clip(
        ((np.arcsin(sphere_y) / math.pi + 0.5) * height).astype(np.int32),
        0,
        height - 1,
    )
    magnitude = 0.2 + 5.8 * generator.random(count) ** 0.34
    flux = 10.0 ** (-0.4 * magnitude)
    warm = generator.random(count)
    colors = np.column_stack(
        (
            0.72 + 0.28 * warm,
            0.82 - 0.04 * warm,
            1.0 - 0.46 * warm,
        )
    ).astype(np.float32)
    for dx, dy, weight in (
        (0, 0, 1.0),
        (-1, 0, 0.28),
        (1, 0, 0.28),
        (0, -1, 0.28),
        (0, 1, 0.28),
    ):
        x = (longitude + dx) % width
        y = np.clip(latitude + dy, 0, height - 1)
        np.add.at(catalogue, (y, x), colors * flux[:, None] * weight)
    return catalogue


def beer_lambert_transmittance(
    extinction_coefficient_per_m: float | np.ndarray,
    path_length_m: float,
) -> np.ndarray:
    """Spectral direct-beam transmission through a homogeneous medium."""

    coefficient = np.maximum(
        np.asarray(extinction_coefficient_per_m, dtype=np.float64), 0.0
    )
    return np.exp(-coefficient * max(path_length_m, 0.0))


def relative_air_mass(altitude_rad: float) -> float:
    """Kasten-Young optical air mass, finite close to the horizon."""

    altitude_deg = math.degrees(max(altitude_rad, math.radians(-4.9)))
    zenith_deg = 90.0 - altitude_deg
    cosine = max(math.cos(math.radians(zenith_deg)), 0.0)
    correction = 0.50572 * max(96.07995 - zenith_deg, 1e-3) ** -1.6364
    return 1.0 / max(cosine + correction, 1e-3)


LEGACY_ZENITH_RELATIVE_OPTICAL_DEPTH = 0.12
"""Retained default for the zenith-relative star extinction.

The physically derived depth now comes from
:mod:`simulator.atmosphere`; at standard pressure with the documented urban
turbidity it is about 0.36 at 550 nm. Passing that here would dim every star by
a further 30 percent, which would invalidate the separately calibrated radiance
scale in the sky shader. The calibration and the physics are therefore kept
separate until a measured sky luminance is available to recalibrate against.
"""


def star_contrast_visibility(
    star_radiance: float,
    sky_radiance: float,
    cloud_optical_depth: float,
    air_mass: float,
    contrast_threshold: float = 0.18,
    vertical_optical_depth: float = LEGACY_ZENITH_RELATIVE_OPTICAL_DEPTH,
) -> float:
    """Smooth background-limited stellar visibility after extinction.

    ``vertical_optical_depth`` is applied relative to the zenith, matching the
    sky shader. Supply :meth:`simulator.atmosphere.AtmosphericOptics.
    vertical_optical_depth` to make the wavelength and pressure dependence real.
    """

    atmospheric_transmission = math.exp(
        -max(vertical_optical_depth, 0.0) * max(air_mass - 1.0, 0.0)
    )
    cloud_transmission = math.exp(-max(cloud_optical_depth, 0.0))
    apparent = max(star_radiance, 0.0) * atmospheric_transmission
    apparent *= cloud_transmission
    contrast = apparent / max(sky_radiance, 1e-9)
    lower = contrast_threshold * 0.55
    upper = contrast_threshold * 1.45
    normalized = min(max((contrast - lower) / (upper - lower), 0.0), 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def grass_tip_displacement_m(
    wind_speed_mps: float, normalized_height: float, gust: float
) -> float:
    """Bounded cantilever-like blade displacement used by the GPU model."""

    height_weight = min(max(normalized_height, 0.0), 1.0) ** 2
    wind_response = 1.0 - math.exp(-max(wind_speed_mps, 0.0) / 3.2)
    gust_scale = min(max(0.72 + 0.28 * gust, 0.35), 1.15)
    return 0.22 * height_weight * wind_response * gust_scale
