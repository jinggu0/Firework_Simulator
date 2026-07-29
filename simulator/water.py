from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

GRAVITY_MPS2 = 9.80665


@dataclass(frozen=True, slots=True)
class WaterConfig:
    wind_speed_mps: float = 2.5
    # Meteorological bearing: direction the wind comes from.
    wind_direction_deg: float = 255.0
    fetch_length_m: float = 1_200.0
    wave_count: int = 32
    minimum_wavelength_m: float = 0.35
    maximum_wavelength_m: float = 24.0
    directional_spread_power: float = 8.0
    choppiness: float = 0.72
    grid_size: tuple[int, int] = (161, 97)
    extent_m: tuple[float, float] = (700.0, 420.0)


@dataclass(frozen=True, slots=True)
class WaveSpectrum:
    """Discrete deep-water spectrum uploaded directly to the vertex shader."""

    components: np.ndarray
    phases: np.ndarray
    significant_wave_height_m: float


def build_directional_spectrum(
    config: WaterConfig, seed: int = 20241005
) -> WaveSpectrum:
    """Discretise a fetch-limited wind sea into directional wave components.

    The spectrum uses a Phillips equilibrium term, finite-fetch suppression,
    deep-water dispersion and cosine directional spreading. Its output is a
    compact real-time approximation rather than an artist-authored wave set.
    """

    if config.wind_speed_mps <= 0.0:
        return WaveSpectrum(
            np.zeros((config.wave_count, 4), dtype=np.float32),
            np.zeros(config.wave_count, dtype=np.float32),
            0.0,
        )
    rng = np.random.default_rng(seed)
    wavelengths = np.geomspace(
        config.maximum_wavelength_m,
        config.minimum_wavelength_m,
        config.wave_count,
    )
    wave_numbers = 2.0 * np.pi / wavelengths
    log_step = abs(math.log(wavelengths[-1] / wavelengths[0])) / max(
        config.wave_count - 1, 1
    )
    delta_k = wave_numbers * log_step

    wind_angle = math.radians(config.wind_direction_deg)
    travel_angle = wind_angle + math.pi
    wind_direction = np.array(
        [math.sin(travel_angle), -math.cos(travel_angle)], dtype=np.float64
    )
    peak_length = min(
        config.maximum_wavelength_m,
        max(0.5, 0.83 * config.wind_speed_mps**2 / GRAVITY_MPS2 * 2.0 * np.pi),
    )
    longest_supported = min(
        config.maximum_wavelength_m,
        max(
            config.minimum_wavelength_m,
            2.0 * np.pi * (config.fetch_length_m / 22_000.0) ** 0.44
            * config.wind_speed_mps**1.1,
        ),
    )
    peak_k = 2.0 * np.pi / min(peak_length, max(longest_supported, 0.5))

    angular_offsets = rng.normal(
        0.0,
        math.radians(32.0) / math.sqrt(config.directional_spread_power),
        config.wave_count,
    )
    angles = travel_angle + angular_offsets
    directions = np.column_stack((np.sin(angles), -np.cos(angles)))
    alignment = np.maximum(directions @ wind_direction, 0.0)
    directional = alignment**config.directional_spread_power

    largest_wave_m = config.wind_speed_mps**2 / GRAVITY_MPS2
    phillips_alpha = 0.0065
    spectrum_density = (
        phillips_alpha
        * np.exp(-1.0 / np.maximum((wave_numbers * largest_wave_m) ** 2, 1e-8))
        / np.maximum(wave_numbers**4, 1e-8)
        * directional
    )
    # Suppress capillary-scale energy and wavelengths not developed by fetch.
    spectrum_density *= np.exp(-(wave_numbers * 0.08) ** 2)
    spectrum_density *= np.exp(-(peak_k / np.maximum(wave_numbers, 1e-6)) ** 4)
    amplitudes = np.sqrt(2.0 * spectrum_density * delta_k)
    amplitudes = np.minimum(amplitudes, 0.14 / np.maximum(wave_numbers, 1e-6))

    components = np.column_stack(
        (directions[:, 0], directions[:, 1], wave_numbers, amplitudes)
    ).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, config.wave_count).astype(np.float32)
    significant_height = 4.0 * math.sqrt(
        max(0.5 * float(np.sum(amplitudes**2)), 0.0)
    )
    return WaveSpectrum(components, phases, significant_height)


def build_water_mesh(config: WaterConfig) -> tuple[np.ndarray, np.ndarray]:
    columns, rows = config.grid_size
    width, depth = config.extent_m
    x = np.linspace(-width * 0.5, width * 0.5, columns, dtype=np.float32)
    z = np.linspace(-depth * 0.35, depth * 0.65, rows, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)
    vertices = np.column_stack((xx.ravel(), zz.ravel())).astype(np.float32)
    index_rows = []
    for row in range(rows - 1):
        left = row * columns
        for column in range(columns - 1):
            a = left + column
            b = a + columns
            index_rows.extend((a, b, a + 1, a + 1, b, b + 1))
    return vertices, np.asarray(index_rows, dtype=np.uint32)
