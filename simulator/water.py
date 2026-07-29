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
    minimum_wavelength_m: float = 0.12
    maximum_wavelength_m: float = 30.0
    directional_spread_power: float = 8.0
    choppiness: float = 0.72
    wind_response_time_s: float = 180.0
    atmosphere_update_interval_s: float = 2.0
    grid_size: tuple[int, int] = (181, 129)
    extent_m: tuple[float, float] = (1_200.0, 900.0)
    far_grid_size: tuple[int, int] = (121, 97)
    far_extent_m: tuple[float, float] = (5_000.0, 4_000.0)


@dataclass(frozen=True, slots=True)
class WaveSpectrum:
    """Discrete deep-water spectrum uploaded directly to the vertex shader."""

    components: np.ndarray
    phases: np.ndarray
    significant_wave_height_m: float


def relax_wave_spectrum(
    current: WaveSpectrum,
    target: WaveSpectrum,
    dt_s: float,
    response_time_s: float,
) -> WaveSpectrum:
    """Relax a wind sea toward new forcing without popping wave phases."""

    alpha = 1.0 - math.exp(-max(dt_s, 0.0) / max(response_time_s, 1e-6))
    components = current.components + alpha * (
        target.components - current.components
    )
    directions = components[:, :2]
    direction_lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    components[:, :2] = directions / np.maximum(direction_lengths, 1e-7)
    amplitudes = components[:, 3]
    significant_height = 4.0 * math.sqrt(
        max(0.5 * float(np.sum(amplitudes**2)), 0.0)
    )
    return WaveSpectrum(
        components.astype(np.float32, copy=False),
        current.phases,
        significant_height,
    )


def build_directional_spectrum(
    config: WaterConfig, seed: int = 20241005
) -> WaveSpectrum:
    """Discretise a fetch-limited wind sea into directional wave components.

    The spectrum uses the fetch-limited JONSWAP formulation with deep-water
    dispersion. Its output is a compact real-time approximation rather than
    an artist-authored wave set.
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
    angular_frequencies = np.sqrt(GRAVITY_MPS2 * wave_numbers)
    delta_omega = np.gradient(angular_frequencies)

    wind_angle = math.radians(config.wind_direction_deg)
    travel_angle = wind_angle + math.pi
    angular_offsets = rng.normal(
        0.0,
        math.radians(32.0) / math.sqrt(config.directional_spread_power),
        config.wave_count,
    )
    angles = travel_angle + angular_offsets
    directions = np.column_stack((np.sin(angles), -np.cos(angles)))

    dimensionless_fetch = max(
        GRAVITY_MPS2
        * config.fetch_length_m
        / max(config.wind_speed_mps**2, 1e-6),
        1.0,
    )
    peak_omega = (
        22.0
        * GRAVITY_MPS2
        / config.wind_speed_mps
        * dimensionless_fetch ** -0.33
    )
    alpha = float(np.clip(
        0.076 * dimensionless_fetch ** -0.22, 0.001, 0.02
    ))
    sigma = np.where(angular_frequencies <= peak_omega, 0.07, 0.09)
    peak_shape = np.exp(
        -(
            (angular_frequencies - peak_omega) ** 2
            / (2.0 * sigma**2 * peak_omega**2)
        )
    )
    spectrum_omega = (
        alpha
        * GRAVITY_MPS2**2
        / np.maximum(angular_frequencies**5, 1e-9)
        * np.exp(
            -1.25
            * (peak_omega / np.maximum(angular_frequencies, 1e-6)) ** 4
        )
        * 3.3**peak_shape
    )
    spectrum_omega *= np.exp(-(wave_numbers * 0.035) ** 2)
    amplitudes = np.sqrt(
        2.0 * spectrum_omega * np.maximum(delta_omega, 0.0)
    )

    components = np.column_stack(
        (directions[:, 0], directions[:, 1], wave_numbers, amplitudes)
    ).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, config.wave_count).astype(np.float32)
    significant_height = 4.0 * math.sqrt(
        max(0.5 * float(np.sum(amplitudes**2)), 0.0)
    )
    return WaveSpectrum(components, phases, significant_height)


def estimate_fetch_length_m(
    water_mask: np.ndarray,
    bounds: np.ndarray,
    wind_velocity_xz_mps: np.ndarray,
    origin_xz_m: tuple[float, float] = (0.0, 0.0),
) -> float:
    velocity = np.asarray(wind_velocity_xz_mps, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        return 100.0
    upwind = -velocity / speed
    minimum_x, minimum_z, maximum_x, maximum_z = map(float, bounds)
    metres_per_pixel = max(
        (maximum_x - minimum_x) / water_mask.shape[1],
        (maximum_z - minimum_z) / water_mask.shape[0],
    )
    step_m = max(metres_per_pixel * 0.5, 1.0)
    maximum_distance = math.hypot(
        maximum_x - minimum_x, maximum_z - minimum_z
    )
    for distance in np.arange(0.0, maximum_distance, step_m):
        point = np.asarray(origin_xz_m) + upwind * distance
        u = (point[0] - minimum_x) / (maximum_x - minimum_x)
        v = (point[1] - minimum_z) / (maximum_z - minimum_z)
        if u < 0.0 or u >= 1.0 or v < 0.0 or v >= 1.0:
            return max(float(distance), 20.0)
        x = min(int(u * water_mask.shape[1]), water_mask.shape[1] - 1)
        y = min(int(v * water_mask.shape[0]), water_mask.shape[0] - 1)
        if water_mask[y, x] < 128:
            return max(float(distance), 20.0)
    return maximum_distance


def build_water_mesh(
    config: WaterConfig,
    grid_size: tuple[int, int] | None = None,
    extent_m: tuple[float, float] | None = None,
    exclude_extent_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    columns, rows = grid_size or config.grid_size
    width, depth = extent_m or config.extent_m
    x = np.linspace(-width * 0.5, width * 0.5, columns, dtype=np.float32)
    z = np.linspace(-depth * 0.5, depth * 0.5, rows, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)
    vertices = np.column_stack((xx.ravel(), zz.ravel())).astype(np.float32)
    index_rows = []
    for row in range(rows - 1):
        left = row * columns
        for column in range(columns - 1):
            a = left + column
            b = a + columns
            if exclude_extent_m is not None:
                centre = 0.25 * (
                    vertices[a] + vertices[a + 1]
                    + vertices[b] + vertices[b + 1]
                )
                if (
                    abs(float(centre[0])) < exclude_extent_m[0] * 0.5
                    and abs(float(centre[1])) < exclude_extent_m[1] * 0.5
                ):
                    continue
            index_rows.extend((a, b, a + 1, a + 1, b, b + 1))
    return vertices, np.asarray(index_rows, dtype=np.uint32)
