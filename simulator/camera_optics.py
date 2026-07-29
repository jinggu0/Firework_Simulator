from __future__ import annotations

import math

import numpy as np

from .config import PhysicalCameraConfig

PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_S = 299_792_458.0


def vertical_fov_deg(config: PhysicalCameraConfig) -> float:
    return math.degrees(
        2.0
        * math.atan(
            config.sensor_height_mm / (2.0 * config.focal_length_mm)
        )
    )


def photon_to_electron_scale(config: PhysicalCameraConfig) -> np.ndarray:
    aperture_factor = (
        math.pi
        / (4.0 * config.f_number**2)
        * config.lens_transmission
    )
    pixel_area_m2 = (config.pixel_pitch_um * 1e-6) ** 2
    wavelengths_m = (
        np.asarray(config.wavelength_rgb_nm, dtype=np.float64) * 1e-9
    )
    quantum_efficiency = np.asarray(
        config.quantum_efficiency_rgb, dtype=np.float64
    )
    photons_per_joule = wavelengths_m / (
        PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S
    )
    return (
        aperture_factor
        * config.shutter_time_s
        * pixel_area_m2
        * photons_per_joule
        * quantum_efficiency
    ).astype(np.float32)


def analog_gain(config: PhysicalCameraConfig) -> float:
    return config.iso / config.base_iso
