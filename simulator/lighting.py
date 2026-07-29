from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .config import LightingConfig


class LightSourceKind(Enum):
    ELECTRIC_LED = "electric_led"
    COMBUSTION = "combustion"
    CELESTIAL = "celestial"


@dataclass(frozen=True, slots=True)
class LedEnergyBudget:
    input_electrical_w: float
    driver_output_w: float
    junction_radiant_w: float
    extracted_radiant_w: float
    converted_radiant_w: float
    luminaire_radiant_w: float
    luminous_flux_lm: float
    window_radiance_w_m2_sr: float


@dataclass(frozen=True, slots=True)
class DynamicLights:
    positions_m: np.ndarray
    colors: np.ndarray
    radiant_power_w: np.ndarray
    count: int


def led_energy_budget(config: LightingConfig) -> LedEnergyBudget:
    driver = config.led_input_power_w * config.led_driver_efficiency
    junction = (
        driver
        * config.led_internal_quantum_efficiency
        * config.led_electrical_efficiency
    )
    extracted = junction * config.led_extraction_efficiency
    converted = extracted * config.led_phosphor_efficiency
    luminaire = converted * config.luminaire_optical_efficiency
    luminous_flux = (
        luminaire * config.led_spectral_luminous_efficacy_lm_w
    )
    window_radiance = (
        luminaire
        * config.interior_utilization_factor
        * config.window_visible_transmittance
        / (math.pi * config.window_emitting_area_m2)
    )
    return LedEnergyBudget(
        config.led_input_power_w,
        driver,
        junction,
        extracted,
        converted,
        luminaire,
        luminous_flux,
        window_radiance,
    )


def burn_profile(normalized_age: np.ndarray) -> np.ndarray:
    age = np.clip(normalized_age, 0.0, 1.0)
    ignition_x = np.clip(age / 0.018, 0.0, 1.0)
    ignition = ignition_x * ignition_x * (3.0 - 2.0 * ignition_x)
    extinction_x = np.clip((1.0 - age) / 0.12, 0.0, 1.0)
    extinction = extinction_x * extinction_x * (3.0 - 2.0 * extinction_x)
    shrinking_surface = np.clip(1.0 - 0.32 * age, 0.0, 1.0)
    return ignition * extinction * shrinking_surface


_PROFILE_AGES = np.linspace(0.0, 1.0, 8193, dtype=np.float64)
BURN_PROFILE_INTEGRAL = float(
    np.trapezoid(burn_profile(_PROFILE_AGES), _PROFILE_AGES)
)


def combustion_peak_radiant_power_w(
    chemical_energy_j: np.ndarray,
    lifetime_s: np.ndarray,
    radiative_fraction: float,
) -> np.ndarray:
    return (
        np.asarray(chemical_energy_j)
        * radiative_fraction
        / (
            np.maximum(np.asarray(lifetime_s), 1e-6)
            * BURN_PROFILE_INTEGRAL
            * 0.97
        )
    )


def radiometric_irradiance_from_illuminance(
    illuminance_lux: float, spectral_luminous_efficacy_lm_w: float
) -> float:
    return max(illuminance_lux, 0.0) / max(
        spectral_luminous_efficacy_lm_w, 1e-6
    )


def cluster_radiant_lights(
    positions_m: np.ndarray,
    colors: np.ndarray,
    radiant_power_w: np.ndarray,
    maximum_lights: int = 8,
) -> DynamicLights:
    positions = np.asarray(positions_m)
    colors = np.asarray(colors)
    power = np.asarray(radiant_power_w)
    output_positions = np.zeros((maximum_lights, 3), dtype=np.float32)
    output_colors = np.zeros((maximum_lights, 3), dtype=np.float32)
    output_power = np.zeros(maximum_lights, dtype=np.float32)
    active = power > 1e-6
    if not np.any(active):
        return DynamicLights(
            output_positions, output_colors, output_power, 0
        )
    positions, colors, power = positions[active], colors[active], power[active]
    total_power = float(np.sum(power, dtype=np.float64))
    centroid = np.sum(
        positions * power[:, None], axis=0, dtype=np.float64
    ) / total_power
    if maximum_lights >= 8:
        relative = positions - centroid
        bins = (
            (relative[:, 0] >= 0.0).astype(np.int32)
            + 2 * (relative[:, 1] >= 0.0).astype(np.int32)
            + 4 * (relative[:, 2] >= 0.0).astype(np.int32)
        )
        bin_count = 8
    else:
        bins = np.zeros(len(power), dtype=np.int32)
        bin_count = 1
    bin_power = np.bincount(
        bins, weights=power, minlength=bin_count
    ).astype(np.float64)
    populated_bins = np.flatnonzero(bin_power > 1e-6)[:maximum_lights]
    write_index = len(populated_bins)
    if write_index:
        position_sums = np.column_stack(
            [
                np.bincount(
                    bins,
                    weights=positions[:, axis] * power,
                    minlength=bin_count,
                )
                for axis in range(3)
            ]
        )
        color_sums = np.column_stack(
            [
                np.bincount(
                    bins,
                    weights=colors[:, axis] * power,
                    minlength=bin_count,
                )
                for axis in range(3)
            ]
        )
        selected_power = bin_power[populated_bins]
        output_power[:write_index] = selected_power
        output_positions[:write_index] = (
            position_sums[populated_bins] / selected_power[:, None]
        )
        output_colors[:write_index] = (
            color_sums[populated_bins] / selected_power[:, None]
        )
    return DynamicLights(
        output_positions, output_colors, output_power, write_index
    )
