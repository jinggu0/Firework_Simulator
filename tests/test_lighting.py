import numpy as np

from simulator.camera_optics import (
    analog_gain,
    photon_to_electron_scale,
    vertical_fov_deg,
)
from simulator.config import LightingConfig, PhysicalCameraConfig
from simulator.lighting import (
    BURN_PROFILE_INTEGRAL,
    cluster_radiant_lights,
    combustion_peak_radiant_power_w,
    led_energy_budget,
)


def test_led_energy_chain_conserves_and_loses_power() -> None:
    budget = led_energy_budget(LightingConfig())
    stages = (
        budget.input_electrical_w,
        budget.driver_output_w,
        budget.junction_radiant_w,
        budget.extracted_radiant_w,
        budget.converted_radiant_w,
        budget.luminaire_radiant_w,
    )
    assert all(left >= right for left, right in zip(stages, stages[1:]))
    assert 100.0 < budget.luminous_flux_lm / stages[0] < 150.0
    assert budget.window_radiance_w_m2_sr > 0.0


def test_combustion_power_integrates_to_radiative_energy() -> None:
    chemical_energy = np.array([400.0])
    lifetime = np.array([2.0])
    peak = combustion_peak_radiant_power_w(
        chemical_energy, lifetime, 0.15
    )
    recovered = peak[0] * lifetime[0] * BURN_PROFILE_INTEGRAL * 0.97
    assert np.isclose(recovered, 60.0)


def test_light_clustering_conserves_radiant_power() -> None:
    positions = np.array(
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    )
    colors = np.ones((3, 3))
    powers = np.array([10.0, 20.0, 30.0])
    lights = cluster_radiant_lights(positions, colors, powers)
    assert lights.count == 2
    assert np.isclose(lights.radiant_power_w.sum(), powers.sum())


def test_physical_camera_uses_lens_and_sensor_parameters() -> None:
    config = PhysicalCameraConfig()
    assert 40.0 < vertical_fov_deg(config) < 50.0
    assert np.all(photon_to_electron_scale(config) > 0.0)
    assert analog_gain(config) == 8.0
