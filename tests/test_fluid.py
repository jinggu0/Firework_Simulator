import numpy as np

from simulator.config import AtmosphereConfig, SmokeConfig
from simulator.fluid import SmokeFluid2D


def small_fluid(wind_mps: float = 0.0) -> SmokeFluid2D:
    config = SmokeConfig(
        grid_size=(48, 36),
        bounds_m=(-60.0, 60.0, 0.0, 90.0),
        pressure_iterations=100,
        vorticity_confinement=0.0,
    )
    atmosphere = AtmosphereConfig(
        wind_velocity_mps=(wind_mps, 0.0, 0.0),
        wind_velocity_100m_mps=(wind_mps, 0.0, 0.0),
    )
    return SmokeFluid2D(config, atmosphere)


def centroid(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    total = float(field.sum())
    return float((field * x).sum() / total), float((field * y).sum() / total)


def test_pressure_projection_reduces_divergence() -> None:
    fluid = small_fluid()
    rng = np.random.default_rng(9)
    fluid.u_mps[:, 1:-1] = rng.normal(0.0, 1.0, fluid.u_mps[:, 1:-1].shape)
    fluid.v_mps[1:-1, :] = rng.normal(0.0, 1.0, fluid.v_mps[1:-1, :].shape)
    before = float(np.sqrt(np.mean(fluid.divergence() ** 2)))
    fluid.project(1.0 / 30.0)
    after = float(np.sqrt(np.mean(fluid.divergence() ** 2)))
    assert after < before * 0.45


def test_burst_source_conserves_smoke_mass_in_slice() -> None:
    fluid = small_fluid()
    smoke_mass = 0.012
    fluid.inject_burst(np.array([0.0, 45.0, 0.0]), smoke_mass, 50_000.0)
    represented_mass = float(fluid.density_kg_m3.sum()) * (
        fluid.dx * fluid.dy * fluid.config.plume_depth_m
    )
    assert np.isclose(represented_mass, smoke_mass, rtol=2e-5)
    assert np.isfinite(fluid.temperature_excess_k).all()


def test_particle_deposition_conserves_interior_mass_and_energy() -> None:
    fluid = small_fluid()
    positions = np.array(
        [[-20.0, 20.0, 3.0], [0.0, 45.0, -8.0], [22.0, 70.0, 5.0]],
        dtype=np.float32,
    )
    smoke_mass = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    thermal_energy = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    accepted_mass, accepted_energy = fluid.inject_particles(
        positions, smoke_mass, thermal_energy
    )
    represented_mass = float(fluid.density_kg_m3.sum()) * (
        fluid.dx * fluid.dy * fluid.config.plume_depth_m
    )
    represented_energy = float(fluid.temperature_excess_k.sum()) * (
        fluid.air_density_kg_m3
        * fluid.dx
        * fluid.dy
        * fluid.config.plume_depth_m
        * 1005.0
    )
    assert np.isclose(accepted_mass, smoke_mass.sum(), rtol=1e-6)
    assert np.isclose(accepted_energy, thermal_energy.sum(), rtol=1e-6)
    assert np.isclose(represented_mass, smoke_mass.sum(), rtol=1e-6)
    assert np.isclose(represented_energy, thermal_energy.sum(), rtol=1e-5)


def test_volume_reconstruction_preserves_column_mass_and_heat() -> None:
    fluid = small_fluid()
    fluid.inject_burst(np.array([0.0, 45.0, 0.0]), 0.012, 50_000.0)
    density, temperature = fluid.reconstruct_volume()
    dz = fluid.config.volume_depth_m / fluid.config.volume_slices
    expected_density_column = (
        fluid.density_kg_m3 * fluid.config.plume_depth_m
    )
    expected_temperature_column = (
        fluid.temperature_excess_k * fluid.config.plume_depth_m
    )
    np.testing.assert_allclose(
        density.sum(axis=0) * dz, expected_density_column, rtol=2e-6
    )
    np.testing.assert_allclose(
        temperature.sum(axis=0) * dz,
        expected_temperature_column,
        rtol=2e-6,
    )


def test_hot_plume_rises_and_wind_advects_it() -> None:
    fluid = small_fluid(wind_mps=5.0)
    fluid.inject_burst(np.array([-20.0, 25.0, 0.0]), 0.015, 600_000.0)
    start_x, start_y = centroid(
        fluid.density_kg_m3, fluid._cell_x, fluid._cell_y
    )
    for _ in range(30):
        fluid.step(1.0 / 30.0)
    end_x, end_y = centroid(
        fluid.density_kg_m3, fluid._cell_x, fluid._cell_y
    )
    assert end_x > start_x + 3.0
    assert end_y > start_y
    assert np.isfinite(fluid.u_mps).all()
    assert np.isfinite(fluid.v_mps).all()
