from __future__ import annotations

import moderngl
import numpy as np
import pygame
import pytest

from simulator.config import AtmosphereConfig, SmokeConfig
from simulator.gpu_fluid_3d import GpuSmokeFluid3D


@pytest.fixture(scope="module")
def compute_context() -> moderngl.Context:
    try:
        pygame.init()
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MAJOR_VERSION, 4
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MINOR_VERSION, 3
        )
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE,
        )
        pygame.display.set_mode(
            (64, 64), pygame.OPENGL | pygame.HIDDEN, vsync=0
        )
        context = moderngl.create_context()
    except (pygame.error, moderngl.Error) as error:
        pytest.skip(f"OpenGL 4.3 unavailable: {error}")
    yield context
    pygame.quit()


def _divergence(
    fluid: GpuSmokeFluid3D,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    return (
        (u[:, :, 1:] - u[:, :, :-1]) / fluid.dx
        + (v[:, 1:, :] - v[:, :-1, :]) / fluid.dy
        + (w[1:, :, :] - w[:-1, :, :]) / fluid.dz
    )


def test_3d_mac_projection_reduces_divergence(
    compute_context: moderngl.Context,
) -> None:
    config = SmokeConfig(
        gpu_3d_grid_size=(24, 16, 10),
        pressure_iterations=100,
        vorticity_confinement=0.0,
    )
    fluid = GpuSmokeFluid3D(
        compute_context, config, AtmosphereConfig()
    )
    rng = np.random.default_rng(19)
    u = rng.normal(0.0, 1.0, (10, 16, 25)).astype(np.float32)
    v = rng.normal(0.0, 1.0, (10, 17, 24)).astype(np.float32)
    w = rng.normal(0.0, 1.0, (11, 16, 24)).astype(np.float32)
    before = float(np.sqrt(np.mean(_divergence(fluid, u, v, w) ** 2)))
    fluid.upload_velocity_fields(u, v, w)
    fluid.step(1.0 / fluid.update_hz)
    projected = fluid.readback_velocity_fields()
    after = float(
        np.sqrt(np.mean(_divergence(fluid, *projected) ** 2))
    )
    assert after < before * 0.25


def test_3d_burst_source_conserves_mass(
    compute_context: moderngl.Context,
) -> None:
    config = SmokeConfig(gpu_3d_grid_size=(24, 16, 10))
    fluid = GpuSmokeFluid3D(
        compute_context, config, AtmosphereConfig()
    )
    smoke_mass_kg = 0.012
    fluid.inject_burst(
        np.array([0.0, 150.0, 0.0]), smoke_mass_kg, 50_000.0
    )
    fluid.step(1.0 / fluid.update_hz)
    density = fluid.readback_state()[..., 0]
    represented_mass_kg = float(density.sum()) * (
        fluid.dx * fluid.dy * fluid.dz
    )
    expected = smoke_mass_kg * np.exp(
        -np.log(2.0) / fluid.update_hz / config.smoke_half_life_s
    )
    assert np.isclose(represented_mass_kg, expected, rtol=4e-3)


def test_3d_plume_moves_up_and_crosswind(
    compute_context: moderngl.Context,
) -> None:
    config = SmokeConfig(
        gpu_3d_grid_size=(24, 16, 10),
        vorticity_confinement=0.0,
    )
    atmosphere = AtmosphereConfig(
        wind_velocity_mps=(0.0, 0.0, 5.0),
        wind_velocity_100m_mps=(0.0, 0.0, 5.0),
    )
    fluid = GpuSmokeFluid3D(compute_context, config, atmosphere)
    fluid.inject_burst(
        np.array([0.0, 120.0, -20.0]), 0.015, 600_000.0
    )
    for _ in range(fluid.update_hz):
        fluid.step(1.0 / fluid.update_hz)
    density = fluid.readback_state()[..., 0]
    z = fluid.z_min + (np.arange(fluid.nz) + 0.5) * fluid.dz
    y = fluid.y_min + (np.arange(fluid.ny) + 0.5) * fluid.dy
    total = float(density.sum())
    centre_y = float((density * y[None, :, None]).sum() / total)
    centre_z = float((density * z[:, None, None]).sum() / total)
    assert centre_y > 120.0
    assert centre_z > -17.0
