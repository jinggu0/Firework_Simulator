from __future__ import annotations

import moderngl
import numpy as np
import pygame
import pytest

from simulator.config import AtmosphereConfig, SmokeConfig
from simulator.gpu_fluid import GpuSmokeFluid2D


@pytest.fixture(scope="module")
def gl_context() -> moderngl.Context:
    try:
        pygame.init()
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_MAJOR_VERSION, 3
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
        pytest.skip(f"OpenGL 3.3 unavailable: {error}")
    yield context
    pygame.quit()


def test_gpu_mac_projection_reduces_divergence(
    gl_context: moderngl.Context,
) -> None:
    config = SmokeConfig(
        grid_size=(48, 36),
        bounds_m=(-60.0, 60.0, 0.0, 90.0),
        pressure_iterations=100,
        vorticity_confinement=0.0,
    )
    fluid = GpuSmokeFluid2D(gl_context, config, AtmosphereConfig())
    rng = np.random.default_rng(9)
    fluid.u_mps[:, 1:-1] = rng.normal(
        0.0, 1.0, fluid.u_mps[:, 1:-1].shape
    )
    fluid.v_mps[1:-1, :] = rng.normal(
        0.0, 1.0, fluid.v_mps[1:-1, :].shape
    )
    before = float(np.sqrt(np.mean(fluid.divergence() ** 2)))
    fluid._upload_initial_state()
    fluid.step(1.0 / 30.0)
    fluid.readback()
    after = float(np.sqrt(np.mean(fluid.divergence() ** 2)))
    assert after < before * 0.12
    assert np.isfinite(fluid.u_mps).all()
    assert np.isfinite(fluid.v_mps).all()


def test_gpu_step_retains_injected_mass(
    gl_context: moderngl.Context,
) -> None:
    config = SmokeConfig(
        grid_size=(48, 36),
        bounds_m=(-60.0, 60.0, 0.0, 90.0),
        vorticity_confinement=0.0,
    )
    fluid = GpuSmokeFluid2D(gl_context, config, AtmosphereConfig())
    smoke_mass_kg = 0.012
    fluid.inject_burst(
        np.array([0.0, 45.0, 0.0]),
        smoke_mass_kg,
        50_000.0,
    )
    fluid.step(1.0 / 30.0)
    fluid.readback()
    represented_mass_kg = float(fluid.density_kg_m3.sum()) * (
        fluid.dx * fluid.dy * config.plume_depth_m
    )
    expected_decay = np.exp(
        -np.log(2.0) / config.update_hz / config.smoke_half_life_s
    )
    assert np.isclose(
        represented_mass_kg,
        smoke_mass_kg * expected_decay,
        rtol=2e-3,
    )


def test_gpu_hot_plume_rises_and_advects_with_wind(
    gl_context: moderngl.Context,
) -> None:
    config = SmokeConfig(
        grid_size=(48, 36),
        bounds_m=(-60.0, 60.0, 0.0, 90.0),
        vorticity_confinement=0.0,
    )
    atmosphere = AtmosphereConfig(
        wind_velocity_mps=(5.0, 0.0, 0.0),
        wind_velocity_100m_mps=(5.0, 0.0, 0.0),
    )
    fluid = GpuSmokeFluid2D(gl_context, config, atmosphere)
    fluid.inject_burst(
        np.array([-20.0, 25.0, 0.0]), 0.015, 600_000.0
    )
    initial_density = fluid.density_kg_m3
    initial_total = float(initial_density.sum())
    start_x = float(
        (initial_density * fluid._cell_x).sum() / initial_total
    )
    start_y = float(
        (initial_density * fluid._cell_y).sum() / initial_total
    )
    for _ in range(30):
        fluid.step(1.0 / 30.0)
    fluid.readback()
    final_density = fluid.density_kg_m3
    final_total = float(final_density.sum())
    end_x = float((final_density * fluid._cell_x).sum() / final_total)
    end_y = float((final_density * fluid._cell_y).sum() / final_total)
    assert end_x > start_x + 3.0
    assert end_y > start_y
