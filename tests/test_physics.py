import numpy as np

from simulator.config import AtmosphereConfig, ShellConfig
from simulator.physics import FireworkWorld


def make_world(seed: int = 20241005) -> FireworkWorld:
    return FireworkWorld(
        AtmosphereConfig(),
        ShellConfig(burst_star_count=256),
        max_particles=1024,
        seed=seed,
    )


def test_shell_bursts_at_configured_fuse_time() -> None:
    world = make_world()
    world.launch()
    dt = 1.0 / 120.0
    for _ in range(367):
        world.update(dt)
    assert not world.shells
    assert world.stars.count == 256
    assert 80.0 < float(world.stars.position_m[:256, 1].mean()) < 250.0
    events = world.consume_burst_events()
    assert len(events) == 1
    assert events[0].smoke_mass_kg > 0.0
    assert events[0].post_blast_thermal_energy_j > 0.0
    assert world.consume_burst_events() == []


def test_fixed_seed_produces_replayable_burst() -> None:
    a, b = make_world(), make_world()
    a.launch()
    b.launch()
    for _ in range(400):
        a.update(1.0 / 120.0)
        b.update(1.0 / 120.0)
    np.testing.assert_array_equal(
        a.stars.position_m[: a.stars.count],
        b.stars.position_m[: b.stars.count],
    )


def test_wind_advects_burning_stars() -> None:
    windy = FireworkWorld(
        AtmosphereConfig(wind_velocity_mps=(8.0, 0.0, 0.0)),
        ShellConfig(fuse_delay_s=0.0, burst_star_count=512),
        1024,
        7,
    )
    calm = FireworkWorld(
        AtmosphereConfig(wind_velocity_mps=(0.0, 0.0, 0.0)),
        ShellConfig(fuse_delay_s=0.0, burst_star_count=512),
        1024,
        7,
    )
    windy.launch()
    calm.launch()
    for _ in range(120):
        windy.update(1.0 / 120.0)
        calm.update(1.0 / 120.0)
    assert windy.stars.position_m[:512, 0].mean() > (
        calm.stars.position_m[:512, 0].mean() + 1.0
    )
