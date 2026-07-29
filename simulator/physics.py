from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import blackbody_rgb
from .config import AtmosphereConfig, ShellConfig

GRAVITY_MPS2 = np.array([0.0, -9.80665, 0.0], dtype=np.float32)


@dataclass(slots=True)
class Shell:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    age_s: float = 0.0
    alive: bool = True


@dataclass(frozen=True, slots=True)
class BurstEvent:
    position_m: np.ndarray
    smoke_mass_kg: float
    post_blast_thermal_energy_j: float


class StarField:
    """Structure-of-arrays storage for vectorized star integration."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.count = 0
        self.position_m = np.empty((capacity, 3), dtype=np.float32)
        self.previous_position_m = np.empty((capacity, 3), dtype=np.float32)
        self.velocity_mps = np.empty((capacity, 3), dtype=np.float32)
        self.age_s = np.empty(capacity, dtype=np.float32)
        self.lifetime_s = np.empty(capacity, dtype=np.float32)
        self.drag_time_s = np.empty(capacity, dtype=np.float32)
        self.luminous_power_w = np.empty(capacity, dtype=np.float32)
        self.color_linear = np.empty((capacity, 3), dtype=np.float32)

    def clear(self) -> None:
        self.count = 0

    def spawn_burst(
        self,
        origin_m: np.ndarray,
        shell_velocity_mps: np.ndarray,
        config: ShellConfig,
        rng: np.random.Generator,
    ) -> None:
        n = min(config.burst_star_count, self.capacity - self.count)
        if n <= 0:
            return

        start, end = self.count, self.count + n
        directions = rng.normal(size=(n, 3)).astype(np.float32)
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-7
        )
        speeds = np.maximum(
            rng.normal(config.star_speed_mean_mps, config.star_speed_std_mps, n),
            0.1,
        ).astype(np.float32)

        self.position_m[start:end] = origin_m
        self.previous_position_m[start:end] = origin_m
        self.velocity_mps[start:end] = (
            directions * speeds[:, None] + shell_velocity_mps[None, :]
        )
        self.age_s[start:end] = 0.0
        self.lifetime_s[start:end] = np.maximum(
            rng.normal(
                config.star_lifetime_mean_s, config.star_lifetime_std_s, n
            ),
            0.05,
        )
        self.drag_time_s[start:end] = config.star_drag_time_s
        self.luminous_power_w[start:end] = config.luminous_power_w
        self.color_linear[start:end] = blackbody_rgb(config.color_temperature_k)
        self.count = end

    def update(self, dt_s: float, atmosphere: AtmosphereConfig) -> None:
        n = self.count
        if n == 0:
            return

        self.previous_position_m[:n] = self.position_m[:n]
        wind_10m = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float32)
        wind_100m = np.asarray(
            atmosphere.wind_velocity_100m_mps, dtype=np.float32
        )
        heights = np.maximum(self.position_m[:n, 1], 10.0)
        wind_alpha = np.clip(np.log(heights / 10.0) / np.log(10.0), 0.0, 1.0)
        wind = wind_10m + (wind_100m - wind_10m) * wind_alpha[:, None]
        relative_velocity = self.velocity_mps[:n] - wind
        # Exponential relaxation is stable for large dt and approximates the
        # strongly size-dependent drag of burning pyrotechnic stars.
        drag = np.exp(-dt_s / self.drag_time_s[:n])[:, None]
        self.velocity_mps[:n] = wind + relative_velocity * drag
        self.velocity_mps[:n] += GRAVITY_MPS2 * dt_s
        self.position_m[:n] += self.velocity_mps[:n] * dt_s
        self.age_s[:n] += dt_s

        alive = (self.age_s[:n] < self.lifetime_s[:n]) & (
            self.position_m[:n, 1] > -2.0
        )
        alive_count = int(np.count_nonzero(alive))
        if alive_count != n:
            for array in (
                self.position_m,
                self.previous_position_m,
                self.velocity_mps,
                self.age_s,
                self.lifetime_s,
                self.drag_time_s,
                self.luminous_power_w,
                self.color_linear,
            ):
                array[:alive_count] = array[:n][alive]
            self.count = alive_count

    def intensity(self) -> np.ndarray:
        """Return instantaneous radiant output including ignition and decay."""

        n = self.count
        normalized_age = self.age_s[:n] / self.lifetime_s[:n]
        ignition_x = np.clip(normalized_age / 0.018, 0.0, 1.0)
        ignition = ignition_x * ignition_x * (3.0 - 2.0 * ignition_x)
        extinction_x = np.clip((1.0 - normalized_age) / 0.12, 0.0, 1.0)
        extinction = extinction_x * extinction_x * (3.0 - 2.0 * extinction_x)
        shrinking_surface = np.clip(1.0 - 0.32 * normalized_age, 0.0, 1.0)
        stochastic_burn = 0.97 + 0.03 * np.sin(
            self.age_s[:n] * 53.0 + np.arange(n, dtype=np.float32) * 1.618
        )
        return (
            self.luminous_power_w[:n]
            * ignition
            * extinction
            * shrinking_surface
            * stochastic_burn
        )


class FireworkWorld:
    def __init__(
        self,
        atmosphere: AtmosphereConfig,
        shell_config: ShellConfig,
        max_particles: int,
        seed: int,
    ) -> None:
        self.atmosphere = atmosphere
        self.shell_config = shell_config
        self.stars = StarField(max_particles)
        self.shells: list[Shell] = []
        self.rng = np.random.default_rng(seed)
        self._burst_events: list[BurstEvent] = []

    def consume_burst_events(self) -> list[BurstEvent]:
        events, self._burst_events = self._burst_events, []
        return events

    def launch(self, position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.shells.append(
            Shell(
                position_m=np.asarray(position_m, dtype=np.float32),
                velocity_mps=np.array(
                    [0.0, self.shell_config.launch_speed_mps, 0.0],
                    dtype=np.float32,
                ),
            )
        )

    def update(self, dt_s: float) -> None:
        config = self.shell_config
        cross_section_m2 = np.pi * (config.diameter_m * 0.5) ** 2
        drag_factor = (
            0.5
            * self.atmosphere.air_density_kg_m3
            * config.drag_coefficient
            * cross_section_m2
            / config.dry_mass_kg
        )

        surviving_shells: list[Shell] = []
        for shell in self.shells:
            wind = np.asarray(
                self.atmosphere.wind_at_height_m(float(shell.position_m[1])),
                dtype=np.float32,
            )
            relative_velocity = shell.velocity_mps - wind
            speed = float(np.linalg.norm(relative_velocity))
            acceleration = GRAVITY_MPS2 - drag_factor * speed * relative_velocity
            shell.velocity_mps += acceleration * dt_s
            shell.position_m += shell.velocity_mps * dt_s
            shell.age_s += dt_s
            if shell.age_s >= config.fuse_delay_s:
                self.stars.spawn_burst(
                    shell.position_m, shell.velocity_mps, config, self.rng
                )
                chemical_energy_j = (
                    config.burst_charge_mass_kg
                    * config.burst_specific_energy_j_kg
                )
                self._burst_events.append(
                    BurstEvent(
                        shell.position_m.copy(),
                        config.burst_charge_mass_kg
                        * config.smoke_yield_fraction,
                        chemical_energy_j * config.post_blast_thermal_fraction,
                    )
                )
            else:
                surviving_shells.append(shell)

        self.shells = surviving_shells
        self.stars.update(dt_s, self.atmosphere)
