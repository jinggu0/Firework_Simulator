from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import blackbody_rgb
from .config import AtmosphereConfig, ShellConfig
from .lighting import burn_profile, combustion_peak_radiant_power_w

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
    chemical_energy_j: float
    smoke_mass_kg: float
    post_blast_thermal_energy_j: float


@dataclass(frozen=True, slots=True)
class CombustionEmission:
    position_m: np.ndarray
    smoke_mass_kg: np.ndarray
    thermal_energy_j: np.ndarray


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
        self.fuel_mass_kg = np.empty(capacity, dtype=np.float32)
        self.emitted_burn_fraction = np.empty(capacity, dtype=np.float32)
        self.last_emission_position_m = np.empty(
            (capacity, 3), dtype=np.float32
        )

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
        self.color_linear[start:end] = blackbody_rgb(config.color_temperature_k)
        self.fuel_mass_kg[start:end] = (
            config.star_composition_mass_kg / config.burst_star_count
        )
        chemical_energy_j = (
            self.fuel_mass_kg[start:end] * config.star_specific_energy_j_kg
        )
        self.luminous_power_w[start:end] = combustion_peak_radiant_power_w(
            chemical_energy_j,
            self.lifetime_s[start:end],
            config.star_radiative_energy_fraction,
        )
        self.emitted_burn_fraction[start:end] = 0.0
        self.last_emission_position_m[start:end] = origin_m
        self.count = end

    def _make_emission(
        self,
        indices: np.ndarray,
        burned_fraction: np.ndarray,
        config: ShellConfig,
    ) -> CombustionEmission | None:
        delta = burned_fraction - self.emitted_burn_fraction[indices]
        emitting = delta > 1e-8
        if not np.any(emitting):
            return None
        selected = indices[emitting]
        burned_mass_kg = self.fuel_mass_kg[selected] * delta[emitting]
        positions = 0.5 * (
            self.last_emission_position_m[selected]
            + self.position_m[selected]
        )
        return CombustionEmission(
            positions.copy(),
            (
                burned_mass_kg * config.star_smoke_yield_fraction
            ).astype(np.float32),
            (
                burned_mass_kg
                * config.star_specific_energy_j_kg
                * config.star_post_combustion_thermal_fraction
            ).astype(np.float32),
        )

    def consume_emission(
        self, config: ShellConfig
    ) -> CombustionEmission | None:
        n = self.count
        if n == 0:
            return None
        indices = np.arange(n)
        normalized_age = np.clip(
            self.age_s[:n] / self.lifetime_s[:n], 0.0, 1.0
        )
        # A spherical star with an approximately constant linear regression
        # rate retains (1-t/t_burn)^3 of its initial reactive mass.
        burned_fraction = 1.0 - (1.0 - normalized_age) ** 3
        emission = self._make_emission(indices, burned_fraction, config)
        self.emitted_burn_fraction[:n] = burned_fraction
        self.last_emission_position_m[:n] = self.position_m[:n]
        return emission

    def update(
        self,
        dt_s: float,
        atmosphere: AtmosphereConfig,
        config: ShellConfig,
    ) -> CombustionEmission | None:
        n = self.count
        if n == 0:
            return None

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
        dying = ~alive
        emission = None
        if np.any(dying):
            dying_indices = np.flatnonzero(dying)
            dying_age = np.clip(
                self.age_s[dying_indices] / self.lifetime_s[dying_indices],
                0.0,
                1.0,
            )
            dying_burned = 1.0 - (1.0 - dying_age) ** 3
            emission = self._make_emission(
                dying_indices, dying_burned, config
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
                self.fuel_mass_kg,
                self.emitted_burn_fraction,
                self.last_emission_position_m,
            ):
                array[:alive_count] = array[:n][alive]
            self.count = alive_count
        return emission

    def intensity(self) -> np.ndarray:
        """Return instantaneous radiant output including ignition and decay."""

        n = self.count
        normalized_age = self.age_s[:n] / self.lifetime_s[:n]
        stochastic_burn = 0.97 + 0.03 * np.sin(
            self.age_s[:n] * 53.0 + np.arange(n, dtype=np.float32) * 1.618
        )
        return (
            self.luminous_power_w[:n]
            * burn_profile(normalized_age)
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
        self._combustion_emissions: list[CombustionEmission] = []

    def consume_burst_events(self) -> list[BurstEvent]:
        events, self._burst_events = self._burst_events, []
        return events

    def consume_combustion_emissions(self) -> list[CombustionEmission]:
        live_emission = self.stars.consume_emission(self.shell_config)
        if live_emission is not None:
            self._combustion_emissions.append(live_emission)
        emissions, self._combustion_emissions = (
            self._combustion_emissions,
            [],
        )
        return emissions

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
                        chemical_energy_j,
                        config.burst_charge_mass_kg
                        * config.smoke_yield_fraction,
                        chemical_energy_j * config.post_blast_thermal_fraction,
                    )
                )
            else:
                surviving_shells.append(shell)

        self.shells = surviving_shells
        emission = self.stars.update(dt_s, self.atmosphere, config)
        if emission is not None:
            self._combustion_emissions.append(emission)
