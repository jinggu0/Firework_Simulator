from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .astronomy import horizontal_direction_eus
from .config import AtmosphereConfig, ShellConfig
from .lighting import burn_profile, combustion_peak_radiant_power_w
from .shells import (
    BreakPattern,
    FlickerMode,
    SecondaryBreak,
    ShellProfile,
    emission_directions,
)

GRAVITY_MPS2 = np.array([0.0, -9.80665, 0.0], dtype=np.float32)

_FLICKER_STEADY = 0
_FLICKER_STROBE = 1
_FLICKER_CRACKLE = 2
_FLICKER_CODE = {
    FlickerMode.STEADY: _FLICKER_STEADY,
    FlickerMode.STROBE: _FLICKER_STROBE,
    FlickerMode.CRACKLE: _FLICKER_CRACKLE,
}

STROBE_DUTY_CYCLE = 0.32
"""Fraction of each strobe period the star is emitting."""

CRACKLE_MODULATION_DEPTH = 0.5
"""Peak relative swing of the crackle modulation about its unit mean."""

COLOR_TRANSITION_FRACTION = 0.08
"""Normalised burn fraction over which a colour-changing star crosses over."""


def tube_direction_eus(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit launch direction for a mortar tube.

    Reuses the project's single azimuth/altitude convention helper rather than
    re-deriving the signs: azimuth is measured north-clockwise and elevation up
    from the horizon, in the East-Up-South frame. An elevation of 90 degrees is
    vertical, which reproduces the previous fixed launch direction.
    """

    return horizontal_direction_eus(azimuth_deg, elevation_deg)


@dataclass(slots=True)
class Shell:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    profile: ShellProfile
    age_s: float = 0.0
    alive: bool = True
    event_id: str = ""


@dataclass(frozen=True, slots=True)
class BurstEvent:
    position_m: np.ndarray
    chemical_energy_j: float
    smoke_mass_kg: float
    post_blast_thermal_energy_j: float
    profile_id: str = ""
    event_id: str = ""


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
        # Optical modulation. Kept as parallel arrays so the hot integration
        # path stays a single vectorised pass.
        self.color_secondary_linear = np.empty((capacity, 3), dtype=np.float32)
        self.color_change_age = np.empty(capacity, dtype=np.float32)
        self.flicker_mode = np.empty(capacity, dtype=np.int8)
        self.flicker_hz = np.empty(capacity, dtype=np.float32)
        self.phase_offset = np.empty(capacity, dtype=np.float32)
        # Combustion coefficients are per star, not per world: a show mixing
        # shell types would otherwise attribute every star's smoke and heat to
        # whichever profile the world was constructed with.
        self.smoke_yield_fraction = np.empty(capacity, dtype=np.float32)
        self.thermal_energy_j_per_kg = np.empty(capacity, dtype=np.float32)
        # Index into ``secondary_specs``; -1 means the star carries no break.
        self.secondary_index = np.empty(capacity, dtype=np.int16)
        self.secondary_specs: list[_SecondaryRecord] = []
        self._has_flicker = False
        self._has_color_change = False
        self._has_secondary = False

    def clear(self) -> None:
        self.count = 0
        self.secondary_specs.clear()
        self._has_flicker = False
        self._has_color_change = False
        self._has_secondary = False

    # -- spawning ----------------------------------------------------------

    def _write_optics(
        self,
        start: int,
        end: int,
        profile: ShellProfile,
        rng: np.random.Generator,
        secondary_index: int,
    ) -> None:
        n = end - start
        self.color_linear[start:end] = profile.primary_color_linear()
        secondary_color = profile.secondary_color_linear()
        if secondary_color is None:
            self.color_secondary_linear[start:end] = profile.primary_color_linear()
            self.color_change_age[start:end] = 2.0  # never reached
        else:
            self.color_secondary_linear[start:end] = secondary_color
            self.color_change_age[start:end] = profile.color_change_fraction
            self._has_color_change = True
        self.flicker_mode[start:end] = _FLICKER_CODE[profile.flicker]
        self.flicker_hz[start:end] = profile.flicker_hz
        if profile.flicker is not FlickerMode.STEADY:
            self._has_flicker = True
        # A per-star phase keeps a flickering break from pulsing in unison,
        # which would read as one blinking object rather than many stars.
        self.phase_offset[start:end] = rng.random(n).astype(np.float32)
        self.secondary_index[start:end] = secondary_index
        if secondary_index >= 0:
            self._has_secondary = True

    def _write_energy(
        self,
        start: int,
        end: int,
        composition_mass_kg: float,
        specific_energy_j_kg: float,
        radiative_fraction: float,
        smoke_yield_fraction: float,
        thermal_fraction: float,
        star_count: int,
    ) -> None:
        self.fuel_mass_kg[start:end] = composition_mass_kg / star_count
        chemical_energy_j = (
            self.fuel_mass_kg[start:end] * specific_energy_j_kg
        )
        self.luminous_power_w[start:end] = combustion_peak_radiant_power_w(
            chemical_energy_j,
            self.lifetime_s[start:end],
            radiative_fraction,
        )
        self.smoke_yield_fraction[start:end] = smoke_yield_fraction
        self.thermal_energy_j_per_kg[start:end] = (
            specific_energy_j_kg * thermal_fraction
        )

    def spawn_burst(
        self,
        origin_m: np.ndarray,
        shell_velocity_mps: np.ndarray,
        config: ShellProfile | ShellConfig,
        rng: np.random.Generator,
    ) -> int:
        """Emit a shell's stars. Returns the number actually spawned."""

        profile = ShellProfile.coerce(config)
        n = min(profile.burst_star_count, self.capacity - self.count)
        if n <= 0:
            return 0

        start, end = self.count, self.count + n
        directions, speed_scale = emission_directions(
            profile.pattern,
            n,
            rng,
            profile.pattern_spread_deg,
        )
        speeds = np.maximum(
            rng.normal(profile.star_speed_mean_mps, profile.star_speed_std_mps, n),
            0.1,
        ).astype(np.float32) * speed_scale

        self.position_m[start:end] = origin_m
        self.previous_position_m[start:end] = origin_m
        self.velocity_mps[start:end] = (
            directions * speeds[:, None]
            + np.asarray(shell_velocity_mps, dtype=np.float32)[None, :]
        )
        self.age_s[start:end] = 0.0
        self.lifetime_s[start:end] = np.maximum(
            rng.normal(
                profile.star_lifetime_mean_s, profile.star_lifetime_std_s, n
            ),
            0.05,
        )
        self.drag_time_s[start:end] = profile.star_drag_time_s

        secondary_index = -1
        carrier_mask: np.ndarray | None = None
        if profile.secondary is not None:
            if profile.secondary.parent_fraction < 1.0:
                # Only a fraction of the primaries carry a break; the rest
                # simply burn out.
                carrier_mask = rng.random(n) < profile.secondary.parent_fraction
                carrier_count = int(carrier_mask.sum())
            else:
                carrier_count = n
            secondary_index = len(self.secondary_specs)
            self.secondary_specs.append(
                _SecondaryRecord(
                    spec=profile.secondary,
                    specific_energy_j_kg=profile.star_specific_energy_j_kg,
                    radiative_fraction=profile.star_radiative_energy_fraction,
                    smoke_yield_fraction=profile.star_smoke_yield_fraction,
                    thermal_fraction=(
                        profile.star_post_combustion_thermal_fraction
                    ),
                    # Carriers die over a spread of steps, so the break is
                    # released in batches. Recording how many parents share the
                    # declared composition mass is what keeps each batch from
                    # claiming the whole budget.
                    carrier_count=carrier_count,
                )
            )
        self._write_optics(start, end, profile, rng, secondary_index)
        if carrier_mask is not None:
            self.secondary_index[start:end] = np.where(
                carrier_mask, secondary_index, -1
            ).astype(np.int16)

        self._write_energy(
            start,
            end,
            profile.star_composition_mass_kg,
            profile.star_specific_energy_j_kg,
            profile.star_radiative_energy_fraction,
            profile.star_smoke_yield_fraction,
            profile.star_post_combustion_thermal_fraction,
            profile.burst_star_count,
        )
        self.emitted_burn_fraction[start:end] = 0.0
        self.last_emission_position_m[start:end] = origin_m
        self.count = end
        return n

    def _spawn_secondary(
        self,
        positions_m: np.ndarray,
        velocities_mps: np.ndarray,
        record: "_SecondaryRecord",
        rng: np.random.Generator,
    ) -> int:
        """Release the children of a set of primary stars that just expired."""

        spec = record.spec
        parents = len(positions_m)
        total = min(parents * spec.star_count, self.capacity - self.count)
        if total <= 0:
            return 0
        parents_used = total // spec.star_count
        if parents_used <= 0:
            return 0
        total = parents_used * spec.star_count

        start, end = self.count, self.count + total
        directions, speed_scale = emission_directions(
            spec.pattern, total, rng, 8.0
        )
        speeds = np.maximum(
            rng.normal(spec.speed_mean_mps, spec.speed_std_mps, total), 0.1
        ).astype(np.float32) * speed_scale
        origins = np.repeat(
            positions_m[:parents_used], spec.star_count, axis=0
        ).astype(np.float32)
        inherited = np.repeat(
            velocities_mps[:parents_used], spec.star_count, axis=0
        ).astype(np.float32)

        self.position_m[start:end] = origins
        self.previous_position_m[start:end] = origins
        self.velocity_mps[start:end] = directions * speeds[:, None] + inherited
        self.age_s[start:end] = 0.0
        self.lifetime_s[start:end] = np.maximum(
            rng.normal(spec.lifetime_mean_s, spec.lifetime_std_s, total), 0.05
        )
        self.drag_time_s[start:end] = spec.drag_time_s

        color = (
            np.asarray(
                _secondary_color(spec), dtype=np.float32
            )
        )
        self.color_linear[start:end] = color
        self.color_secondary_linear[start:end] = color
        self.color_change_age[start:end] = 2.0
        self.flicker_mode[start:end] = _FLICKER_STEADY
        self.flicker_hz[start:end] = 0.0
        self.phase_offset[start:end] = rng.random(total).astype(np.float32)
        # Children do not carry a further break; a third generation would need
        # its own declared mass budget.
        self.secondary_index[start:end] = -1

        # Children draw from the secondary's own declared composition mass, so
        # a break can never inflate the parent shell's energy or smoke budget.
        # The share is taken against every parent that carries the break, not
        # against this batch: carriers expire over a spread of steps, and
        # scaling by the batch size would hand the full budget to each batch.
        # A capacity-limited release therefore under-reports, never over.
        released_fraction = parents_used / max(record.carrier_count, 1)
        self._write_energy(
            start,
            end,
            spec.composition_mass_kg * released_fraction,
            record.specific_energy_j_kg,
            record.radiative_fraction,
            record.smoke_yield_fraction,
            record.thermal_fraction,
            total,
        )
        self.emitted_burn_fraction[start:end] = 0.0
        self.last_emission_position_m[start:end] = origins
        self.count = end
        return total

    # -- combustion emission ----------------------------------------------

    def _make_emission(
        self,
        indices: np.ndarray,
        burned_fraction: np.ndarray,
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
                burned_mass_kg * self.smoke_yield_fraction[selected]
            ).astype(np.float32),
            (
                burned_mass_kg * self.thermal_energy_j_per_kg[selected]
            ).astype(np.float32),
        )

    def consume_emission(self) -> CombustionEmission | None:
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
        emission = self._make_emission(indices, burned_fraction)
        self.emitted_burn_fraction[:n] = burned_fraction
        self.last_emission_position_m[:n] = self.position_m[:n]
        return emission

    # -- integration -------------------------------------------------------

    def update(
        self,
        dt_s: float,
        atmosphere: AtmosphereConfig,
        rng: np.random.Generator | None = None,
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
        pending_breaks: list[tuple[int, np.ndarray, np.ndarray]] = []
        if np.any(dying):
            dying_indices = np.flatnonzero(dying)
            dying_age = np.clip(
                self.age_s[dying_indices] / self.lifetime_s[dying_indices],
                0.0,
                1.0,
            )
            dying_burned = 1.0 - (1.0 - dying_age) ** 3
            emission = self._make_emission(dying_indices, dying_burned)
            if self._has_secondary:
                # Capture break carriers before compaction moves the arrays.
                carriers = self.secondary_index[dying_indices]
                for spec_index in np.unique(carriers[carriers >= 0]):
                    selected = dying_indices[carriers == spec_index]
                    pending_breaks.append(
                        (
                            int(spec_index),
                            self.position_m[selected].copy(),
                            self.velocity_mps[selected].copy(),
                        )
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
                self.color_secondary_linear,
                self.color_change_age,
                self.flicker_mode,
                self.flicker_hz,
                self.phase_offset,
                self.smoke_yield_fraction,
                self.thermal_energy_j_per_kg,
                self.secondary_index,
            ):
                array[:alive_count] = array[:n][alive]
            self.count = alive_count

        if pending_breaks and rng is not None:
            for record_index, positions, velocities in pending_breaks:
                self._spawn_secondary(
                    positions,
                    velocities,
                    self.secondary_specs[record_index],
                    rng,
                )
        return emission

    # -- appearance --------------------------------------------------------

    def _flicker_factor(self) -> np.ndarray:
        """Unit-mean temporal modulation of radiant output.

        Both modes are normalised so their mean over many periods is one, which
        keeps the star's time-integrated radiant energy equal to its chemical
        budget. The residual for a finite burn is bounded by one period out of
        the burn, roughly ``1 / (flicker_hz * lifetime_s)``.
        """

        n = self.count
        mode = self.flicker_mode[:n]
        factor = np.ones(n, dtype=np.float32)
        phase = self.age_s[:n] * self.flicker_hz[:n] + self.phase_offset[:n]
        cycle = phase - np.floor(phase)

        strobe = mode == _FLICKER_STROBE
        if np.any(strobe):
            # Smooth pulse edges rather than a hard square wave, which would
            # alias badly against the 60 Hz shutter.
            edge = np.clip(cycle[strobe] / STROBE_DUTY_CYCLE, 0.0, 1.0)
            pulse = np.sin(np.pi * edge) ** 2
            # sin^2 has mean 1/2 over its support, so the normalisation that
            # preserves unit mean is 2 / duty.
            factor[strobe] = (2.0 / STROBE_DUTY_CYCLE) * pulse

        crackle = mode == _FLICKER_CRACKLE
        if np.any(crackle):
            factor[crackle] = 1.0 + CRACKLE_MODULATION_DEPTH * np.sin(
                2.0 * np.pi * cycle[crackle]
            )
        return factor

    def current_color_linear(self) -> np.ndarray:
        """Per-star linear RGB, accounting for colour-changing stars."""

        n = self.count
        if not self._has_color_change:
            return self.color_linear[:n]
        normalized_age = self.age_s[:n] / self.lifetime_s[:n]
        alpha = np.clip(
            (normalized_age - self.color_change_age[:n])
            / COLOR_TRANSITION_FRACTION,
            0.0,
            1.0,
        )
        # Smoothstep so the crossover reads as a transition, not a cut.
        alpha = (alpha * alpha * (3.0 - 2.0 * alpha))[:, None]
        return (
            self.color_linear[:n] * (1.0 - alpha)
            + self.color_secondary_linear[:n] * alpha
        ).astype(np.float32)

    def intensity(self) -> np.ndarray:
        """Return instantaneous radiant output including ignition and decay."""

        n = self.count
        normalized_age = self.age_s[:n] / self.lifetime_s[:n]
        stochastic_burn = 0.97 + 0.03 * np.sin(
            self.age_s[:n] * 53.0 + np.arange(n, dtype=np.float32) * 1.618
        )
        power = (
            self.luminous_power_w[:n]
            * burn_profile(normalized_age)
            * stochastic_burn
        )
        if self._has_flicker:
            power = power * self._flicker_factor()
        return power


@dataclass(frozen=True, slots=True)
class _SecondaryRecord:
    """A pending secondary break plus the parent's combustion coefficients.

    Carrying the coefficients here keeps the child stars' smoke and heat tied
    to the shell that launched them rather than to whichever profile the world
    happens to hold.
    """

    spec: SecondaryBreak
    specific_energy_j_kg: float
    radiative_fraction: float
    smoke_yield_fraction: float
    thermal_fraction: float
    carrier_count: int = 1
    """Parents sharing this break's declared composition mass."""


def _secondary_color(spec: SecondaryBreak) -> np.ndarray:
    from .color import blackbody_rgb, wavelength_rgb

    if spec.emission_wavelength_nm is None:
        return blackbody_rgb(spec.color_temperature_k)
    return wavelength_rgb(spec.emission_wavelength_nm)


class FireworkWorld:
    def __init__(
        self,
        atmosphere: AtmosphereConfig,
        shell_config: ShellProfile | ShellConfig,
        max_particles: int,
        seed: int,
    ) -> None:
        self.atmosphere = atmosphere
        self.shell_profile = ShellProfile.coerce(shell_config)
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
        live_emission = self.stars.consume_emission()
        if live_emission is not None:
            self._combustion_emissions.append(live_emission)
        emissions, self._combustion_emissions = (
            self._combustion_emissions,
            [],
        )
        return emissions

    def launch(
        self,
        position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        profile: ShellProfile | None = None,
        azimuth_deg: float = 0.0,
        elevation_deg: float = 90.0,
        event_id: str = "",
    ) -> Shell:
        """Fire a shell from a position along a tube heading.

        The default elevation of 90 degrees is vertical, reproducing the
        previous fixed launch direction.
        """

        selected = profile or self.shell_profile
        direction = tube_direction_eus(azimuth_deg, elevation_deg)
        shell = Shell(
            position_m=np.asarray(position_m, dtype=np.float32).copy(),
            velocity_mps=(direction * selected.launch_speed_mps).astype(
                np.float32
            ),
            profile=selected,
            event_id=event_id,
        )
        self.shells.append(shell)
        return shell

    def update(self, dt_s: float) -> None:
        surviving_shells: list[Shell] = []
        for shell in self.shells:
            profile = shell.profile
            cross_section_m2 = np.pi * (profile.diameter_m * 0.5) ** 2
            # Density at the shell's own altitude rather than at the surface.
            # Air is about 1.5% thinner at a 160 m break and 2.8% at 300 m, so
            # the surface value overstates drag through the whole climb.
            drag_factor = (
                0.5
                * self.atmosphere.air_density_at_height_kg_m3(
                    float(shell.position_m[1])
                )
                * profile.drag_coefficient
                * cross_section_m2
                / profile.dry_mass_kg
            )
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
            if shell.age_s >= profile.fuse_delay_s:
                self.stars.spawn_burst(
                    shell.position_m, shell.velocity_mps, profile, self.rng
                )
                chemical_energy_j = profile.burst_chemical_energy_j
                self._burst_events.append(
                    BurstEvent(
                        shell.position_m.copy(),
                        chemical_energy_j,
                        profile.burst_charge_mass_kg
                        * profile.smoke_yield_fraction,
                        chemical_energy_j * profile.post_blast_thermal_fraction,
                        profile.profile_id,
                        shell.event_id,
                    )
                )
            else:
                surviving_shells.append(shell)

        self.shells = surviving_shells
        emission = self.stars.update(dt_s, self.atmosphere, self.rng)
        if emission is not None:
            self._combustion_emissions.append(emission)
