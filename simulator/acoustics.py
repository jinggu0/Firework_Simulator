from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import AcousticConfig, AtmosphereConfig

DRY_AIR_GAS_CONSTANT_J_KG_K = 287.05
HEAT_CAPACITY_RATIO = 1.4
SEDOV_DIMENSIONLESS_RADIUS = 1.033


def saturation_vapor_pressure_pa(temperature_k: float) -> float:
    """Buck equation over liquid water for the simulator's weather range."""

    temperature_c = temperature_k - 273.15
    return 611.21 * math.exp(
        (18.678 - temperature_c / 234.5)
        * (temperature_c / (257.14 + temperature_c))
    )


def speed_of_sound_mps(atmosphere: AtmosphereConfig) -> float:
    vapor_pressure = (
        atmosphere.relative_humidity
        * saturation_vapor_pressure_pa(atmosphere.temperature_k)
    )
    mixing_ratio = 0.622 * vapor_pressure / max(
        atmosphere.pressure_pa - vapor_pressure, 1.0
    )
    specific_humidity = mixing_ratio / (1.0 + mixing_ratio)
    moist_air_gas_constant = DRY_AIR_GAS_CONSTANT_J_KG_K * (
        1.0 + 0.608 * specific_humidity
    )
    return math.sqrt(
        HEAT_CAPACITY_RATIO
        * moist_air_gas_constant
        * atmosphere.temperature_k
    )


@dataclass(frozen=True, slots=True)
class ShockState:
    radius_m: float
    velocity_mps: float
    strong_shock: bool


@dataclass(slots=True)
class _PropagatingBlast:
    position_m: np.ndarray
    chemical_energy_j: float
    age_s: float
    seed: int
    preparation_requested: bool = False


@dataclass(frozen=True, slots=True)
class SoundArrival:
    propagation_delay_s: float
    distance_m: float
    effective_sound_speed_mps: float
    peak_pressure_pa: float
    rms_pressure_pa: float
    sound_pressure_level_db: float
    direction_to_source: np.ndarray
    stereo_pan: float
    seed: int


class FireworkAcoustics:
    """Strong-shock transition, delayed propagation, and pressure synthesis."""

    def __init__(self, config: AcousticConfig, seed: int) -> None:
        self.config = config
        self._seed = seed
        self._event_index = 0
        self._blasts: list[_PropagatingBlast] = []
        self._tail_template = self._build_tail_template(seed)
        sample_count = len(self._tail_template)
        self._sample_time_s = (
            np.arange(sample_count, dtype=np.float32)
            / self.config.sample_rate_hz
        )
        self._tail_template *= np.exp(
            -self._sample_time_s
            / (self.config.reverberation_time_s * 0.34)
        )

    def _build_tail_template(self, seed: int) -> np.ndarray:
        sample_count = int(round(
            self.config.sample_rate_hz
            * self.config.reverberation_time_s
        ))
        rng = np.random.default_rng(seed)
        white = rng.normal(0.0, 1.0, sample_count).astype(np.float32)
        spectrum = np.fft.rfft(white)
        frequencies = np.fft.rfftfreq(
            sample_count, 1.0 / self.config.sample_rate_hz
        )
        low_pass = 1.0 / np.sqrt(1.0 + (frequencies / 180.0) ** 8)
        high_pass = 1.0 - 1.0 / np.sqrt(
            1.0 + (frequencies / 32.0) ** 8
        )
        tail = np.fft.irfft(
            spectrum * low_pass * high_pass, n=sample_count
        ).astype(np.float32)
        tail /= max(float(np.std(tail)), 1e-7)
        return tail

    @staticmethod
    def shock_state(
        elapsed_s: float,
        chemical_energy_j: float,
        atmosphere: AtmosphereConfig,
    ) -> ShockState:
        if elapsed_s <= 0.0 or chemical_energy_j <= 0.0:
            return ShockState(0.0, math.inf, True)
        density = atmosphere.air_density_kg_m3
        radius = SEDOV_DIMENSIONLESS_RADIUS * (
            chemical_energy_j * elapsed_s * elapsed_s / density
        ) ** 0.2
        velocity = 0.4 * radius / elapsed_s
        return ShockState(
            radius,
            velocity,
            velocity > speed_of_sound_mps(atmosphere),
        )

    @staticmethod
    def strong_shock_transition(
        chemical_energy_j: float,
        atmosphere: AtmosphereConfig,
    ) -> tuple[float, float]:
        sound_speed = speed_of_sound_mps(atmosphere)
        energy_scale = (
            chemical_energy_j / atmosphere.air_density_kg_m3
        ) ** 0.2
        transition_time = (
            0.4
            * SEDOV_DIMENSIONLESS_RADIUS
            * energy_scale
            / sound_speed
        ) ** (5.0 / 3.0)
        transition_radius = FireworkAcoustics.shock_state(
            transition_time, chemical_energy_j, atmosphere
        ).radius_m
        return transition_time, transition_radius

    def emit(self, position_m: np.ndarray, chemical_energy_j: float) -> None:
        self._blasts.append(
            _PropagatingBlast(
                np.asarray(position_m, dtype=np.float32).copy(),
                chemical_energy_j,
                0.0,
                self._seed + self._event_index * 7919,
            )
        )
        self._event_index += 1

    def predict_arrival(
        self,
        source_position_m: np.ndarray,
        chemical_energy_j: float,
        listener_position_m: np.ndarray,
        listener_right: np.ndarray,
        atmosphere: AtmosphereConfig,
    ) -> SoundArrival:
        blast = _PropagatingBlast(
            np.asarray(source_position_m, dtype=np.float32),
            chemical_energy_j,
            0.0,
            self._seed,
        )
        return self._arrival_for(
            blast, listener_position_m, listener_right, atmosphere
        )[1]

    def _arrival_for(
        self,
        blast: _PropagatingBlast,
        listener_position_m: np.ndarray,
        listener_right: np.ndarray,
        atmosphere: AtmosphereConfig,
    ) -> tuple[float, SoundArrival]:
        source_to_listener = (
            np.asarray(listener_position_m, dtype=np.float32)
            - blast.position_m
        )
        distance = max(float(np.linalg.norm(source_to_listener)), 0.5)
        propagation_direction = source_to_listener / distance
        average_height = max(
            0.5 * (
                float(listener_position_m[1]) + float(blast.position_m[1])
            ),
            10.0,
        )
        wind = np.asarray(
            atmosphere.wind_at_height_m(average_height), dtype=np.float32
        )
        effective_speed = max(
            speed_of_sound_mps(atmosphere)
            + float(np.dot(wind, propagation_direction)),
            250.0,
        )
        shock_time, shock_radius = self.strong_shock_transition(
            blast.chemical_energy_j
            * self.config.blast_wave_energy_fraction,
            atmosphere,
        )
        delay = shock_time + max(distance - shock_radius, 0.0) / effective_speed
        duration = (
            self.config.impulse_duration_s
            + self.config.range_duration_s_m * distance
        )
        acoustic_energy = (
            blast.chemical_energy_j
            * self.config.acoustic_energy_fraction
        )
        intensity_w_m2 = acoustic_energy / (
            4.0 * math.pi * distance * distance * duration
        )
        absorption_gain = 10.0 ** (
            -self.config.atmospheric_absorption_db_m * distance / 20.0
        )
        rms_pressure = math.sqrt(
            intensity_w_m2
            * atmosphere.air_density_kg_m3
            * effective_speed
        ) * absorption_gain
        peak_pressure = math.sqrt(2.0) * rms_pressure
        spl = 20.0 * math.log10(
            max(rms_pressure, 1e-12) / self.config.reference_pressure_pa
        )
        direction_to_source = -propagation_direction
        pan = float(
            np.clip(np.dot(direction_to_source, listener_right), -1.0, 1.0)
        )
        return delay, SoundArrival(
            delay,
            distance,
            effective_speed,
            peak_pressure,
            rms_pressure,
            spl,
            direction_to_source.copy(),
            pan,
            blast.seed,
        )

    def update(
        self,
        dt_s: float,
        listener_position_m: np.ndarray,
        listener_right: np.ndarray,
        atmosphere: AtmosphereConfig,
    ) -> list[SoundArrival]:
        arrivals: list[SoundArrival] = []
        propagating: list[_PropagatingBlast] = []
        for blast in self._blasts:
            blast.age_s += dt_s
            delay, arrival = self._arrival_for(
                blast, listener_position_m, listener_right, atmosphere
            )
            if blast.age_s >= delay:
                arrivals.append(arrival)
            else:
                propagating.append(blast)
        self._blasts = propagating
        return arrivals

    def prepare_upcoming(
        self,
        horizon_s: float,
        listener_position_m: np.ndarray,
        listener_right: np.ndarray,
        atmosphere: AtmosphereConfig,
    ) -> list[SoundArrival]:
        upcoming: list[SoundArrival] = []
        for blast in self._blasts:
            if blast.preparation_requested:
                continue
            delay, arrival = self._arrival_for(
                blast, listener_position_m, listener_right, atmosphere
            )
            if delay - blast.age_s <= horizon_s:
                blast.preparation_requested = True
                upcoming.append(arrival)
        return upcoming

    def synthesize_pcm(self, arrival: SoundArrival) -> np.ndarray:
        sample_rate = self.config.sample_rate_hz
        duration_s = self.config.reverberation_time_s
        sample_count = int(round(sample_rate * duration_s))
        positive_duration = min(
            0.008 + arrival.distance_m * 0.00004, 0.045
        )
        shock_count = min(
            int(math.ceil(positive_duration * 2.5 * sample_rate)),
            sample_count,
        )
        shock_time = self._sample_time_s[:shock_count]
        normalized_time = shock_time / positive_duration
        pressure = (
            self._tail_template
            * arrival.rms_pressure_pa
            * (0.22 if arrival.seed % 2 == 0 else -0.22)
        )
        pressure = pressure.copy()
        pressure[:shock_count] += (
            arrival.peak_pressure_pa
            * (1.0 - normalized_time)
            * np.exp(-self.config.shock_decay * normalized_time)
        )

        pan = arrival.stereo_pan
        left_gain = math.sqrt(0.5 * (1.0 - pan))
        right_gain = math.sqrt(0.5 * (1.0 + pan))
        normalized = np.clip(
            pressure / self.config.full_scale_pressure_pa, -1.0, 1.0
        )
        stereo = np.column_stack(
            (normalized * left_gain, normalized * right_gain)
        )
        return np.ascontiguousarray(stereo * 32767.0, dtype=np.int16)
