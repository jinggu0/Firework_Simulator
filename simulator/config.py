from __future__ import annotations

from dataclasses import dataclass, field
import math

from .camera import CameraConfig

@dataclass(frozen=True, slots=True)
class AtmosphereConfig:
    """Initial atmospheric state in SI units."""

    temperature_k: float = 286.15
    pressure_pa: float = 101_325.0
    relative_humidity: float = 0.60
    wind_velocity_mps: tuple[float, float, float] = (1.4, 0.0, 0.2)
    wind_velocity_100m_mps: tuple[float, float, float] = (1.8, 0.0, 0.4)
    air_density_kg_m3: float = 1.225
    cloud_cover_fraction: float = 0.0

    def wind_at_height_m(self, height_m: float) -> tuple[float, float, float]:
        alpha = min(max(math.log(max(height_m, 10.0) / 10.0) / math.log(10.0), 0.0), 1.0)
        return tuple(
            low * (1.0 - alpha) + high * alpha
            for low, high in zip(
                self.wind_velocity_mps, self.wind_velocity_100m_mps
            )
        )


@dataclass(frozen=True, slots=True)
class ShellConfig:
    """A calibratable aerial shell, expressed in SI units."""

    dry_mass_kg: float = 1.15
    diameter_m: float = 0.10
    launch_speed_mps: float = 78.0
    fuse_delay_s: float = 3.05
    drag_coefficient: float = 0.47
    burst_star_count: int = 8_000
    star_speed_mean_mps: float = 30.0
    star_speed_std_mps: float = 4.5
    star_lifetime_mean_s: float = 2.25
    star_lifetime_std_s: float = 0.22
    star_drag_time_s: float = 1.35
    luminous_power_w: float = 155.0
    color_temperature_k: float = 2_300.0
    star_composition_mass_kg: float = 0.78
    star_smoke_yield_fraction: float = 0.24
    star_specific_energy_j_kg: float = 4_200_000.0
    star_post_combustion_thermal_fraction: float = 0.06
    burst_charge_mass_kg: float = 0.085
    burst_specific_energy_j_kg: float = 3_000_000.0
    smoke_yield_fraction: float = 0.12
    post_blast_thermal_fraction: float = 0.18


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    """Low-Mach post-blast plume solver settings in SI units."""

    grid_size: tuple[int, int] = (64, 36)
    bounds_m: tuple[float, float, float, float] = (-320.0, 320.0, 0.0, 360.0)
    update_hz: int = 30
    pressure_iterations: int = 24
    smoke_diffusivity_m2_s: float = 0.025
    thermal_diffusivity_m2_s: float = 0.035
    kinematic_viscosity_m2_s: float = 1.5e-5
    smoke_half_life_s: float = 38.0
    thermal_half_life_s: float = 7.5
    vorticity_confinement: float = 0.18
    source_radius_m: float = 7.5
    plume_depth_m: float = 24.0
    volume_depth_m: float = 120.0
    volume_slices: int = 24
    volume_ray_steps: int = 40
    max_temperature_excess_k: float = 850.0


@dataclass(frozen=True, slots=True)
class AcousticConfig:
    """Far-field blast acoustics and calibrated playback settings."""

    sample_rate_hz: int = 48_000
    blast_wave_energy_fraction: float = 0.32
    acoustic_energy_fraction: float = 0.01
    atmospheric_absorption_db_m: float = 0.00012
    impulse_duration_s: float = 0.16
    range_duration_s_m: float = 0.00035
    shock_decay: float = 3.2
    reverberation_time_s: float = 1.8
    full_scale_pressure_pa: float = 20.0
    reference_pressure_pa: float = 20.0e-6


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    target_fps: int = 60
    vsync: bool = True
    physics_hz: int = 120
    vertical_fov_deg: float = 52.0
    exposure_ev100: float = 7.0
    shutter_time_s: float = 1.0 / 60.0
    bloom_strength: float = 0.72
    reflection_scale: float = 0.5
    reflection_hz: int = 30
    max_particles: int = 250_000


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)
    acoustics: AcousticConfig = field(default_factory=AcousticConfig)
    random_seed: int = 20241005
