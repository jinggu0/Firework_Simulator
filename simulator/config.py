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

    def air_density_at_height_kg_m3(self, height_m: float) -> float:
        """Moist-air density aloft, from the barometric and lapse relations.

        A shell bursting near 160 m sits in air about 1.5% thinner than the
        surface value, and one reaching 300 m about 2.8% thinner. Using the
        surface density for the whole trajectory overstates drag throughout the
        climb.

        Imported locally because :mod:`simulator.atmosphere` depends on this
        module's dataclasses.
        """

        from .atmosphere import (
            moist_air_density_at_height_kg_m3,
        )

        return moist_air_density_at_height_kg_m3(
            self.temperature_k,
            self.pressure_pa,
            self.relative_humidity,
            height_m,
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
    color_temperature_k: float = 2_300.0
    star_composition_mass_kg: float = 0.78
    star_radiative_energy_fraction: float = 0.15
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

    prefer_gpu_solver: bool = True
    prefer_3d_gpu_solver: bool = True
    gpu_update_hz: int = 60
    gpu_3d_update_hz: int = 30
    gpu_3d_grid_size: tuple[int, int, int] = (32, 24, 10)
    gpu_3d_bounds_m: tuple[float, float, float, float, float, float] = (
        -200.0, 200.0, 0.0, 300.0, -60.0, 60.0
    )
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
    volume_depth_bias_m: float = 0.20
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
class LightingConfig:
    """Calibratable source-energy and light-transport parameters."""

    led_input_power_w: float = 35.0
    led_driver_efficiency: float = 0.92
    led_internal_quantum_efficiency: float = 0.82
    led_electrical_efficiency: float = 0.90
    led_extraction_efficiency: float = 0.86
    led_phosphor_efficiency: float = 0.83
    luminaire_optical_efficiency: float = 0.88
    led_spectral_luminous_efficacy_lm_w: float = 300.0
    interior_utilization_factor: float = 0.42
    window_visible_transmittance: float = 0.70
    window_emitting_area_m2: float = 8.0
    twilight_spectral_luminous_efficacy_lm_w: float = 100.0
    calibrated_urban_ambient_illuminance_lux: float = 1.2
    # air_extinction_per_m was 0.00012 here with no recorded source. It implied
    # 32.6 km of visibility, roughly twice what the event's own weather record
    # supports once the aerosol is grown by the observed humidity, and it was
    # applied only between a lamp and a surface — never along the view path.
    # Air extinction is now derived per channel from the same optical depths
    # that dim the stars; see simulator.atmosphere.SurfaceExtinction.
    dynamic_light_count: int = 8
    street_lamp_input_power_w: float = 72.0
    street_lamp_light_count: int = 4


@dataclass(frozen=True, slots=True)
class PhysicalCameraConfig:
    """Thin-lens and CMOS sensor parameters."""

    sensor_width_mm: float = 36.0
    sensor_height_mm: float = 20.25
    focal_length_mm: float = 24.0
    f_number: float = 2.8
    lens_transmission: float = 0.90
    shutter_time_s: float = 1.0 / 60.0
    iso: float = 800.0
    base_iso: float = 100.0
    pixel_pitch_um: float = 5.9
    quantum_efficiency_rgb: tuple[float, float, float] = (0.42, 0.52, 0.36)
    wavelength_rgb_nm: tuple[float, float, float] = (610.0, 550.0, 460.0)
    full_well_electrons: float = 45_000.0
    read_noise_electrons: float = 2.2
    enable_sensor_noise: bool = True

    white_balance_temperature_k: float = 6504.0
    """Planckian reference the balance renders neutral.

    An operator setting, not a measurement. 6504 K is the only value derivable
    from the pipeline's own colour space — the renderer's radiance carries sRGB
    primaries, so balancing there makes a neutral scene render neutral and the
    stage a correction rather than a look. See
    :func:`simulator.camera_optics.white_balance_gains`.
    """

    # Brown-Conrady lens distortion, in OpenCV's k1 k2 p1 p2 k3 convention on
    # normalised image coordinates. **Identity by default**: no calibration of
    # the lens exists, and invented coefficients would put a fabricated optical
    # claim into every frame. `camera_optics.load_lens_calibration` reads a
    # measured set when one is obtained.
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    distortion_k3: float = 0.0
    distortion_p1: float = 0.0
    distortion_p2: float = 0.0


@dataclass(frozen=True, slots=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    target_fps: int = 60
    vsync: bool = True
    physics_hz: int = 120
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
    lighting: LightingConfig = field(default_factory=LightingConfig)
    physical_camera: PhysicalCameraConfig = field(
        default_factory=PhysicalCameraConfig
    )
    random_seed: int = 20241005
