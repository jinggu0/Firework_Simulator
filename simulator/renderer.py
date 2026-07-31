from __future__ import annotations

import math
from pathlib import Path

import moderngl
import numpy as np

from .camera import FreeCamera
from .astronomy import CelestialState
from .camera_optics import vertical_fov_deg
from .config import (
    AtmosphereConfig,
    LightingConfig,
    PhysicalCameraConfig,
    RenderConfig,
    SmokeConfig,
)
from .fluid import SmokeFluid2D
from .lighting import (
    cluster_radiant_lights,
    radiometric_irradiance_from_illuminance,
)
from .passes import (
    LandPass,
    ParticlePass,
    PostProcessPass,
    RenderTargets,
    ScenePass,
    SkyPass,
    SmokePass,
    WaterPass,
    initialise_static_lights,
)
from .physics import FireworkWorld

SCENE_ASSET = (
    Path(__file__).resolve().parent.parent / "assets" / "yeouido_scene.npz"
)

WATER_MASK_UNIT = 1
TERRAIN_UNIT = 2

# Twilight illuminance spans four orders of magnitude across the show, so the
# shader receives a log-normalised strength rather than raw lux.
TWILIGHT_FLOOR_LUX = 0.0002
TWILIGHT_CEILING_LUX = 3.4


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    return np.array(
        [[f / aspect, 0, 0, 0], [0, f, 0, 0],
         [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
         [0, 0, -1, 0]],
        dtype=np.float32,
    )


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
    side /= np.linalg.norm(side)
    up = np.cross(side, forward)
    result = np.eye(4, dtype=np.float32)
    result[0, :3], result[1, :3], result[2, :3] = side, up, -forward
    result[:3, 3] = -result[:3, :3] @ eye
    return result


class Renderer:
    """Coordinates the render passes over a shared linear-HDR target.

    Pass ordering, the planar-reflection pre-pass, and the uniforms several
    passes share are cross-pass concerns and live here. Everything a single
    pass owns lives in :mod:`simulator.passes`.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        config: RenderConfig,
        atmosphere: AtmosphereConfig | None = None,
        smoke_config: SmokeConfig | None = None,
        lighting_config: LightingConfig | None = None,
        camera_config: PhysicalCameraConfig | None = None,
    ) -> None:
        self.ctx, self.config, self.time_s = ctx, config, 0.0
        self.lighting_config = lighting_config or LightingConfig()
        self.camera_config = camera_config or PhysicalCameraConfig()
        atmosphere = atmosphere or AtmosphereConfig()
        smoke_config = smoke_config or SmokeConfig()

        physical_fov_deg = vertical_fov_deg(self.camera_config)
        tan_half_fov = math.tan(math.radians(physical_fov_deg) * 0.5)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.quad_buffer = ctx.buffer(quad.tobytes())

        self.sky = SkyPass(
            ctx, config, self.quad_buffer, atmosphere, tan_half_fov
        )
        self.post = PostProcessPass(
            ctx, config, self.camera_config, self.quad_buffer, tan_half_fov
        )
        self.scene = ScenePass(
            ctx, self.lighting_config, SCENE_ASSET, config, self.camera_config
        )
        scene_data = self.scene.data
        self.water = WaterPass(
            ctx,
            self.lighting_config,
            atmosphere,
            scene_data.water_mask,
            scene_data.water_mask_bounds,
        )
        self.land = LandPass(
            ctx,
            self.lighting_config,
            self.water.config,
            scene_data.water_mask_bounds,
            scene_data.terrain_bounds,
        )
        initialise_static_lights(
            (self.scene.program, self.land.program),
            self.land.static_light_color,
            self.land.static_light_power_w,
        )
        self.scene.set_terrain(TERRAIN_UNIT)
        self.particles = ParticlePass(ctx, config, self.camera_config)
        self.smoke = SmokePass(ctx, smoke_config)
        self.targets = RenderTargets(ctx, config)

        # The river mask and terrain height are sampled by the water, land, and
        # scene passes, so they are bound once by the coordinator rather than
        # duplicated per pass.
        self.water_mask_texture = ctx.texture(
            (
                scene_data.water_mask.shape[1],
                scene_data.water_mask.shape[0],
            ),
            components=1,
            data=np.ascontiguousarray(scene_data.water_mask).tobytes(),
            dtype="f1",
        )
        self.water_mask_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.water_mask_texture.repeat_x = False
        self.water_mask_texture.repeat_y = False
        self.terrain_texture = ctx.texture(
            (
                scene_data.terrain_height_m.shape[1],
                scene_data.terrain_height_m.shape[0],
            ),
            components=1,
            data=np.ascontiguousarray(scene_data.terrain_height_m).tobytes(),
            dtype="f4",
        )
        self.terrain_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.terrain_texture.repeat_x = False
        self.terrain_texture.repeat_y = False

        self.reflection_interval_s = 1.0 / max(config.reflection_hz, 1)
        self.reflection_accumulator_s = self.reflection_interval_s
        self.reflection_sky_accumulator_s = 1.0
        self.reflection_ready = False
        self.reflection_camera_position = np.full(3, np.inf, dtype=np.float32)
        self.reflection_camera_forward = np.zeros(3, dtype=np.float32)
        self.last_rendered_smoke_revision = -1
        self.scene_illuminance_lux = 0.0
        self.projection = _perspective(
            physical_fov_deg, config.width / config.height, .1, 2500
        )

    # -- compatibility accessors -------------------------------------------
    # Named attributes several tools and the application title bar already read.

    @property
    def significant_wave_height_m(self) -> float:
        return self.water.significant_wave_height_m

    @property
    def star_catalogue_is_measured(self) -> bool:
        return self.sky.catalogue_is_measured

    @property
    def atmospheric_optics(self):
        return self.sky.atmospheric_optics

    @property
    def hdr_texture(self) -> moderngl.Texture:
        return self.targets.hdr_texture

    @property
    def frame_index(self) -> int:
        return self.post.frame_index

    @property
    def display_mode(self):
        return self.post.mode

    def toggle_display_mode(self):
        """Switch between the camera and the observer display transform."""

        return self.post.toggle_mode()

    # -- per-frame updates -------------------------------------------------

    def _update_camera(self, camera: FreeCamera) -> None:
        view = _look_at(camera.position_m, camera.position_m + camera.forward)
        view_projection = self.projection @ view
        matrix_bytes = view_projection.T.astype(np.float32).tobytes()
        inverse_bytes = (
            np.linalg.inv(view_projection).T.astype(np.float32).tobytes()
        )
        self.particles.set_view_projection(matrix_bytes)
        self.water.set_view_projection(matrix_bytes, camera.position_m)
        self.land.set_view_projection(matrix_bytes)
        self.scene.set_view_projection(matrix_bytes, camera.position_m)
        self.smoke.set_view_projection(
            matrix_bytes, inverse_bytes, camera.position_m
        )
        self.smoke.program["camera_inside"] = int(
            np.all(camera.position_m >= self.smoke.bounds[0])
            and np.all(camera.position_m <= self.smoke.bounds[1])
        )
        self._set_sky_camera(camera.forward, camera.right)

    def _set_sky_camera(self, forward: np.ndarray, right: np.ndarray) -> None:
        self.sky.set_camera(forward, right, np.cross(right, forward))

    def _update_static_lights(self, camera: FreeCamera) -> None:
        maximum = min(self.lighting_config.street_lamp_light_count, 4)
        positions = self.scene.data.static_light_positions
        selected = np.zeros((4, 3), dtype=np.float32)
        count = min(len(positions), maximum)
        if count:
            distances_squared = np.sum(
                (positions[:, [0, 2]] - camera.position_m[[0, 2]]) ** 2, axis=1
            )
            nearest = np.argpartition(distances_squared, count - 1)[:count]
            selected[:count] = positions[nearest]
        self.scene.program["static_light_count"] = count
        self.scene.program["static_light_position"].write(selected.tobytes())
        # The land mesh covers the full 5 km patch; two nearest luminaires
        # bound its fragment cost while detailed objects retain four.
        self.land.program["static_light_count"] = min(count, 2)
        self.land.program["static_light_position"].write(selected.tobytes())

    def _update_environment_animation(
        self, atmosphere: AtmosphereConfig
    ) -> None:
        wind = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float32)
        wind_xz = wind[[0, 2]]
        wind_speed = float(np.linalg.norm(wind_xz))
        self.sky.set_time(self.time_s)
        self.sky.program["wind_xz"].value = tuple(wind_xz)
        self.scene.set_environment(self.time_s, wind_xz, wind_speed)
        self.water.set_wind_speed(wind_speed)

    def _render_reflection(self, camera: FreeCamera) -> None:
        reflected_position = camera.position_m.copy()
        reflected_position[1] *= -1.0
        reflected_forward = camera.forward.copy()
        reflected_forward[1] *= -1.0
        reflection_view_projection = self.projection @ _look_at(
            reflected_position, reflected_position + reflected_forward
        )
        matrix_bytes = (
            reflection_view_projection.T.astype(np.float32).tobytes()
        )
        self.land.set_view_projection(matrix_bytes)
        self.scene.set_view_projection(matrix_bytes, reflected_position)
        self.water.set_reflection_view_projection(matrix_bytes)
        self._set_sky_camera(reflected_forward, camera.right)

        self.targets.reflection_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.targets.reflection_fbo.clear(0, 0, 0, 1, depth=1)
        self.sky.draw()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.water_mask_texture.use(WATER_MASK_UNIT)
        self.terrain_texture.use(TERRAIN_UNIT)
        self.land.draw()
        self.scene.draw_reflection()

    def _update_celestial(
        self, celestial: CelestialState, atmosphere: AtmosphereConfig
    ) -> None:
        twilight_floor = math.log10(TWILIGHT_FLOOR_LUX)
        twilight_ceiling = math.log10(TWILIGHT_CEILING_LUX)
        twilight_strength = float(
            np.clip(
                (
                    math.log10(
                        max(
                            celestial.twilight_illuminance_lux,
                            TWILIGHT_FLOOR_LUX,
                        )
                    )
                    - twilight_floor
                )
                / (twilight_ceiling - twilight_floor),
                0.0,
                1.0,
            )
        )
        cloud = atmosphere.cloud_cover_fraction
        # Molecular scattering follows the observed station pressure, so the
        # extinction the stars receive tracks the weather timeline.
        self.sky.set_atmosphere(atmosphere)
        self.sky.set_celestial_frame(celestial.equatorial_from_eus)
        self.sky.update_celestial(celestial, twilight_strength, cloud)
        ambient_scale = 0.8 + twilight_strength * 1.2 + cloud * 0.25
        self.water.set_ambient_scale(ambient_scale)
        self.land.set_ambient_scale(ambient_scale)
        ambient_illuminance_lux = (
            celestial.twilight_illuminance_lux
            + celestial.moon_illuminance_lux
            + self.lighting_config.calibrated_urban_ambient_illuminance_lux
        )
        # Human Vision Mode adapts to the scene's own computed illuminance.
        # Reading the rendered frame back would be more direct but would stall
        # the pipeline every frame.
        self.scene_illuminance_lux = ambient_illuminance_lux
        self.scene.set_ambient_irradiance(
            radiometric_irradiance_from_illuminance(
                ambient_illuminance_lux,
                self.lighting_config.twilight_spectral_luminous_efficacy_lm_w,
            )
        )

    def _update_dynamic_lights(self, world: FireworkWorld) -> np.ndarray:
        count = world.stars.count
        maximum = min(self.lighting_config.dynamic_light_count, 8)
        if count:
            radiant_power_w = world.stars.intensity()
            lights = cluster_radiant_lights(
                world.stars.position_m[:count],
                world.stars.color_linear[:count],
                radiant_power_w,
                maximum,
            )
        else:
            radiant_power_w = np.empty(0, dtype=np.float32)
            lights = cluster_radiant_lights(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                radiant_power_w,
                maximum,
            )
        for program in (self.scene.program, self.water.program):
            program["dynamic_light_count"] = lights.count
            program["dynamic_light_position"].write(
                lights.positions_m.tobytes()
            )
            program["dynamic_light_color"].write(lights.colors.tobytes())
            program["dynamic_light_power_w"].write(
                lights.radiant_power_w.tobytes()
            )
        return radiant_power_w

    def _reflection_is_stale(
        self, camera: FreeCamera, fluid_updated: bool
    ) -> bool:
        camera_changed = (
            float(
                np.linalg.norm(
                    camera.position_m - self.reflection_camera_position
                )
            )
            > 0.025
            or float(np.dot(camera.forward, self.reflection_camera_forward))
            < 0.9999985
        )
        reflection_invalid = camera_changed or (
            self.reflection_sky_accumulator_s >= 1.0
        )
        return not self.reflection_ready or (
            self.reflection_accumulator_s >= self.reflection_interval_s
            and reflection_invalid
            # Scheduled away from a completed fluid step to keep the CPU and
            # GPU spikes from landing on the same frame.
            and not fluid_updated
        )

    # -- frame -------------------------------------------------------------

    def render(
        self,
        world: FireworkWorld,
        camera: FreeCamera,
        celestial: CelestialState,
        frame_dt_s: float,
        smoke: SmokeFluid2D | None = None,
    ) -> None:
        self.time_s += frame_dt_s
        self._update_environment_animation(world.atmosphere)
        self._update_static_lights(camera)
        self._update_celestial(celestial, world.atmosphere)
        radiant_power_w = self._update_dynamic_lights(world)
        self.water.update_forcing(world.atmosphere, frame_dt_s)
        self.reflection_accumulator_s += frame_dt_s
        self.reflection_sky_accumulator_s += frame_dt_s

        smoke_revision = smoke.revision if smoke is not None else -1
        fluid_updated = smoke_revision != self.last_rendered_smoke_revision
        if self._reflection_is_stale(camera, fluid_updated):
            self._render_reflection(camera)
            self.reflection_accumulator_s %= self.reflection_interval_s
            self.reflection_sky_accumulator_s = 0.0
            self.reflection_ready = True
            self.reflection_camera_position[:] = camera.position_m
            self.reflection_camera_forward[:] = camera.forward
        self.last_rendered_smoke_revision = smoke_revision

        self._update_camera(camera)
        self.targets.hdr_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.targets.hdr_fbo.clear(0, 0, 0, 1, depth=1)
        self.sky.draw()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.water_mask_texture.use(WATER_MASK_UNIT)
        self.terrain_texture.use(TERRAIN_UNIT)
        self.land.draw()
        self.scene.draw()
        self.water.draw(self.time_s, self.targets.reflection_texture)
        self.particles.draw(world, radiant_power_w, self.time_s)
        if smoke is not None and smoke.has_visible_smoke():
            self.targets.smoke_fbo.use()
            self.smoke.draw(
                smoke, camera.position_m, self.targets.scene_depth_texture
            )
        self.post.run(self.targets, frame_dt_s, self.scene_illuminance_lux)
