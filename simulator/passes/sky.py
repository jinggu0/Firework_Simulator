"""Sky radiance: twilight, cloud, moon, and the star catalogue."""

from __future__ import annotations

import moderngl
import numpy as np

from .. import shaders
from ..astronomy import CelestialState
from ..atmosphere import AtmosphericOptics, from_atmosphere_config
from ..config import AtmosphereConfig, RenderConfig
from ..environmental_optics import periodic_cloud_noise, procedural_star_catalogue
from ..starcatalogue import StarCatalogue

CLOUD_NOISE_UNIT = 8
STAR_CATALOGUE_UNIT = 9


class SkyPass:
    """Full-screen background pass.

    Owns the cloud-noise and star-catalogue textures. The catalogue is stored
    in J2000 equatorial coordinates and the view ray is rotated into that frame
    each second, so the texture is built once rather than re-rasterised as the
    sky turns.
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        config: RenderConfig,
        quad_buffer: moderngl.Buffer,
        atmosphere: AtmosphereConfig,
        tan_half_fov: float,
    ) -> None:
        self.ctx = ctx
        self.program = shaders.program(ctx, "quad.vert", "background.frag")
        self.vao = ctx.simple_vertex_array(
            self.program, quad_buffer, "in_position"
        )
        self.program["tan_half_fov"] = tan_half_fov
        self.program["aspect"] = config.width / config.height

        cloud_noise = periodic_cloud_noise()
        self.cloud_noise_texture = ctx.texture(
            (cloud_noise.shape[1], cloud_noise.shape[0]),
            components=1,
            data=cloud_noise.tobytes(),
            dtype="f1",
        )
        self.cloud_noise_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.cloud_noise_texture.repeat_x = True
        self.cloud_noise_texture.repeat_y = True

        # Prefer measured astrometry. Until `python -m tools.import_star_catalogue`
        # has been run, fall back to the procedural field and record that the
        # sky is not a real one.
        self.catalogue_source = StarCatalogue.load_if_present()
        if self.catalogue_source is not None:
            catalogue = self.catalogue_source.equatorial_flux_map()
            self.catalogue_is_measured = self.catalogue_source.is_measured
        else:
            catalogue = procedural_star_catalogue()
            self.catalogue_is_measured = False
        self.star_catalogue_texture = ctx.texture(
            (catalogue.shape[1], catalogue.shape[0]),
            components=3,
            data=catalogue.astype(np.float16).tobytes(),
            dtype="f2",
        )
        self.star_catalogue_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.star_catalogue_texture.repeat_x = True
        self.star_catalogue_texture.repeat_y = False

        self.program["cloud_noise"] = CLOUD_NOISE_UNIT
        self.program["star_catalogue"] = STAR_CATALOGUE_UNIT
        self.atmospheric_optics: AtmosphericOptics = from_atmosphere_config(
            atmosphere
        )
        self.program["star_optical_depth"] = float(
            self.atmospheric_optics.vertical_optical_depth(550.0)
        )
        self.program["airlight_field"] = 0.0
        self.set_celestial_frame(np.eye(3))

    # -- per-frame state ---------------------------------------------------

    def set_celestial_frame(self, rotation: np.ndarray) -> None:
        """East-Up-South to J2000 equatorial, as three unambiguous rows."""

        for axis, row in enumerate("xyz"):
            self.program[f"celestial_row_{row}"].value = tuple(
                float(value) for value in rotation[axis]
            )

    def set_atmosphere(self, atmosphere: AtmosphereConfig) -> None:
        """Track the observed pressure through the molecular optical depth."""

        optics = from_atmosphere_config(atmosphere)
        if optics != self.atmospheric_optics:
            self.atmospheric_optics = optics
            self.program["star_optical_depth"] = float(
                optics.vertical_optical_depth(550.0)
            )

    def set_camera(self, forward, right, up) -> None:
        self.program["camera_forward"].value = tuple(forward)
        self.program["camera_right"].value = tuple(right)
        self.program["camera_up"].value = tuple(up)

    def update_celestial(
        self,
        celestial: CelestialState,
        twilight_strength: float,
        cloud_cover: float,
    ) -> None:
        self.program["sun_direction"].value = tuple(celestial.sun_direction_eus)
        self.program["moon_direction"].value = tuple(
            celestial.moon_direction_eus
        )
        self.program["twilight_strength"] = float(twilight_strength)
        self.program["moon_strength"] = float(
            min(celestial.moon_illuminance_lux / 0.25, 1.0) * 0.04
        )
        self.program["cloud_cover"] = cloud_cover

    def set_time(self, time_s: float) -> None:
        self.program["time_s"] = time_s

    # -- drawing -----------------------------------------------------------

    def bind_textures(self) -> None:
        self.cloud_noise_texture.use(CLOUD_NOISE_UNIT)
        self.star_catalogue_texture.use(STAR_CATALOGUE_UNIT)

    def draw(self) -> None:
        self.bind_textures()
        self.vao.render(moderngl.TRIANGLE_STRIP)

    def draw_airlight(self, framebuffer: moderngl.Framebuffer) -> None:
        """Render the airlight source field the haze pass mixes toward.

        The same shader and the same weather state as the visible sky, with the
        view ray projected onto the horizontal and stars suppressed. Evaluating
        the sky model twice rather than reimplementing a horizon colour is what
        keeps aerial perspective and the sky from drifting apart: a change to
        twilight, cloud, or the moon reaches both by construction.

        The camera uniforms are whatever the coordinator set last, so this must
        be called after the main camera is bound and not between the reflection
        pre-pass's mirrored camera and its restoration.
        """

        framebuffer.use()
        self.program["airlight_field"] = 1.0
        self.bind_textures()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        self.program["airlight_field"] = 0.0
