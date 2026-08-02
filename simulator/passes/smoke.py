"""Volumetric plume, ray-marched against the opaque depth buffer."""

from __future__ import annotations

import math

import moderngl
import numpy as np

from .. import shaders
from ..config import SmokeConfig
from ..volume import active_slice_bounds, box_vertices

SMOKE_STATE_2D_UNIT = 4
SMOKE_STATE_3D_UNIT = 5
SCENE_DEPTH_UNIT = 6

BOX_INDICES = np.array(
    [
        0, 3, 2, 0, 2, 1,
        4, 5, 6, 4, 6, 7,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
        0, 1, 5, 0, 5, 4,
        3, 7, 6, 3, 6, 2,
    ],
    dtype=np.uint32,
)


class SmokePass:
    """Front-to-back Beer-Lambert integration through the plume volume.

    The proxy box is shrunk to the fluid's active bounds each step, so empty
    parts of the domain create neither fragments nor ray loops. Texture
    coordinates still reference the full domain: this is empty-space skipping,
    not a change to density, temperature, or optical path length.
    """

    def __init__(self, ctx: moderngl.Context, config: SmokeConfig) -> None:
        self.ctx = ctx
        self.config = config
        x0, x1, y0, y1 = config.bounds_m
        z0 = -0.5 * config.volume_depth_m
        z1 = 0.5 * config.volume_depth_m

        self.program = shaders.program(ctx, "smoke.vert", "smoke.frag")
        self.buffer = ctx.buffer(
            box_vertices((x0, y0, z0), (x1, y1, z1)).tobytes(), dynamic=True
        )
        self.index_buffer = ctx.buffer(BOX_INDICES.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.buffer, "3f", "in_position")],
            self.index_buffer,
            index_element_size=4,
        )
        self.state_texture = ctx.texture(
            config.grid_size, components=2, dtype="f4"
        )
        self.state_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.state_texture.repeat_x = False
        self.state_texture.repeat_y = False
        self.program["smoke_state"] = SMOKE_STATE_2D_UNIT
        self.program["smoke_state_3d"] = SMOKE_STATE_3D_UNIT
        self.program["scene_depth"] = SCENE_DEPTH_UNIT
        self.program["smoke_is_3d"] = 0
        self.program["depth_bias_m"] = config.volume_depth_bias_m
        self.program["volume_min"].value = (x0, y0, z0)
        self.program["volume_max"].value = (x1, y1, z1)
        self.program["smoke_xy_bounds"].value = (x0, y0, x1, y1)
        self.program["smoke_field_min"].value = (x0, y0, z0)
        self.program["smoke_field_max"].value = (x1, y1, z1)

        dz = config.volume_depth_m / config.volume_slices
        depth_sigma_m = max(config.plume_depth_m * 0.5, dz)
        depth_integral = (
            math.sqrt(2.0 * math.pi)
            * depth_sigma_m
            * math.erf(
                config.volume_depth_m / (2.0 * math.sqrt(2.0) * depth_sigma_m)
            )
        )
        self.program["depth_profile_sigma_m"] = depth_sigma_m
        self.program["depth_profile_scale"] = (
            config.plume_depth_m / depth_integral
        )
        # Four sigma on either side retains >99.993% of the normalized
        # Gaussian mass while avoiding fragments in optically empty tails.
        self.active_depth_m = min(config.volume_depth_m, 8.0 * depth_sigma_m)
        self.program["ray_steps"] = config.volume_ray_steps
        self.bounds = np.array([[x0, y0, z0], [x1, y1, z1]], dtype=np.float32)
        self.texture_bounds = self.bounds.copy()
        self.revision = -1

    def set_view_projection(
        self, matrix_bytes: bytes, inverse_bytes: bytes, camera_position
    ) -> None:
        self.program["view_projection"].write(matrix_bytes)
        self.program["inverse_view_projection"].write(inverse_bytes)
        self.program["camera_position"].value = tuple(camera_position)

    def _refresh_bounds(self, smoke, camera_position: np.ndarray) -> None:
        active_bounds_method = getattr(smoke, "active_render_bounds", None)
        active_bounds = (
            active_bounds_method(self.active_depth_m)
            if active_bounds_method is not None
            else active_slice_bounds(
                smoke.density_kg_m3,
                (self.texture_bounds[0, 0], self.texture_bounds[0, 1]),
                (self.texture_bounds[1, 0], self.texture_bounds[1, 1]),
                -0.5 * self.active_depth_m,
                0.5 * self.active_depth_m,
            )
        )
        if active_bounds is None:
            return
        minimum, maximum = active_bounds
        self.bounds[:] = (minimum, maximum)
        self.buffer.write(box_vertices(minimum, maximum).tobytes())
        self.program["volume_min"].value = tuple(minimum)
        self.program["volume_max"].value = tuple(maximum)

    def set_camera_inside(self, camera_position) -> None:
        """Which face of the proxy box the ray marcher must keep.

        Set per draw rather than per fluid revision. The frame draws this pass
        twice from two different cameras — the viewer's and the one mirrored
        below the water datum — and the mirrored one is always outside a plume
        that sits above the datum.
        """

        self.program["camera_inside"] = int(
            np.all(camera_position >= self.bounds[0])
            and np.all(camera_position <= self.bounds[1])
        )

    def draw(
        self,
        smoke,
        camera_position: np.ndarray,
        depth_texture,
        reflected_path: bool = False,
    ) -> None:
        if smoke.revision != self.revision:
            if getattr(smoke, "render_state_texture", None) is None:
                # CPU solver: upload the two-channel slice the shader samples.
                state = np.empty(
                    (
                        smoke.density_kg_m3.shape[0],
                        smoke.density_kg_m3.shape[1],
                        2,
                    ),
                    dtype=np.float32,
                )
                state[:, :, 0] = smoke.density_kg_m3
                state[:, :, 1] = smoke.temperature_excess_k
                self.state_texture.write(state.tobytes())
            self._refresh_bounds(smoke, camera_position)
            self.revision = smoke.revision

        state_texture = getattr(
            smoke, "render_state_texture", self.state_texture
        )
        is_3d = int(getattr(smoke, "is_3d", False))
        self.program["smoke_is_3d"] = is_3d
        if is_3d:
            self.program["smoke_field_min"].value = (
                smoke.x_min, smoke.y_min, smoke.z_min
            )
            self.program["smoke_field_max"].value = (
                smoke.x_max, smoke.y_max, smoke.z_max
            )
        state_texture.use(SMOKE_STATE_3D_UNIT if is_3d else SMOKE_STATE_2D_UNIT)
        depth_texture.use(SCENE_DEPTH_UNIT)
        self.program["reflected_path"] = int(reflected_path)
        self.set_camera_inside(camera_position)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self.vao.render(moderngl.TRIANGLES)
