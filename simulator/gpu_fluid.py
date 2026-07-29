from __future__ import annotations

import math

import moderngl
import numpy as np

from .config import AtmosphereConfig, SmokeConfig
from .fluid import SmokeFluid2D


QUAD_VERTEX = """
#version 330
in vec2 in_position;
void main() { gl_Position = vec4(in_position, 0.0, 1.0); }
"""


COMMON_VELOCITY_GLSL = """
uniform sampler2D velocity_u;
uniform sampler2D velocity_v;
uniform vec4 domain;
uniform vec2 cell_size;
uniform ivec2 cell_count;

float sample_u(vec2 world) {
    vec2 index = vec2(
        (world.x - domain.x) / cell_size.x + 0.5,
        (world.y - domain.z) / cell_size.y
    );
    return texture(
        velocity_u, index / vec2(cell_count.x + 1, cell_count.y)
    ).r;
}
float sample_v(vec2 world) {
    vec2 index = vec2(
        (world.x - domain.x) / cell_size.x,
        (world.y - domain.z) / cell_size.y + 0.5
    );
    return texture(
        velocity_v, index / vec2(cell_count.x, cell_count.y + 1)
    ).r;
}
vec2 sample_velocity(vec2 world) {
    return vec2(sample_u(world), sample_v(world));
}
vec2 midpoint_backtrace(vec2 world, float dt_s) {
    vec2 initial_velocity = sample_velocity(world);
    vec2 midpoint = world - 0.5 * dt_s * initial_velocity;
    return world - dt_s * sample_velocity(midpoint);
}
"""


STATE_ADVECTION_FRAGMENT = (
    """
#version 330
uniform sampler2D source_state;
uniform sampler2D source_increment;
uniform float dt_s;
out vec2 output_state;
"""
    + COMMON_VELOCITY_GLSL
    + """
void main() {
    vec2 world = vec2(
        domain.x + gl_FragCoord.x * cell_size.x,
        domain.z + gl_FragCoord.y * cell_size.y
    );
    vec2 previous_world = midpoint_backtrace(world, dt_s);
    vec2 uv = (previous_world - domain.xz) / (domain.yw - domain.xz);
    ivec2 cell = ivec2(gl_FragCoord.xy);
    output_state = max(
        texture(source_state, uv).rg
        + texelFetch(source_increment, cell, 0).rg,
        vec2(0.0)
    );
}
"""
)


U_ADVECTION_FRAGMENT = (
    """
#version 330
uniform sampler2D source_field;
uniform float dt_s;
out float output_velocity;
"""
    + COMMON_VELOCITY_GLSL
    + """
void main() {
    vec2 world = vec2(
        domain.x + (gl_FragCoord.x - 0.5) * cell_size.x,
        domain.z + gl_FragCoord.y * cell_size.y
    );
    vec2 previous_world = midpoint_backtrace(world, dt_s);
    vec2 index = vec2(
        (previous_world.x - domain.x) / cell_size.x + 0.5,
        (previous_world.y - domain.z) / cell_size.y
    );
    output_velocity = texture(
        source_field, index / vec2(cell_count.x + 1, cell_count.y)
    ).r;
}
"""
)


V_ADVECTION_FRAGMENT = (
    """
#version 330
uniform sampler2D source_field;
uniform float dt_s;
out float output_velocity;
"""
    + COMMON_VELOCITY_GLSL
    + """
void main() {
    vec2 world = vec2(
        domain.x + gl_FragCoord.x * cell_size.x,
        domain.z + (gl_FragCoord.y - 0.5) * cell_size.y
    );
    vec2 previous_world = midpoint_backtrace(world, dt_s);
    vec2 index = vec2(
        (previous_world.x - domain.x) / cell_size.x,
        (previous_world.y - domain.z) / cell_size.y + 0.5
    );
    output_velocity = texture(
        source_field, index / vec2(cell_count.x, cell_count.y + 1)
    ).r;
}
"""
)


STATE_EVOLUTION_FRAGMENT = """
#version 330
uniform sampler2D source_state;
uniform vec2 cell_size;
uniform vec2 diffusivity;
uniform vec2 half_life_s;
uniform float dt_s;
out vec2 output_state;
void main() {
    ivec2 size = textureSize(source_state, 0);
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 left = max(cell - ivec2(1, 0), ivec2(0));
    ivec2 right = min(cell + ivec2(1, 0), size - 1);
    ivec2 down = max(cell - ivec2(0, 1), ivec2(0));
    ivec2 up = min(cell + ivec2(0, 1), size - 1);
    vec2 centre = texelFetch(source_state, cell, 0).rg;
    vec2 laplacian =
          (texelFetch(source_state, left, 0).rg
           - 2.0 * centre
           + texelFetch(source_state, right, 0).rg)
          / (cell_size.x * cell_size.x)
        + (texelFetch(source_state, down, 0).rg
           - 2.0 * centre
           + texelFetch(source_state, up, 0).rg)
          / (cell_size.y * cell_size.y);
    vec2 decay = exp(-log(2.0) * dt_s / half_life_s);
    output_state = max((centre + diffusivity * dt_s * laplacian) * decay,
                       vec2(0.0));
}
"""


VELOCITY_FORCE_COMMON = (
    """
uniform sampler2D smoke_state;
uniform sampler2D source_field;
uniform float dt_s;
uniform float viscosity;
uniform float vorticity_confinement;
uniform float ambient_temperature_k;
uniform float air_density_kg_m3;
uniform float background_wind_mps;
"""
    + COMMON_VELOCITY_GLSL
    + """
float curl(vec2 world) {
    float dvdx = (
        sample_v(world + vec2(0.5 * cell_size.x, 0.0))
        - sample_v(world - vec2(0.5 * cell_size.x, 0.0))
    ) / cell_size.x;
    float dudy = (
        sample_u(world + vec2(0.0, 0.5 * cell_size.y))
        - sample_u(world - vec2(0.0, 0.5 * cell_size.y))
    ) / cell_size.y;
    return dvdx - dudy;
}
vec2 vorticity_force(vec2 world) {
    float centre_curl = curl(world);
    float gradient_x = (
        abs(curl(world + vec2(cell_size.x, 0.0)))
        - abs(curl(world - vec2(cell_size.x, 0.0)))
    ) / (2.0 * cell_size.x);
    float gradient_y = (
        abs(curl(world + vec2(0.0, cell_size.y)))
        - abs(curl(world - vec2(0.0, cell_size.y)))
    ) / (2.0 * cell_size.y);
    vec2 gradient = vec2(gradient_x, gradient_y);
    float gradient_length = length(gradient);
    if (gradient_length < 1e-7) return vec2(0.0);
    vec2 normal = gradient / gradient_length;
    return vorticity_confinement * centre_curl
         * vec2(cell_size.x * normal.y, -cell_size.y * normal.x);
}
float laplacian_field(ivec2 cell) {
    ivec2 size = textureSize(source_field, 0);
    ivec2 left = max(cell - ivec2(1, 0), ivec2(0));
    ivec2 right = min(cell + ivec2(1, 0), size - 1);
    ivec2 down = max(cell - ivec2(0, 1), ivec2(0));
    ivec2 up = min(cell + ivec2(0, 1), size - 1);
    float centre = texelFetch(source_field, cell, 0).r;
    return (
          (texelFetch(source_field, left, 0).r
           - 2.0 * centre
           + texelFetch(source_field, right, 0).r)
          / (cell_size.x * cell_size.x)
        + (texelFetch(source_field, down, 0).r
           - 2.0 * centre
           + texelFetch(source_field, up, 0).r)
          / (cell_size.y * cell_size.y)
    );
}
"""
)


U_FORCE_FRAGMENT = (
    """
#version 330
out float output_velocity;
"""
    + VELOCITY_FORCE_COMMON
    + """
void main() {
    ivec2 face = ivec2(gl_FragCoord.xy);
    int nx = cell_count.x;
    if (face.x == 0) {
        output_velocity = background_wind_mps;
        return;
    }
    if (face.x == nx) {
        output_velocity = texelFetch(
            source_field, ivec2(nx - 1, face.y), 0
        ).r;
        return;
    }
    vec2 world = vec2(
        domain.x + float(face.x) * cell_size.x,
        domain.z + (float(face.y) + 0.5) * cell_size.y
    );
    float velocity = texelFetch(source_field, face, 0).r;
    velocity += dt_s * (
        vorticity_force(world).x + viscosity * laplacian_field(face)
    );
    output_velocity = velocity;
}
"""
)


V_FORCE_FRAGMENT = (
    """
#version 330
out float output_velocity;
"""
    + VELOCITY_FORCE_COMMON
    + """
void main() {
    ivec2 face = ivec2(gl_FragCoord.xy);
    int ny = cell_count.y;
    if (face.y == 0) {
        output_velocity = 0.0;
        return;
    }
    if (face.y == ny) {
        output_velocity = texelFetch(
            source_field, ivec2(face.x, ny - 1), 0
        ).r;
        return;
    }
    vec2 world = vec2(
        domain.x + (float(face.x) + 0.5) * cell_size.x,
        domain.z + float(face.y) * cell_size.y
    );
    vec2 state_uv = (world - domain.xz) / (domain.yw - domain.xz);
    vec2 state = texture(smoke_state, state_uv).rg;
    float buoyancy = 9.80665 * (
        state.g / ambient_temperature_k - state.r / air_density_kg_m3
    );
    float velocity = texelFetch(source_field, face, 0).r;
    velocity += dt_s * (
        buoyancy
        + vorticity_force(world).y
        + viscosity * laplacian_field(face)
    );
    output_velocity = velocity;
}
"""
)


DIVERGENCE_FRAGMENT = """
#version 330
uniform sampler2D velocity_u;
uniform sampler2D velocity_v;
uniform vec2 cell_size;
out float output_divergence;
void main() {
    ivec2 cell = ivec2(gl_FragCoord.xy);
    float u_left = texelFetch(velocity_u, cell, 0).r;
    float u_right = texelFetch(
        velocity_u, cell + ivec2(1, 0), 0
    ).r;
    float v_down = texelFetch(velocity_v, cell, 0).r;
    float v_up = texelFetch(
        velocity_v, cell + ivec2(0, 1), 0
    ).r;
    output_divergence = (u_right - u_left) / cell_size.x
                      + (v_up - v_down) / cell_size.y;
}
"""


PRESSURE_FRAGMENT = """
#version 330
uniform sampler2D source_pressure;
uniform sampler2D divergence;
uniform vec2 cell_size;
uniform float air_density_kg_m3;
uniform float dt_s;
out float output_pressure;
float rhs_at(ivec2 cell) {
    return air_density_kg_m3
         * texelFetch(divergence, cell, 0).r / dt_s;
}
float jacobi_once(ivec2 cell) {
    ivec2 size = textureSize(source_pressure, 0);
    ivec2 left = max(cell - ivec2(1, 0), ivec2(0));
    ivec2 right = min(cell + ivec2(1, 0), size - 1);
    ivec2 down = max(cell - ivec2(0, 1), ivec2(0));
    ivec2 up = min(cell + ivec2(0, 1), size - 1);
    float dx2 = cell_size.x * cell_size.x;
    float dy2 = cell_size.y * cell_size.y;
    return (
        (texelFetch(source_pressure, left, 0).r
         + texelFetch(source_pressure, right, 0).r) * dy2
        + (texelFetch(source_pressure, down, 0).r
           + texelFetch(source_pressure, up, 0).r) * dx2
        - rhs_at(cell) * dx2 * dy2
    ) / (2.0 * (dx2 + dy2));
}
void main() {
    ivec2 size = textureSize(source_pressure, 0);
    ivec2 cell = ivec2(gl_FragCoord.xy);
    ivec2 left = max(cell - ivec2(1, 0), ivec2(0));
    ivec2 right = min(cell + ivec2(1, 0), size - 1);
    ivec2 down = max(cell - ivec2(0, 1), ivec2(0));
    ivec2 up = min(cell + ivec2(0, 1), size - 1);
    float dx2 = cell_size.x * cell_size.x;
    float dy2 = cell_size.y * cell_size.y;
    output_pressure = (
        (jacobi_once(left) + jacobi_once(right)) * dy2
        + (jacobi_once(down) + jacobi_once(up)) * dx2
        - rhs_at(cell) * dx2 * dy2
    ) / (2.0 * (dx2 + dy2));
}
"""


PROJECT_U_FRAGMENT = """
#version 330
uniform sampler2D source_velocity;
uniform sampler2D pressure;
uniform float dt_s;
uniform float air_density_kg_m3;
uniform float dx;
uniform float background_wind_mps;
out float output_velocity;
void main() {
    ivec2 face = ivec2(gl_FragCoord.xy);
    int nx = textureSize(pressure, 0).x;
    if (face.x == 0) {
        output_velocity = background_wind_mps;
    } else if (face.x == nx) {
        output_velocity = texelFetch(
            source_velocity, ivec2(nx - 1, face.y), 0
        ).r;
    } else {
        float p_right = texelFetch(pressure, face, 0).r;
        float p_left = texelFetch(
            pressure, face - ivec2(1, 0), 0
        ).r;
        output_velocity = texelFetch(source_velocity, face, 0).r
            - dt_s * (p_right - p_left) / (air_density_kg_m3 * dx);
    }
}
"""


PROJECT_V_FRAGMENT = """
#version 330
uniform sampler2D source_velocity;
uniform sampler2D pressure;
uniform float dt_s;
uniform float air_density_kg_m3;
uniform float dy;
out float output_velocity;
void main() {
    ivec2 face = ivec2(gl_FragCoord.xy);
    int ny = textureSize(pressure, 0).y;
    if (face.y == 0) {
        output_velocity = 0.0;
    } else if (face.y == ny) {
        output_velocity = texelFetch(
            source_velocity, ivec2(face.x, ny - 1), 0
        ).r;
    } else {
        float p_up = texelFetch(pressure, face, 0).r;
        float p_down = texelFetch(
            pressure, face - ivec2(0, 1), 0
        ).r;
        output_velocity = texelFetch(source_velocity, face, 0).r
            - dt_s * (p_up - p_down) / (air_density_kg_m3 * dy);
    }
}
"""


class GpuSmokeFluid2D(SmokeFluid2D):
    """OpenGL 3.3 ping-pong implementation of the existing MAC solver."""

    backend_name = "gpu_fragment_mac"

    @property
    def render_state_texture(self) -> moderngl.Texture:
        return self.state_textures[0]

    def __init__(
        self,
        ctx: moderngl.Context,
        config: SmokeConfig,
        atmosphere: AtmosphereConfig,
    ) -> None:
        super().__init__(config, atmosphere)
        self.ctx = ctx
        self.update_hz = config.gpu_update_hz
        self.pressure_iterations_per_step = max(
            round(
                config.pressure_iterations
                * config.update_hz
                / self.update_hz
            ),
            1,
        )
        quad = np.array(
            [-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32
        )
        self.quad_buffer = ctx.buffer(quad.tobytes())
        self.programs = {
            "state_advection": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=STATE_ADVECTION_FRAGMENT,
            ),
            "u_advection": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=U_ADVECTION_FRAGMENT,
            ),
            "v_advection": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=V_ADVECTION_FRAGMENT,
            ),
            "state_evolution": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=STATE_EVOLUTION_FRAGMENT,
            ),
            "u_force": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=U_FORCE_FRAGMENT,
            ),
            "v_force": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=V_FORCE_FRAGMENT,
            ),
            "divergence": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=DIVERGENCE_FRAGMENT,
            ),
            "pressure": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=PRESSURE_FRAGMENT,
            ),
            "project_u": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=PROJECT_U_FRAGMENT,
            ),
            "project_v": ctx.program(
                vertex_shader=QUAD_VERTEX,
                fragment_shader=PROJECT_V_FRAGMENT,
            ),
        }
        self.vaos = {
            name: ctx.simple_vertex_array(program, self.quad_buffer, "in_position")
            for name, program in self.programs.items()
        }
        self.state_textures = self._textures((self.nx, self.ny), 2)
        self.source_texture = self._texture((self.nx, self.ny), 2)
        self.u_textures = self._textures((self.nx + 1, self.ny), 1)
        self.v_textures = self._textures((self.nx, self.ny + 1), 1)
        self.pressure_textures = self._textures((self.nx, self.ny), 1)
        self.divergence_texture = self._texture((self.nx, self.ny), 1)
        self.state_fbos = [ctx.framebuffer([t]) for t in self.state_textures]
        self.u_fbos = [ctx.framebuffer([t]) for t in self.u_textures]
        self.v_fbos = [ctx.framebuffer([t]) for t in self.v_textures]
        self.pressure_fbos = [
            ctx.framebuffer([t]) for t in self.pressure_textures
        ]
        self.divergence_fbo = ctx.framebuffer([self.divergence_texture])
        self.velocity_index = 0
        self.last_pressure_index = 0
        self.cpu_arrays_are_snapshot = False
        self.total_smoke_mass_kg = 0.0
        self.active_minimum_xy = np.array(
            [self.x_max, self.y_max], dtype=np.float32
        )
        self.active_maximum_xy = np.array(
            [self.x_min, self.y_min], dtype=np.float32
        )
        self._configure_uniforms()
        self._upload_initial_state()
        for texture in self.pressure_textures:
            texture.write(np.zeros((self.ny, self.nx), np.float32).tobytes())
        self.divergence_texture.write(
            np.zeros((self.ny, self.nx), np.float32).tobytes()
        )

    def _texture(
        self, size: tuple[int, int], components: int
    ) -> moderngl.Texture:
        texture = self.ctx.texture(size, components=components, dtype="f4")
        texture.filter = moderngl.LINEAR, moderngl.LINEAR
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _textures(
        self, size: tuple[int, int], components: int
    ) -> list[moderngl.Texture]:
        return [self._texture(size, components) for _ in range(2)]

    def _configure_uniforms(self) -> None:
        domain = (self.x_min, self.x_max, self.y_min, self.y_max)
        cell_size = (self.dx, self.dy)
        cell_count = (self.nx, self.ny)
        for name in (
            "state_advection", "u_advection", "v_advection",
            "u_force", "v_force",
        ):
            program = self.programs[name]
            program["velocity_u"] = 1
            program["velocity_v"] = 2
            program["domain"].value = domain
            program["cell_size"].value = cell_size
            program["cell_count"].value = cell_count
        self.programs["state_advection"]["source_state"] = 0
        self.programs["state_advection"]["source_increment"] = 6
        self.programs["u_advection"]["source_field"] = 0
        self.programs["v_advection"]["source_field"] = 0
        state_evolution = self.programs["state_evolution"]
        state_evolution["source_state"] = 0
        state_evolution["cell_size"].value = cell_size
        state_evolution["diffusivity"].value = (
            self.config.smoke_diffusivity_m2_s,
            self.config.thermal_diffusivity_m2_s,
        )
        state_evolution["half_life_s"].value = (
            self.config.smoke_half_life_s,
            self.config.thermal_half_life_s,
        )
        for name in ("u_force", "v_force"):
            program = self.programs[name]
            program["source_field"] = 0
            program["viscosity"] = self.config.kinematic_viscosity_m2_s
            program["vorticity_confinement"] = (
                self.config.vorticity_confinement
            )
        self.programs["v_force"]["smoke_state"] = 3
        divergence = self.programs["divergence"]
        divergence["velocity_u"] = 1
        divergence["velocity_v"] = 2
        divergence["cell_size"].value = cell_size
        pressure = self.programs["pressure"]
        pressure["source_pressure"] = 0
        pressure["divergence"] = 4
        pressure["cell_size"].value = cell_size
        for name in ("project_u", "project_v"):
            program = self.programs[name]
            program["source_velocity"] = 0
            program["pressure"] = 5
        self.programs["project_u"]["dx"] = self.dx
        self.programs["project_v"]["dy"] = self.dy

    def _set_step_uniforms(self, dt_s: float) -> None:
        for name in (
            "state_advection", "u_advection", "v_advection",
            "state_evolution", "u_force", "v_force", "pressure",
            "project_u", "project_v",
        ):
            self.programs[name]["dt_s"] = dt_s
        self.programs["u_force"]["background_wind_mps"] = (
            self.background_wind_mps
        )
        self.programs["v_force"]["ambient_temperature_k"] = (
            self.ambient_temperature_k
        )
        self.programs["v_force"]["air_density_kg_m3"] = (
            self.air_density_kg_m3
        )
        self.programs["pressure"]["air_density_kg_m3"] = (
            self.air_density_kg_m3
        )
        for name in ("project_u", "project_v"):
            self.programs[name]["air_density_kg_m3"] = self.air_density_kg_m3
        self.programs["project_u"]["background_wind_mps"] = (
            self.background_wind_mps
        )

    def _upload_initial_state(self) -> None:
        state = np.empty((self.ny, self.nx, 2), dtype=np.float32)
        state[:, :, 0] = self.density_kg_m3
        state[:, :, 1] = self.temperature_excess_k
        for texture in self.state_textures:
            texture.write(state.tobytes())
        for texture in self.u_textures:
            texture.write(self.u_mps.tobytes())
        for texture in self.v_textures:
            texture.write(self.v_mps.tobytes())
        self.source_texture.write(
            np.zeros_like(state, dtype=np.float32).tobytes()
        )

    def _extend_active_bounds(
        self, positions_m: np.ndarray, margin_m: float
    ) -> None:
        if not len(positions_m):
            return
        positions_xy = np.asarray(positions_m)[:, :2]
        self.active_minimum_xy = np.minimum(
            self.active_minimum_xy,
            positions_xy.min(axis=0) - margin_m,
        )
        self.active_maximum_xy = np.maximum(
            self.active_maximum_xy,
            positions_xy.max(axis=0) + margin_m,
        )
        self.active_minimum_xy[:] = np.maximum(
            self.active_minimum_xy, (self.x_min, self.y_min)
        )
        self.active_maximum_xy[:] = np.minimum(
            self.active_maximum_xy, (self.x_max, self.y_max)
        )

    def inject_burst(
        self,
        position_m: np.ndarray,
        smoke_mass_kg: float,
        thermal_energy_j: float,
    ) -> None:
        self._discard_diagnostic_snapshot()
        mass_before = float(
            self.density_kg_m3.sum(dtype=np.float64)
        )
        super().inject_burst(
            position_m, smoke_mass_kg, thermal_energy_j
        )
        density_delta = (
            float(self.density_kg_m3.sum(dtype=np.float64)) - mass_before
        )
        if density_delta > 0.0:
            self.total_smoke_mass_kg += density_delta * (
                self.dx * self.dy * self.config.plume_depth_m
            )
            self._extend_active_bounds(
                np.asarray(position_m, dtype=np.float32).reshape(1, 3),
                self.config.source_radius_m * 3.0,
            )

    def inject_particles(
        self,
        positions_m: np.ndarray,
        smoke_mass_kg: np.ndarray,
        thermal_energy_j: np.ndarray,
    ) -> tuple[float, float]:
        self._discard_diagnostic_snapshot()
        accepted_mass, accepted_energy = super().inject_particles(
            positions_m, smoke_mass_kg, thermal_energy_j
        )
        if accepted_mass > 0.0:
            self.total_smoke_mass_kg += accepted_mass
            self._extend_active_bounds(
                np.asarray(positions_m),
                max(self.dx, self.dy),
            )
        return accepted_mass, accepted_energy

    def _discard_diagnostic_snapshot(self) -> None:
        if self.cpu_arrays_are_snapshot:
            self.density_kg_m3.fill(0.0)
            self.temperature_excess_k.fill(0.0)
            self.cpu_arrays_are_snapshot = False

    def has_visible_smoke(self) -> bool:
        return self.total_smoke_mass_kg > 1e-9

    def active_render_bounds(
        self, volume_depth_m: float
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.has_visible_smoke():
            return None
        half_depth = 0.5 * volume_depth_m
        return (
            np.asarray(
                [
                    self.active_minimum_xy[0],
                    self.active_minimum_xy[1],
                    -half_depth,
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    self.active_maximum_xy[0],
                    self.active_maximum_xy[1],
                    half_depth,
                ],
                dtype=np.float32,
            ),
        )

    def _render(self, name: str, framebuffer: moderngl.Framebuffer) -> None:
        framebuffer.use()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.vaos[name].render(moderngl.TRIANGLE_STRIP)

    def step(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        self._set_step_uniforms(dt_s)
        self._discard_diagnostic_snapshot()
        source = np.empty((self.ny, self.nx, 2), dtype=np.float32)
        source[:, :, 0] = self.density_kg_m3
        source[:, :, 1] = self.temperature_excess_k
        self.source_texture.write(source.tobytes())
        self.density_kg_m3.fill(0.0)
        self.temperature_excess_k.fill(0.0)
        velocity_input = self.velocity_index
        velocity_advected = 1 - velocity_input

        self.state_textures[0].use(0)
        self.u_textures[velocity_input].use(1)
        self.v_textures[velocity_input].use(2)
        self.source_texture.use(6)
        self._render("state_advection", self.state_fbos[1])
        self.u_textures[velocity_input].use(0)
        self._render("u_advection", self.u_fbos[velocity_advected])
        self.v_textures[velocity_input].use(0)
        self._render("v_advection", self.v_fbos[velocity_advected])

        self.state_textures[1].use(0)
        self._render("state_evolution", self.state_fbos[0])
        self.u_textures[velocity_advected].use(0)
        self.u_textures[velocity_advected].use(1)
        self.v_textures[velocity_advected].use(2)
        self.state_textures[0].use(3)
        self._render("u_force", self.u_fbos[velocity_input])
        self.v_textures[velocity_advected].use(0)
        self._render("v_force", self.v_fbos[velocity_input])

        self.u_textures[velocity_input].use(1)
        self.v_textures[velocity_input].use(2)
        self._render("divergence", self.divergence_fbo)
        for framebuffer in self.pressure_fbos:
            framebuffer.clear(0.0)
        pressure_index = 0
        self.divergence_texture.use(4)
        pressure_passes = math.ceil(
            self.pressure_iterations_per_step / 2
        )
        for _ in range(pressure_passes):
            output_index = 1 - pressure_index
            self.pressure_textures[pressure_index].use(0)
            self._render("pressure", self.pressure_fbos[output_index])
            pressure_index = output_index

        self.pressure_textures[pressure_index].use(5)
        self.u_textures[velocity_input].use(0)
        self._render("project_u", self.u_fbos[velocity_advected])
        self.v_textures[velocity_input].use(0)
        self._render("project_v", self.v_fbos[velocity_advected])
        self.velocity_index = velocity_advected
        self.last_pressure_index = pressure_index
        self.total_smoke_mass_kg *= math.exp(
            -math.log(2.0) * dt_s / self.config.smoke_half_life_s
        )
        if self.has_visible_smoke():
            horizontal_growth = (
                abs(self.background_wind_mps) + 8.0
            ) * dt_s
            self.active_minimum_xy += (
                -horizontal_growth,
                -1.0 * dt_s,
            )
            self.active_maximum_xy += (
                horizontal_growth,
                12.0 * dt_s,
            )
            self.active_minimum_xy[:] = np.maximum(
                self.active_minimum_xy, (self.x_min, self.y_min)
            )
            self.active_maximum_xy[:] = np.minimum(
                self.active_maximum_xy, (self.x_max, self.y_max)
            )
        self.revision += 1

    def readback(self) -> None:
        """Synchronize GPU fields for diagnostics and conservation tests."""

        state_result = np.frombuffer(
            self.state_textures[0].read(alignment=1), dtype=np.float32
        ).reshape(self.ny, self.nx, 2)
        self.density_kg_m3[:] = state_result[:, :, 0]
        self.temperature_excess_k[:] = state_result[:, :, 1]
        self.u_mps[:] = np.frombuffer(
            self.u_textures[self.velocity_index].read(alignment=1),
            dtype=np.float32,
        ).reshape(self.ny, self.nx + 1)
        self.v_mps[:] = np.frombuffer(
            self.v_textures[self.velocity_index].read(alignment=1),
            dtype=np.float32,
        ).reshape(self.ny + 1, self.nx)
        self.pressure_pa[:] = np.frombuffer(
            self.pressure_textures[self.last_pressure_index].read(alignment=1),
            dtype=np.float32,
        ).reshape(self.ny, self.nx)
        np.maximum(self.density_kg_m3, 0.0, out=self.density_kg_m3)
        np.maximum(
            self.temperature_excess_k, 0.0, out=self.temperature_excess_k
        )
        self.cpu_arrays_are_snapshot = True


def create_smoke_solver(
    ctx: moderngl.Context,
    config: SmokeConfig,
    atmosphere: AtmosphereConfig,
) -> SmokeFluid2D:
    if config.prefer_gpu_solver:
        try:
            return GpuSmokeFluid2D(ctx, config, atmosphere)
        except (moderngl.Error, KeyError, ValueError):
            pass
    solver = SmokeFluid2D(config, atmosphere)
    return solver
