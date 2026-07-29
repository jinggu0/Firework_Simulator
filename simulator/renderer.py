from __future__ import annotations

import math

import moderngl
import numpy as np

from .config import RenderConfig
from .physics import FireworkWorld
from .water import WaterConfig, build_directional_spectrum, build_water_mesh


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    return np.array(
        [[f / aspect, 0, 0, 0], [0, f, 0, 0],
         [0, 0, (far + near) / (near - far), -1],
         [0, 0, 2 * far * near / (near - far), 0]],
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


QUAD_VERTEX = """
#version 330
in vec2 in_position;
out vec2 uv;
void main() { uv = in_position * .5 + .5; gl_Position = vec4(in_position, 0, 1); }
"""

BACKGROUND_FRAGMENT = """
#version 330
in vec2 uv; out vec4 frag_color; uniform float time_s;
void main() {
    float horizon = smoothstep(.18, .58, uv.y);
    vec3 sky = mix(vec3(.0030, .0040, .0060),
                   vec3(.00020, .00035, .00075), horizon);
    float water_mask = 1.0 - smoothstep(0.0, .19, uv.y);
    float ripple = sin(uv.x * 410.0 + time_s * 1.2)
                 * sin(uv.x * 93.0 - time_s * .7);
    vec3 water = vec3(.00018, .00032, .00042) + ripple * .00005;
    frag_color = vec4(mix(sky, water, water_mask), 1);
}
"""

PARTICLE_VERTEX = """
#version 330
in vec3 in_position; in vec3 in_color; in float in_power;
uniform mat4 view_projection; uniform float reflection; uniform float time_s;
out vec3 particle_color; out float particle_power;
void main() {
    vec3 p = in_position;
    if (reflection > .5) {
        p.y = -p.y;
        p.x += sin(p.x * .12 + time_s * 2.1) * (.7 + .012 * abs(p.y));
    }
    vec4 clip = view_projection * vec4(p, 1);
    gl_Position = clip;
    gl_PointSize = clamp(2.0 + 1150.0 / max(clip.w, 1.0), 2.0, 18.0);
    particle_color = in_color;
    particle_power = in_power * (reflection > .5 ? .13 : 1.0);
}
"""

PARTICLE_FRAGMENT = """
#version 330
in vec3 particle_color; in float particle_power; out vec4 frag_color;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(q, q);
    if (r2 > 1.0) discard;
    float radiance = (exp(-r2 * 18.0) + exp(-r2 * 3.0) * .16)
                   * particle_power * .006;
    frag_color = vec4(particle_color * radiance, 1);
}
"""

TONEMAP_FRAGMENT = """
#version 330
uniform sampler2D hdr_texture; uniform float exposure_scale;
in vec2 uv; out vec4 frag_color;
vec3 aces(vec3 x) {
    return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14), 0.0, 1.0);
}
void main() {
    vec3 mapped = aces(texture(hdr_texture, uv).rgb * exposure_scale);
    frag_color = vec4(pow(mapped, vec3(1.0/2.2)), 1);
}
"""

WATER_VERTEX = """
#version 330
in vec2 in_xz;
uniform mat4 view_projection;
uniform vec4 waves[32];
uniform float phases[32];
uniform float time_s;
uniform float choppiness;
out vec3 world_position;
out vec3 world_normal;
void main() {
    vec2 displaced_xz = in_xz;
    float height = 0.0;
    vec2 gradient = vec2(0.0);
    for (int i = 0; i < 32; ++i) {
        vec2 direction = waves[i].xy;
        float k = waves[i].z;
        float amplitude = waves[i].w;
        float omega = sqrt(9.80665 * k);
        float theta = k * dot(direction, in_xz) - omega * time_s + phases[i];
        height += amplitude * sin(theta);
        gradient += amplitude * k * direction * cos(theta);
        displaced_xz += choppiness * amplitude * direction * cos(theta);
    }
    world_position = vec3(displaced_xz.x, height, displaced_xz.y);
    world_normal = normalize(vec3(-gradient.x, 1.0, -gradient.y));
    gl_Position = view_projection * vec4(world_position, 1.0);
}
"""

WATER_FRAGMENT = """
#version 330
in vec3 world_position;
in vec3 world_normal;
uniform vec3 camera_position;
out vec4 frag_color;
void main() {
    vec3 n = normalize(world_normal);
    vec3 view_direction = normalize(camera_position - world_position);
    float n_dot_v = max(dot(n, view_direction), 0.0);
    float fresnel = 0.02037 + (1.0 - 0.02037) * pow(1.0 - n_dot_v, 5.0);
    vec3 reflected = reflect(-view_direction, n);
    float sky_factor = smoothstep(-0.15, 0.75, reflected.y);
    vec3 sky_radiance = mix(vec3(.0018, .0024, .0035),
                            vec3(.00018, .00032, .00072), sky_factor);
    vec3 water_body = vec3(.00010, .00028, .00034);
    float grazing_haze = pow(1.0 - n_dot_v, 3.0) * .0007;
    vec3 radiance = mix(water_body, sky_radiance, fresnel) + grazing_haze;
    frag_color = vec4(radiance, 1.0);
}
"""


class Renderer:
    """Linear-HDR renderer. Terrain, water, and atmosphere are separate passes."""

    def __init__(self, ctx: moderngl.Context, config: RenderConfig) -> None:
        self.ctx, self.config, self.time_s = ctx, config, 0.0
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.quad_buffer = ctx.buffer(quad.tobytes())
        self.background_program = ctx.program(
            vertex_shader=QUAD_VERTEX, fragment_shader=BACKGROUND_FRAGMENT
        )
        self.background_vao = ctx.simple_vertex_array(
            self.background_program, self.quad_buffer, "in_position"
        )
        self.tonemap_program = ctx.program(
            vertex_shader=QUAD_VERTEX, fragment_shader=TONEMAP_FRAGMENT
        )
        self.tonemap_vao = ctx.simple_vertex_array(
            self.tonemap_program, self.quad_buffer, "in_position"
        )
        self.tonemap_program["hdr_texture"] = 0
        self.water_config = WaterConfig()
        water_spectrum = build_directional_spectrum(self.water_config)
        water_vertices, water_indices = build_water_mesh(self.water_config)
        self.water_program = ctx.program(
            vertex_shader=WATER_VERTEX, fragment_shader=WATER_FRAGMENT
        )
        self.water_vertex_buffer = ctx.buffer(water_vertices.tobytes())
        self.water_index_buffer = ctx.buffer(water_indices.tobytes())
        self.water_vao = ctx.vertex_array(
            self.water_program,
            [(self.water_vertex_buffer, "2f", "in_xz")],
            self.water_index_buffer,
            index_element_size=4,
        )
        self.water_program["waves"].write(water_spectrum.components.tobytes())
        self.water_program["phases"].write(water_spectrum.phases.tobytes())
        self.water_program["choppiness"] = self.water_config.choppiness
        self.particle_program = ctx.program(
            vertex_shader=PARTICLE_VERTEX, fragment_shader=PARTICLE_FRAGMENT
        )
        self.stride = 7
        self.particle_buffer = ctx.buffer(
            reserve=config.max_particles * self.stride * 4, dynamic=True
        )
        self.particle_vao = ctx.vertex_array(
            self.particle_program,
            [(self.particle_buffer, "3f 3f 1f",
              "in_position", "in_color", "in_power")],
        )
        self.hdr_texture = ctx.texture(
            (config.width, config.height), components=4, dtype="f2"
        )
        depth = ctx.depth_renderbuffer((config.width, config.height))
        self.hdr_fbo = ctx.framebuffer([self.hdr_texture], depth)
        projection = _perspective(
            config.vertical_fov_deg, config.width / config.height, .1, 2500
        )
        camera_position = np.array([0, 24, 235], dtype=np.float32)
        view = _look_at(
            camera_position,
            np.array([0, 105, 0], dtype=np.float32),
        )
        view_projection = projection @ view
        self.particle_program["view_projection"].write(
            view_projection.T.astype(np.float32).tobytes()
        )
        self.water_program["view_projection"].write(
            view_projection.T.astype(np.float32).tobytes()
        )
        self.water_program["camera_position"].value = tuple(camera_position)

    def render(self, world: FireworkWorld, frame_dt_s: float) -> None:
        self.time_s += frame_dt_s
        self.hdr_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hdr_fbo.clear(0, 0, 0, 1, depth=1)
        self.background_program["time_s"] = self.time_s
        self.background_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.water_program["time_s"] = self.time_s
        self.water_vao.render(moderngl.TRIANGLES)
        count = world.stars.count
        if count:
            data = np.empty((count, self.stride), dtype=np.float32)
            data[:, :3] = world.stars.position_m[:count]
            data[:, 3:6] = world.stars.color_linear[:count]
            data[:, 6] = world.stars.intensity()
            self.particle_buffer.write(data.tobytes())
            self.ctx.enable(moderngl.BLEND)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.particle_program["time_s"] = self.time_s
            for reflection in (1.0, 0.0):
                self.particle_program["reflection"] = reflection
                self.particle_vao.render(moderngl.POINTS, vertices=count)
        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)
        self.hdr_texture.use(0)
        self.tonemap_program["exposure_scale"] = 2.0 ** (
            6.0 - self.config.exposure_ev100
        )
        self.tonemap_vao.render(moderngl.TRIANGLE_STRIP)
