from __future__ import annotations

import math
from pathlib import Path

import moderngl
import numpy as np

from .camera import FreeCamera
from .astronomy import CelestialState
from .config import AtmosphereConfig, RenderConfig, SmokeConfig
from .fluid import SmokeFluid2D
from .physics import FireworkWorld
from .scene import load_scene
from .water import (
    WaterConfig,
    build_directional_spectrum,
    build_water_mesh,
    estimate_fetch_length_m,
)


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


QUAD_VERTEX = """
#version 330
in vec2 in_position;
out vec2 uv;
void main() { uv = in_position * .5 + .5; gl_Position = vec4(in_position, 0, 1); }
"""

BACKGROUND_FRAGMENT = """
#version 330
in vec2 uv; out vec4 frag_color;
uniform vec3 camera_forward; uniform vec3 camera_right; uniform vec3 camera_up;
uniform float tan_half_fov; uniform float aspect;
uniform vec3 sun_direction; uniform vec3 moon_direction;
uniform float twilight_strength; uniform float moon_strength;
uniform float cloud_cover;
void main() {
    vec2 screen = uv * 2.0 - 1.0;
    vec3 ray = normalize(camera_forward
                       + camera_right * screen.x * aspect * tan_half_fov
                       + camera_up * screen.y * tan_half_fov);
    float altitude = asin(clamp(ray.y, -1.0, 1.0));
    float above_horizon = smoothstep(-.025, .06, ray.y);
    float zenith = smoothstep(-.02, .75, ray.y);
    vec3 night = mix(vec3(.00034, .00043, .00062),
                     vec3(.000055, .000085, .00018), zenith);
    float sun_alignment = max(dot(ray, sun_direction), 0.0);
    float western_twilight = pow(sun_alignment, 24.0) * twilight_strength;
    vec3 twilight_color = mix(vec3(.0028, .00072, .00020),
                              vec3(.00042, .00075, .0015), zenith);
    float urban_horizon = exp(-max(altitude, 0.0) * 7.5)
                        * (.00032 + cloud_cover * .00070);
    float moon_disc = smoothstep(cos(radians(.31)), cos(radians(.24)),
                                 dot(ray, moon_direction)) * moon_strength;
    vec3 sky = night
             + twilight_color * western_twilight * (1.0 - cloud_cover * .45)
             + vec3(.00045, .00036, .00024) * urban_horizon
             + vec3(.72, .80, 1.0) * moon_disc;
    vec3 below = vec3(.000025, .000035, .000055);
    frag_color = vec4(mix(below, sky, above_horizon), 1);
}
"""

PARTICLE_VERTEX = """
#version 330
in vec3 in_position; in vec3 in_trail_start; in vec3 in_color; in float in_power;
uniform mat4 view_projection; uniform float reflection; uniform float time_s;
out vec4 trail_start_clip; out vec3 particle_color; out float particle_power;
void main() {
    vec3 p = in_position;
    vec3 start = in_trail_start;
    if (reflection > .5) {
        p.y = -p.y;
        start.y = -start.y;
        p.x += sin(p.x * .12 + time_s * 2.1) * (.7 + .012 * abs(p.y));
        start.x += sin(start.x * .12 + time_s * 2.1)
                 * (.7 + .012 * abs(start.y));
    }
    gl_Position = view_projection * vec4(p, 1);
    trail_start_clip = view_projection * vec4(start, 1);
    particle_color = in_color;
    particle_power = in_power * (reflection > .5 ? .13 : 1.0);
}
"""

PARTICLE_GEOMETRY = """
#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
in vec4 trail_start_clip[];
in vec3 particle_color[];
in float particle_power[];
out vec3 trail_color;
out float trail_power;
out vec2 trail_coordinate;
uniform vec2 viewport_size;
void emit_trail_vertex(vec4 clip, vec2 offset_ndc, vec2 coordinate) {
    gl_Position = clip;
    gl_Position.xy += offset_ndc * clip.w;
    trail_color = particle_color[0];
    trail_power = particle_power[0];
    trail_coordinate = coordinate;
    EmitVertex();
}
void main() {
    vec4 start_clip = trail_start_clip[0];
    vec4 end_clip = gl_in[0].gl_Position;
    if (start_clip.w <= 0.0 || end_clip.w <= 0.0) return;
    vec2 start_ndc = start_clip.xy / start_clip.w;
    vec2 end_ndc = end_clip.xy / end_clip.w;
    vec2 delta_pixels = (end_ndc - start_ndc) * viewport_size * .5;
    float length_pixels = length(delta_pixels);
    vec2 direction = length_pixels > .1
                   ? delta_pixels / length_pixels : vec2(0.0, 1.0);
    vec2 perpendicular = vec2(-direction.y, direction.x);
    float half_width = clamp(1.15 + 900.0 / max(end_clip.w, 1.0), 1.2, 5.5);
    vec2 offset = perpendicular * half_width * 2.0 / viewport_size;
    if (length_pixels < 1.0) {
        vec2 extension = direction * (1.0 - length_pixels) * 2.0 / viewport_size;
        start_clip.xy -= extension * start_clip.w;
    }
    emit_trail_vertex(start_clip, -offset, vec2(0.0, -1.0));
    emit_trail_vertex(start_clip,  offset, vec2(0.0,  1.0));
    emit_trail_vertex(end_clip,   -offset, vec2(1.0, -1.0));
    emit_trail_vertex(end_clip,    offset, vec2(1.0,  1.0));
    EndPrimitive();
}
"""

PARTICLE_FRAGMENT = """
#version 330
in vec3 trail_color; in float trail_power; in vec2 trail_coordinate;
out vec4 frag_color;
void main() {
    float cross_section = exp(-trail_coordinate.y * trail_coordinate.y * 3.4);
    float longitudinal = mix(.62, 1.0, trail_coordinate.x);
    float radiance = cross_section * longitudinal * trail_power * .0045;
    frag_color = vec4(trail_color * radiance, 1);
}
"""

TONEMAP_FRAGMENT = """
#version 330
uniform sampler2D hdr_texture; uniform sampler2D bloom_texture;
uniform float exposure_scale; uniform float bloom_strength;
in vec2 uv; out vec4 frag_color;
vec3 aces(vec3 x) {
    return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14), 0.0, 1.0);
}
void main() {
    vec3 hdr = texture(hdr_texture, uv).rgb;
    vec3 bloom = texture(bloom_texture, uv).rgb * bloom_strength;
    vec3 mapped = aces((hdr + bloom) * exposure_scale);
    frag_color = vec4(pow(mapped, vec3(1.0/2.2)), 1);
}
"""

BLOOM_PREFILTER_FRAGMENT = """
#version 330
uniform sampler2D hdr_texture;
in vec2 uv; out vec4 frag_color;
void main() {
    vec3 color = texture(hdr_texture, uv).rgb;
    float brightness = max(max(color.r, color.g), color.b);
    float soft = clamp((brightness - .035) / .10, 0.0, 1.0);
    frag_color = vec4(color * soft, 1.0);
}
"""

BLOOM_BLUR_FRAGMENT = """
#version 330
uniform sampler2D source_texture; uniform vec2 direction;
in vec2 uv; out vec4 frag_color;
void main() {
    vec2 texel = 1.0 / vec2(textureSize(source_texture, 0));
    vec3 color = texture(source_texture, uv).rgb * .227027;
    color += texture(source_texture, uv + direction * texel * 1.384615).rgb * .316216;
    color += texture(source_texture, uv - direction * texel * 1.384615).rgb * .316216;
    color += texture(source_texture, uv + direction * texel * 3.230769).rgb * .070270;
    color += texture(source_texture, uv - direction * texel * 3.230769).rgb * .070270;
    frag_color = vec4(color, 1.0);
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
out vec4 reflection_clip;
uniform mat4 reflection_view_projection;
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
        float wavelength = 6.28318530718 / k;
        float geometry_weight = smoothstep(1.8, 7.0, wavelength);
        height += amplitude * sin(theta) * geometry_weight;
        gradient += amplitude * k * direction * cos(theta) * geometry_weight;
        displaced_xz += choppiness * amplitude * direction * cos(theta)
                      * geometry_weight;
    }
    world_position = vec3(displaced_xz.x, height, displaced_xz.y);
    world_normal = normalize(vec3(-gradient.x, 1.0, -gradient.y));
    reflection_clip = reflection_view_projection
                    * vec4(world_position, 1.0);
    gl_Position = view_projection * vec4(world_position, 1.0);
}
"""

WATER_FRAGMENT = """
#version 330
in vec3 world_position;
in vec3 world_normal;
in vec4 reflection_clip;
uniform vec3 camera_position;
uniform sampler2D water_mask;
uniform sampler2D reflection_texture;
uniform vec4 water_mask_bounds;
uniform float sky_ambient_scale;
uniform vec4 waves[32];
uniform float phases[32];
uniform float time_s;
out vec4 frag_color;
void main() {
    vec2 mask_uv = (world_position.xz - water_mask_bounds.xy)
                 / (water_mask_bounds.zw - water_mask_bounds.xy);
    if (any(lessThan(mask_uv, vec2(0.0)))
        || any(greaterThan(mask_uv, vec2(1.0)))
        || texture(water_mask, mask_uv).r < 0.5) discard;
    vec2 fine_gradient = vec2(0.0);
    for (int i = 0; i < 32; ++i) {
        vec2 direction = waves[i].xy;
        float k = waves[i].z;
        float amplitude = waves[i].w;
        float omega = sqrt(9.80665 * k);
        float theta = k * dot(direction, world_position.xz)
                    - omega * time_s + phases[i];
        float wavelength = 6.28318530718 / k;
        float fine_weight = 1.0 - smoothstep(1.8, 7.0, wavelength);
        fine_gradient += amplitude * k * direction * cos(theta) * fine_weight;
    }
    vec3 n = normalize(
        world_normal + vec3(-fine_gradient.x, 0.0, -fine_gradient.y)
    );
    vec3 view_direction = normalize(camera_position - world_position);
    float n_dot_v = max(dot(n, view_direction), 0.0);
    float fresnel = 0.02037 + (1.0 - 0.02037) * pow(1.0 - n_dot_v, 5.0);
    vec3 reflected_direction = reflect(-view_direction, n);
    float sky_factor = smoothstep(-0.15, 0.75, reflected_direction.y);
    vec3 fallback_radiance = mix(vec3(.0018, .0024, .0035),
                                 vec3(.00018, .00032, .00072), sky_factor)
                           * sky_ambient_scale;
    vec2 reflection_uv = reflection_clip.xy / reflection_clip.w * .5 + .5;
    reflection_uv += vec2(n.x, -n.z) * .018;
    bool reflection_valid = reflection_clip.w > 0.0
                         && all(greaterThanEqual(reflection_uv, vec2(0.0)))
                         && all(lessThanEqual(reflection_uv, vec2(1.0)));
    vec3 reflected_radiance = reflection_valid
        ? texture(reflection_texture, reflection_uv).rgb
        : fallback_radiance;
    vec3 water_body = vec3(.00010, .00028, .00034);
    float grazing_haze = pow(1.0 - n_dot_v, 3.0) * .0007;
    vec3 radiance = mix(water_body, reflected_radiance, fresnel)
                  + grazing_haze;
    frag_color = vec4(radiance, 1.0);
}
"""

LAND_VERTEX = """
#version 330
in vec2 in_xz;
uniform mat4 view_projection;
uniform sampler2D terrain_height;
uniform vec4 terrain_bounds;
out vec3 world_position;
out vec3 world_normal;
void main() {
    vec2 uv = (in_xz - terrain_bounds.xy) / (terrain_bounds.zw - terrain_bounds.xy);
    ivec2 dimensions = textureSize(terrain_height, 0);
    vec2 metres_per_texel = (terrain_bounds.zw - terrain_bounds.xy)
                          / vec2(dimensions);
    float height = texture(terrain_height, uv).r;
    float left = textureOffset(terrain_height, uv, ivec2(-1, 0)).r;
    float right = textureOffset(terrain_height, uv, ivec2(1, 0)).r;
    float back = textureOffset(terrain_height, uv, ivec2(0, -1)).r;
    float front = textureOffset(terrain_height, uv, ivec2(0, 1)).r;
    world_normal = normalize(vec3(
        -(right - left) / (2.0 * metres_per_texel.x),
        1.0,
        -(front - back) / (2.0 * metres_per_texel.y)
    ));
    world_position = vec3(in_xz.x, height - 0.04, in_xz.y);
    gl_Position = view_projection * vec4(world_position, 1.0);
}
"""

LAND_FRAGMENT = """
#version 330
in vec3 world_position;
in vec3 world_normal;
uniform sampler2D water_mask;
uniform vec4 water_mask_bounds;
uniform float sky_ambient_scale;
out vec4 frag_color;
void main() {
    vec2 uv = (world_position.xz - water_mask_bounds.xy)
            / (water_mask_bounds.zw - water_mask_bounds.xy);
    if (all(greaterThanEqual(uv, vec2(0.0)))
        && all(lessThanEqual(uv, vec2(1.0)))
        && texture(water_mask, uv).r >= 0.5) discard;
    float variation = sin(world_position.x * .07) * sin(world_position.z * .051);
    float sky_light = max(world_normal.y, 0.15);
    vec3 ground = (vec3(.00032, .00042, .00030) + variation * .000035)
                * sky_light * sky_ambient_scale;
    frag_color = vec4(ground, 1.0);
}
"""

SCENE_VERTEX = """
#version 330
in vec3 in_position; in vec3 in_normal; in float in_material;
uniform mat4 view_projection;
uniform sampler2D terrain_height;
uniform vec4 terrain_bounds;
out vec3 world_position; out vec3 world_normal; out float material;
void main() {
    vec2 terrain_uv = (in_position.xz - terrain_bounds.xy)
                    / (terrain_bounds.zw - terrain_bounds.xy);
    float base_height = texture(terrain_height, terrain_uv).r;
    world_position = in_position + vec3(0.0, base_height, 0.0);
    world_normal = in_normal;
    material = in_material;
    gl_Position = view_projection * vec4(world_position, 1.0);
}
"""

SCENE_FRAGMENT = """
#version 330
in vec3 world_position; in vec3 world_normal; in float material;
out vec4 frag_color;
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
void main() {
    vec3 n = normalize(world_normal);
    float sky_light = max(n.y * .5 + .5, .08);
    vec3 concrete = vec3(.0015, .0017, .0020) * sky_light;
    if (material > 3.5) {
        float green_variation = hash21(floor(world_position.xz * .15));
        vec3 green = mix(vec3(.00016, .00034, .00016),
                         vec3(.00032, .00052, .00022), green_variation);
        frag_color = vec4(green * sky_light, 1.0);
        return;
    }
    if (material > 2.5) {
        float lane_hint = smoothstep(.46, .50,
            abs(fract(world_position.x * .12 + world_position.z * .08) - .5));
        vec3 asphalt = vec3(.00030, .00032, .00035);
        frag_color = vec4(asphalt + lane_hint * vec3(.00016), 1.0);
        return;
    }
    if (material > 1.5) {
        frag_color = vec4(vec3(.0024, .0026, .0027), 1.0);
        return;
    }
    if (material > .5) {
        frag_color = vec4(concrete * .7, 1.0);
        return;
    }
    float facade_axis = dot(world_position.xz, abs(n.zx));
    vec2 cell = floor(vec2(facade_axis / 4.2, world_position.y / 3.25));
    vec2 within = fract(vec2(facade_axis / 4.2, world_position.y / 3.25));
    float pane = step(.13, within.x) * step(within.x, .82)
               * step(.18, within.y) * step(within.y, .72);
    float occupied = step(.42, hash21(cell));
    float temperature = hash21(cell + 17.0);
    vec3 window_color = mix(vec3(1.0, .42, .12), vec3(.55, .72, 1.0), temperature);
    vec3 emission = window_color * pane * occupied * .035;
    frag_color = vec4(concrete + emission, 1.0);
}
"""

SMOKE_VERTEX = """
#version 330
in vec3 in_position;
uniform mat4 view_projection;
out vec3 surface_position;
void main() {
    surface_position = in_position;
    gl_Position = view_projection * vec4(in_position, 1.0);
}
"""

SMOKE_FRAGMENT = """
#version 330
in vec3 surface_position; out vec4 frag_color;
uniform sampler3D smoke_density;
uniform sampler3D temperature_excess;
uniform vec3 camera_position;
uniform vec3 volume_min;
uniform vec3 volume_max;
uniform int camera_inside;
uniform int ray_steps;
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
void main() {
    if ((camera_inside == 0 && !gl_FrontFacing)
        || (camera_inside == 1 && gl_FrontFacing)) discard;
    vec3 ray = normalize(surface_position - camera_position);
    vec3 safe_ray = mix(ray, vec3(1e-6),
                        lessThan(abs(ray), vec3(1e-6)));
    vec3 t0 = (volume_min - camera_position) / safe_ray;
    vec3 t1 = (volume_max - camera_position) / safe_ray;
    vec3 near_axis = min(t0, t1);
    vec3 far_axis = max(t0, t1);
    float t_near = max(max(near_axis.x, near_axis.y), near_axis.z);
    float t_far = min(min(far_axis.x, far_axis.y), far_axis.z);
    t_near = max(t_near, 0.0);
    if (t_far <= t_near) discard;

    float step_m = (t_far - t_near) / float(ray_steps);
    float jitter = hash12(gl_FragCoord.xy);
    float transmittance = 1.0;
    vec3 radiance = vec3(0.0);
    for (int i = 0; i < 64; ++i) {
        if (i >= ray_steps) break;
        float distance_m = t_near + (float(i) + jitter) * step_m;
        vec3 world = camera_position + ray * distance_m;
        vec3 uvw = (world - volume_min) / (volume_max - volume_min);
        float density = max(texture(smoke_density, uvw).r, 0.0);
        float temperature = max(texture(temperature_excess, uvw).r, 0.0);
        float step_alpha = 1.0 - exp(-density * 4500.0 * step_m);
        float warm = clamp(temperature / 850.0, 0.0, 1.0);
        vec3 scattered = mix(vec3(.0010, .00115, .00135),
                             vec3(.018, .0062, .0014), warm);
        radiance += transmittance * scattered * step_alpha;
        transmittance *= 1.0 - step_alpha;
        if (transmittance < .01) break;
    }
    float alpha = 1.0 - transmittance;
    if (alpha < .001) discard;
    frag_color = vec4(radiance, alpha);
}
"""


class Renderer:
    """Linear-HDR renderer. Terrain, water, and atmosphere are separate passes."""

    def __init__(
        self,
        ctx: moderngl.Context,
        config: RenderConfig,
        atmosphere: AtmosphereConfig | None = None,
        smoke_config: SmokeConfig | None = None,
    ) -> None:
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
        self.background_program["tan_half_fov"] = math.tan(
            math.radians(config.vertical_fov_deg) * 0.5
        )
        self.background_program["aspect"] = config.width / config.height
        self.tonemap_program = ctx.program(
            vertex_shader=QUAD_VERTEX, fragment_shader=TONEMAP_FRAGMENT
        )
        self.tonemap_vao = ctx.simple_vertex_array(
            self.tonemap_program, self.quad_buffer, "in_position"
        )
        self.tonemap_program["hdr_texture"] = 0
        self.tonemap_program["bloom_texture"] = 3
        self.tonemap_program["bloom_strength"] = config.bloom_strength
        self.bloom_prefilter_program = ctx.program(
            vertex_shader=QUAD_VERTEX, fragment_shader=BLOOM_PREFILTER_FRAGMENT
        )
        self.bloom_prefilter_vao = ctx.simple_vertex_array(
            self.bloom_prefilter_program, self.quad_buffer, "in_position"
        )
        self.bloom_prefilter_program["hdr_texture"] = 0
        self.bloom_blur_program = ctx.program(
            vertex_shader=QUAD_VERTEX, fragment_shader=BLOOM_BLUR_FRAGMENT
        )
        self.bloom_blur_vao = ctx.simple_vertex_array(
            self.bloom_blur_program, self.quad_buffer, "in_position"
        )
        self.bloom_blur_program["source_texture"] = 3
        self.scene_program = ctx.program(
            vertex_shader=SCENE_VERTEX, fragment_shader=SCENE_FRAGMENT
        )
        scene_path = Path(__file__).resolve().parent.parent / "assets" / "yeouido_scene.npz"
        self.scene_vaos: list[tuple[moderngl.VertexArray, int]] = []
        self.reflection_scene_vaos: list[
            tuple[moderngl.VertexArray, int]
        ] = []
        water_mask = np.full((1, 1), 255, dtype=np.uint8)
        water_mask_bounds = np.array(
            [-10_000.0, -10_000.0, 10_000.0, 10_000.0], dtype=np.float32
        )
        terrain_height = np.zeros((1, 1), dtype=np.float32)
        terrain_bounds = water_mask_bounds.copy()
        if scene_path.exists():
            scene = load_scene(scene_path)
            water_mask = scene.water_mask
            water_mask_bounds = scene.water_mask_bounds
            terrain_height = scene.terrain_height_m
            terrain_bounds = scene.terrain_bounds
            for scene_index, vertices in enumerate((
                scene.building_vertices,
                scene.bridge_vertices,
                scene.road_vertices,
                scene.vegetation_vertices,
            )):
                if not len(vertices):
                    continue
                buffer = ctx.buffer(vertices.tobytes())
                vao = ctx.vertex_array(
                    self.scene_program,
                    [(buffer, "3f 3f 1f",
                      "in_position", "in_normal", "in_material")],
                )
                self.scene_vaos.append((vao, len(vertices)))
                if scene_index < 2:
                    self.reflection_scene_vaos.append((vao, len(vertices)))
        atmosphere = atmosphere or AtmosphereConfig()
        wind = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float32)
        wind_speed = float(np.linalg.norm(wind[[0, 2]]))
        wind_from = math.degrees(math.atan2(-float(wind[0]), float(wind[2]))) % 360.0
        fetch_length_m = estimate_fetch_length_m(
            water_mask, water_mask_bounds, wind[[0, 2]]
        )
        self.water_config = WaterConfig(
            wind_speed_mps=max(wind_speed, 0.1),
            wind_direction_deg=wind_from,
            fetch_length_m=fetch_length_m,
        )
        water_spectrum = build_directional_spectrum(self.water_config)
        near_vertices, near_indices = build_water_mesh(self.water_config)
        far_vertices, far_indices = build_water_mesh(
            self.water_config,
            self.water_config.far_grid_size,
            self.water_config.far_extent_m,
            self.water_config.extent_m,
        )
        land_vertices, land_indices = build_water_mesh(
            self.water_config,
            self.water_config.far_grid_size,
            self.water_config.far_extent_m,
        )
        self.water_program = ctx.program(
            vertex_shader=WATER_VERTEX, fragment_shader=WATER_FRAGMENT
        )
        self.water_vaos: list[moderngl.VertexArray] = []
        self.water_buffers: list[moderngl.Buffer] = []
        for vertices, indices in (
            (far_vertices, far_indices),
            (near_vertices, near_indices),
        ):
            vertex_buffer = ctx.buffer(vertices.tobytes())
            index_buffer = ctx.buffer(indices.tobytes())
            self.water_buffers.extend((vertex_buffer, index_buffer))
            self.water_vaos.append(
                ctx.vertex_array(
                    self.water_program,
                    [(vertex_buffer, "2f", "in_xz")],
                    index_buffer,
                    index_element_size=4,
                )
            )
        self.land_program = ctx.program(
            vertex_shader=LAND_VERTEX, fragment_shader=LAND_FRAGMENT
        )
        self.land_vertex_buffer = ctx.buffer(land_vertices.tobytes())
        self.land_index_buffer = ctx.buffer(land_indices.tobytes())
        self.land_vao = ctx.vertex_array(
            self.land_program,
            [(self.land_vertex_buffer, "2f", "in_xz")],
            self.land_index_buffer,
            index_element_size=4,
        )
        self.water_program["waves"].write(water_spectrum.components.tobytes())
        self.water_program["phases"].write(water_spectrum.phases.tobytes())
        self.water_program["choppiness"] = self.water_config.choppiness
        self.significant_wave_height_m = water_spectrum.significant_wave_height_m
        self.water_program["reflection_texture"] = 7
        self.water_mask_texture = ctx.texture(
            (water_mask.shape[1], water_mask.shape[0]),
            components=1,
            data=np.ascontiguousarray(water_mask).tobytes(),
            dtype="f1",
        )
        self.water_mask_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.water_mask_texture.repeat_x = False
        self.water_mask_texture.repeat_y = False
        self.terrain_texture = ctx.texture(
            (terrain_height.shape[1], terrain_height.shape[0]),
            components=1,
            data=np.ascontiguousarray(terrain_height).tobytes(),
            dtype="f4",
        )
        self.terrain_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.terrain_texture.repeat_x = False
        self.terrain_texture.repeat_y = False
        self.water_program["water_mask"] = 1
        self.water_program["water_mask_bounds"].value = tuple(water_mask_bounds)
        self.land_program["water_mask"] = 1
        self.land_program["water_mask_bounds"].value = tuple(water_mask_bounds)
        self.land_program["terrain_height"] = 2
        self.land_program["terrain_bounds"].value = tuple(terrain_bounds)
        self.scene_program["terrain_height"] = 2
        self.scene_program["terrain_bounds"].value = tuple(terrain_bounds)
        self.particle_program = ctx.program(
            vertex_shader=PARTICLE_VERTEX,
            geometry_shader=PARTICLE_GEOMETRY,
            fragment_shader=PARTICLE_FRAGMENT,
        )
        self.particle_program["viewport_size"].value = (
            float(config.width),
            float(config.height),
        )
        self.stride = 10
        self.particle_buffer = ctx.buffer(
            reserve=config.max_particles * self.stride * 4, dynamic=True
        )
        self.particle_vao = ctx.vertex_array(
            self.particle_program,
            [
                (
                    self.particle_buffer,
                    "3f 3f 3f 1f",
                    "in_position",
                    "in_trail_start",
                    "in_color",
                    "in_power",
                )
            ],
        )
        smoke_config = smoke_config or SmokeConfig()
        sx0, sx1, sy0, sy1 = smoke_config.bounds_m
        sz0, sz1 = (
            -0.5 * smoke_config.volume_depth_m,
            0.5 * smoke_config.volume_depth_m,
        )
        smoke_vertices = np.array(
            [
                sx0, sy0, sz0, sx1, sy0, sz0,
                sx1, sy1, sz0, sx0, sy1, sz0,
                sx0, sy0, sz1, sx1, sy0, sz1,
                sx1, sy1, sz1, sx0, sy1, sz1,
            ],
            dtype=np.float32,
        ).reshape(-1, 3)
        smoke_indices = np.array(
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
        self.smoke_program = ctx.program(
            vertex_shader=SMOKE_VERTEX, fragment_shader=SMOKE_FRAGMENT
        )
        self.smoke_buffer = ctx.buffer(smoke_vertices.tobytes())
        self.smoke_index_buffer = ctx.buffer(smoke_indices.tobytes())
        self.smoke_vao = ctx.vertex_array(
            self.smoke_program,
            [(self.smoke_buffer, "3f", "in_position")],
            self.smoke_index_buffer,
            index_element_size=4,
        )
        smoke_size = (
            smoke_config.grid_size[0],
            smoke_config.grid_size[1],
            smoke_config.volume_slices,
        )
        self.smoke_density_texture = ctx.texture3d(
            smoke_size, components=1, dtype="f4"
        )
        self.smoke_temperature_texture = ctx.texture3d(
            smoke_size, components=1, dtype="f4"
        )
        for texture in (
            self.smoke_density_texture, self.smoke_temperature_texture
        ):
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.repeat_x = False
            texture.repeat_y = False
            texture.repeat_z = False
        self.smoke_program["smoke_density"] = 4
        self.smoke_program["temperature_excess"] = 5
        self.smoke_program["volume_min"].value = (sx0, sy0, sz0)
        self.smoke_program["volume_max"].value = (sx1, sy1, sz1)
        self.smoke_program["ray_steps"] = smoke_config.volume_ray_steps
        self.smoke_bounds = np.array(
            [[sx0, sy0, sz0], [sx1, sy1, sz1]], dtype=np.float32
        )
        self.smoke_revision = -1
        self.hdr_texture = ctx.texture(
            (config.width, config.height), components=4, dtype="f2"
        )
        depth = ctx.depth_renderbuffer((config.width, config.height))
        self.hdr_fbo = ctx.framebuffer([self.hdr_texture], depth)
        reflection_size = (
            max(int(config.width * config.reflection_scale), 1),
            max(int(config.height * config.reflection_scale), 1),
        )
        self.reflection_texture = ctx.texture(
            reflection_size, components=4, dtype="f2"
        )
        self.reflection_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.reflection_texture.repeat_x = False
        self.reflection_texture.repeat_y = False
        reflection_depth = ctx.depth_renderbuffer(reflection_size)
        self.reflection_fbo = ctx.framebuffer(
            [self.reflection_texture], reflection_depth
        )
        self.reflection_interval_s = 1.0 / max(config.reflection_hz, 1)
        self.reflection_accumulator_s = self.reflection_interval_s
        self.reflection_ready = False
        self.last_rendered_smoke_revision = -1
        bloom_size = (max(config.width // 2, 1), max(config.height // 2, 1))
        self.bloom_textures = [
            ctx.texture(bloom_size, components=4, dtype="f2") for _ in range(2)
        ]
        for texture in self.bloom_textures:
            texture.filter = moderngl.LINEAR, moderngl.LINEAR
            texture.repeat_x = False
            texture.repeat_y = False
        self.bloom_fbos = [
            ctx.framebuffer(color_attachments=[texture])
            for texture in self.bloom_textures
        ]
        projection = _perspective(
            config.vertical_fov_deg, config.width / config.height, .1, 2500
        )
        self.projection = projection

    def _update_camera(self, camera: FreeCamera) -> None:
        view = _look_at(camera.position_m, camera.position_m + camera.forward)
        view_projection = self.projection @ view
        matrix_bytes = view_projection.T.astype(np.float32).tobytes()
        for program in (
            self.particle_program,
            self.water_program,
            self.land_program,
            self.scene_program,
            self.smoke_program,
        ):
            program["view_projection"].write(matrix_bytes)
        self.water_program["camera_position"].value = tuple(camera.position_m)
        self.smoke_program["camera_position"].value = tuple(camera.position_m)
        self.smoke_program["camera_inside"] = int(
            np.all(camera.position_m >= self.smoke_bounds[0])
            and np.all(camera.position_m <= self.smoke_bounds[1])
        )
        self._set_background_camera(camera.forward, camera.right)

    def _set_background_camera(
        self, forward: np.ndarray, right: np.ndarray
    ) -> None:
        camera_up = np.cross(right, forward)
        self.background_program["camera_forward"].value = tuple(forward)
        self.background_program["camera_right"].value = tuple(right)
        self.background_program["camera_up"].value = tuple(camera_up)

    def _render_reflection(self, camera: FreeCamera) -> None:
        reflected_position = camera.position_m.copy()
        reflected_position[1] *= -1.0
        reflected_forward = camera.forward.copy()
        reflected_forward[1] *= -1.0
        reflection_view = _look_at(
            reflected_position,
            reflected_position + reflected_forward,
        )
        reflection_view_projection = self.projection @ reflection_view
        matrix_bytes = (
            reflection_view_projection.T.astype(np.float32).tobytes()
        )
        self.land_program["view_projection"].write(matrix_bytes)
        self.scene_program["view_projection"].write(matrix_bytes)
        self.water_program["reflection_view_projection"].write(matrix_bytes)
        self._set_background_camera(reflected_forward, camera.right)

        self.reflection_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.reflection_fbo.clear(0, 0, 0, 1, depth=1)
        self.background_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.water_mask_texture.use(1)
        self.terrain_texture.use(2)
        self.land_vao.render(moderngl.TRIANGLES)
        for vao, vertex_count in self.reflection_scene_vaos:
            vao.render(moderngl.TRIANGLES, vertices=vertex_count)

    def _update_celestial(
        self, celestial: CelestialState, atmosphere: AtmosphereConfig
    ) -> None:
        twilight_floor = math.log10(0.0002)
        twilight_ceiling = math.log10(3.4)
        twilight_strength = np.clip(
            (math.log10(max(celestial.twilight_illuminance_lux, 0.0002))
             - twilight_floor)
            / (twilight_ceiling - twilight_floor),
            0.0,
            1.0,
        )
        cloud = atmosphere.cloud_cover_fraction
        self.background_program["sun_direction"].value = tuple(
            celestial.sun_direction_eus
        )
        self.background_program["moon_direction"].value = tuple(
            celestial.moon_direction_eus
        )
        self.background_program["twilight_strength"] = float(twilight_strength)
        self.background_program["moon_strength"] = float(
            min(celestial.moon_illuminance_lux / 0.25, 1.0) * 0.04
        )
        self.background_program["cloud_cover"] = cloud
        ambient_scale = 0.8 + float(twilight_strength) * 1.2 + cloud * 0.25
        self.water_program["sky_ambient_scale"] = ambient_scale
        self.land_program["sky_ambient_scale"] = ambient_scale

    def render(
        self,
        world: FireworkWorld,
        camera: FreeCamera,
        celestial: CelestialState,
        frame_dt_s: float,
        smoke: SmokeFluid2D | None = None,
    ) -> None:
        self.time_s += frame_dt_s
        self._update_celestial(celestial, world.atmosphere)
        self.reflection_accumulator_s += frame_dt_s
        smoke_revision = smoke.revision if smoke is not None else -1
        fluid_updated = smoke_revision != self.last_rendered_smoke_revision
        if (
            not self.reflection_ready
            or (
                self.reflection_accumulator_s >= self.reflection_interval_s
                and not fluid_updated
            )
        ):
            self._render_reflection(camera)
            self.reflection_accumulator_s %= self.reflection_interval_s
            self.reflection_ready = True
        self.last_rendered_smoke_revision = smoke_revision
        self._update_camera(camera)
        self.hdr_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hdr_fbo.clear(0, 0, 0, 1, depth=1)
        self.background_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.water_mask_texture.use(1)
        self.terrain_texture.use(2)
        self.land_vao.render(moderngl.TRIANGLES)
        for vao, vertex_count in self.scene_vaos:
            vao.render(moderngl.TRIANGLES, vertices=vertex_count)
        self.water_program["time_s"] = self.time_s
        self.reflection_texture.use(7)
        for water_vao in self.water_vaos:
            water_vao.render(moderngl.TRIANGLES)
        count = world.stars.count
        if count:
            data = np.empty((count, self.stride), dtype=np.float32)
            data[:, :3] = world.stars.position_m[:count]
            data[:, 3:6] = (
                world.stars.position_m[:count]
                - world.stars.velocity_mps[:count] * self.config.shutter_time_s
            )
            data[:, 6:9] = world.stars.color_linear[:count]
            data[:, 9] = world.stars.intensity()
            self.particle_buffer.write(data.tobytes())
            self.ctx.enable(moderngl.BLEND)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.particle_program["time_s"] = self.time_s
            for reflection in (1.0, 0.0):
                self.particle_program["reflection"] = reflection
                self.particle_vao.render(moderngl.POINTS, vertices=count)
        if smoke is not None and np.any(smoke.density_kg_m3 > 1e-8):
            if smoke.revision != self.smoke_revision:
                density_volume, temperature_volume = (
                    smoke.reconstruct_volume()
                )
                self.smoke_density_texture.write(
                    np.ascontiguousarray(density_volume).tobytes()
                )
                self.smoke_temperature_texture.write(
                    np.ascontiguousarray(temperature_volume).tobytes()
                )
                self.smoke_revision = smoke.revision
            self.smoke_density_texture.use(4)
            self.smoke_temperature_texture.use(5)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.depth_mask = False
            self.ctx.blend_func = (
                moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            )
            self.smoke_vao.render(moderngl.TRIANGLES)
            self.ctx.depth_mask = True
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.bloom_fbos[0].use()
        self.hdr_texture.use(0)
        self.bloom_prefilter_vao.render(moderngl.TRIANGLE_STRIP)
        self.bloom_fbos[1].use()
        self.bloom_textures[0].use(3)
        self.bloom_blur_program["direction"].value = (1.0, 0.0)
        self.bloom_blur_vao.render(moderngl.TRIANGLE_STRIP)
        self.bloom_fbos[0].use()
        self.bloom_textures[1].use(3)
        self.bloom_blur_program["direction"].value = (0.0, 1.0)
        self.bloom_blur_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.screen.use()
        self.ctx.disable(moderngl.BLEND)
        self.hdr_texture.use(0)
        self.bloom_textures[0].use(3)
        self.tonemap_program["exposure_scale"] = 2.0 ** (
            10.0 - self.config.exposure_ev100
        )
        self.tonemap_vao.render(moderngl.TRIANGLE_STRIP)
