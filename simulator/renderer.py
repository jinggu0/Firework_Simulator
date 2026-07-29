from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import moderngl
import numpy as np

from .camera import FreeCamera
from .astronomy import CelestialState
from .camera_optics import analog_gain, photon_to_electron_scale, vertical_fov_deg
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
    led_energy_budget,
    radiometric_irradiance_from_illuminance,
)
from .physics import FireworkWorld
from .scene import load_scene
from .volume import active_slice_bounds, box_vertices
from .water import (
    WaterConfig,
    build_directional_spectrum,
    build_water_mesh,
    estimate_fetch_length_m,
    relax_wave_spectrum,
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
uniform float bloom_strength;
uniform vec3 photon_to_electron;
uniform float analog_gain;
uniform float full_well_electrons;
uniform float read_noise_electrons;
uniform float tan_half_fov;
uniform float aspect;
uniform float frame_index;
uniform int sensor_noise_enabled;
in vec2 uv; out vec4 frag_color;
vec3 aces(vec3 x) {
    return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14), 0.0, 1.0);
}
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float gaussian(vec2 seed) {
    float u1 = max(hash12(seed), 1e-6);
    float u2 = hash12(seed + vec2(17.17, 91.73));
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
void main() {
    vec3 hdr = texture(hdr_texture, uv).rgb;
    vec3 bloom = texture(bloom_texture, uv).rgb * bloom_strength;
    vec2 sensor_position = (uv * 2.0 - 1.0)
                         * vec2(aspect * tan_half_fov, tan_half_fov);
    float cos_theta = inversesqrt(1.0 + dot(sensor_position, sensor_position));
    float lens_vignetting = pow(cos_theta, 4.0);
    vec3 electrons = max(hdr + bloom, vec3(0.0))
                   * photon_to_electron * lens_vignetting;
    if (sensor_noise_enabled != 0) {
        vec2 seed = gl_FragCoord.xy + vec2(frame_index * 13.37);
        vec3 sigma = sqrt(electrons
                         + vec3(read_noise_electrons * read_noise_electrons));
        electrons += sigma * vec3(
            gaussian(seed),
            gaussian(seed + vec2(31.1, 7.9)),
            gaussian(seed + vec2(83.7, 43.2))
        );
    }
    electrons = clamp(electrons, vec3(0.0), vec3(full_well_electrons));
    vec3 normalized_signal = electrons / full_well_electrons * analog_gain;
    vec3 mapped = aces(normalized_signal);
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
uniform float air_extinction_per_m;
uniform int dynamic_light_count;
uniform vec3 dynamic_light_position[8];
uniform vec3 dynamic_light_color[8];
uniform float dynamic_light_power_w[8];
out vec4 frag_color;
const float PI = 3.14159265359;
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
    // Fixed-cost GGX reflection from energy-conserving firework clusters.
    float roughness = 0.11;
    float alpha = roughness * roughness;
    float alpha_squared = alpha * alpha;
    for (int i = 0; i < 8; ++i) {
        if (i >= dynamic_light_count) break;
        vec3 displacement = dynamic_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), 1.0);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float n_dot_l = max(dot(n, light_direction), 0.0);
        if (n_dot_l <= 0.0) continue;
        vec3 half_vector = normalize(view_direction + light_direction);
        float n_dot_h = max(dot(n, half_vector), 0.0);
        float v_dot_h = max(dot(view_direction, half_vector), 0.0);
        float denominator = n_dot_h * n_dot_h
                          * (alpha_squared - 1.0) + 1.0;
        float distribution = alpha_squared
                           / max(PI * denominator * denominator, 1e-5);
        float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        float geometry_v = n_dot_v / mix(n_dot_v, 1.0, k);
        float geometry_l = n_dot_l / mix(n_dot_l, 1.0, k);
        vec3 specular_fresnel = vec3(0.02037)
            + (vec3(1.0) - vec3(0.02037)) * pow(1.0 - v_dot_h, 5.0);
        vec3 brdf = specular_fresnel * distribution
                  * geometry_v * geometry_l
                  / max(4.0 * n_dot_v * n_dot_l, 1e-5);
        float irradiance = dynamic_light_power_w[i]
                         * exp(-air_extinction_per_m * distance_m)
                         / (4.0 * PI * distance_squared);
        radiance += dynamic_light_color[i] * irradiance * brdf * n_dot_l;
    }
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
in vec3 in_position;
in vec3 in_normal;
in float in_surface;
in vec2 in_surface_uv;
in float in_facade_style;
uniform mat4 view_projection;
uniform sampler2D terrain_height;
uniform vec4 terrain_bounds;
out vec3 world_position;
out vec3 world_normal;
out float surface;
out vec2 surface_uv;
out float facade_style;
void main() {
    vec2 terrain_uv = (in_position.xz - terrain_bounds.xy)
                    / (terrain_bounds.zw - terrain_bounds.xy);
    float base_height = texture(terrain_height, terrain_uv).r;
    world_position = in_position + vec3(0.0, base_height, 0.0);
    world_normal = in_normal;
    surface = in_surface;
    surface_uv = in_surface_uv;
    facade_style = in_facade_style;
    gl_Position = view_projection * vec4(world_position, 1.0);
}
"""

SCENE_FRAGMENT = """
#version 330
in vec3 world_position;
in vec3 world_normal;
in float surface;
in vec2 surface_uv;
in float facade_style;
uniform vec3 camera_position;
uniform float ambient_irradiance_w_m2;
uniform float window_radiance_w_m2_sr;
uniform float air_extinction_per_m;
uniform int dynamic_light_count;
uniform vec3 dynamic_light_position[8];
uniform vec3 dynamic_light_color[8];
uniform float dynamic_light_power_w[8];
out vec4 frag_color;
const float PI = 3.14159265359;
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float interval(float value, float lower, float upper, float antialias) {
    return smoothstep(lower - antialias, lower + antialias, value)
         * (1.0 - smoothstep(upper - antialias, upper + antialias, value));
}
vec3 reflected_radiance(vec3 n, vec3 albedo) {
    vec3 result = albedo * ambient_irradiance_w_m2 / PI;
    for (int i = 0; i < 8; ++i) {
        if (i >= dynamic_light_count) break;
        vec3 displacement = dynamic_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), 1.0);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float irradiance = dynamic_light_power_w[i]
                         * exp(-air_extinction_per_m * distance_m)
                         / (4.0 * PI * distance_squared);
        result += albedo * dynamic_light_color[i] * irradiance
                * max(dot(n, light_direction), 0.0) / PI;
    }
    return result;
}
void main() {
    vec3 n = normalize(world_normal);
    if (surface > 3.5) {
        float green_variation = hash21(floor(world_position.xz * .15));
        vec3 green = mix(vec3(.055, .16, .045),
                         vec3(.12, .25, .075), green_variation);
        frag_color = vec4(reflected_radiance(n, green), 1.0);
        return;
    }
    if (surface > 2.5) {
        float lane_hint = smoothstep(.46, .50,
            abs(fract(world_position.x * .12 + world_position.z * .08) - .5));
        vec3 asphalt = mix(
            vec3(.035, .038, .042), vec3(.32), lane_hint * .22
        );
        frag_color = vec4(reflected_radiance(n, asphalt), 1.0);
        return;
    }
    if (surface > 1.5) {
        frag_color = vec4(
            reflected_radiance(n, vec3(.22, .24, .26)), 1.0
        );
        return;
    }
    if (surface > .5) {
        vec3 roof = vec3(.12, .13, .14);
        if (facade_style > 1.5 && facade_style < 2.5) {
            roof = vec3(.30, .18, .055);
        } else if (facade_style > 4.5 && facade_style < 5.5) {
            roof = vec3(.31, .035, .025);
        }
        float roof_variation = hash21(floor(surface_uv * .08)) * .025;
        frag_color = vec4(
            reflected_radiance(n, roof + roof_variation), 1.0
        );
        return;
    }

    float bay_width = 4.2;
    float floor_height = 3.25;
    vec4 pane_bounds = vec4(.13, .82, .18, .72);
    vec3 wall = vec3(.18, .20, .23);
    vec3 glass = vec3(.025, .042, .075);
    float glass_amount = .18;
    float occupancy_threshold = .68;
    if (facade_style > .5 && facade_style < 1.5) {
        bay_width = 3.6; floor_height = 4.57;
        pane_bounds = vec4(.05, .95, .08, .92);
        wall = vec3(.055, .09, .14);
        glass = vec3(.02, .055, .12);
        glass_amount = .88; occupancy_threshold = .72;
    } else if (facade_style > 1.5 && facade_style < 2.5) {
        bay_width = 3.25; floor_height = 4.0;
        pane_bounds = vec4(.04, .96, .06, .93);
        wall = vec3(.32, .21, .06);
        glass = vec3(.16, .085, .018);
        glass_amount = .93; occupancy_threshold = .75;
    } else if (facade_style > 2.5 && facade_style < 3.5) {
        bay_width = 3.4; floor_height = 3.05;
        pane_bounds = vec4(.18, .78, .24, .68);
        wall = vec3(.25, .22, .18);
        glass = vec3(.025, .04, .06);
        glass_amount = .28; occupancy_threshold = .55;
    } else if (facade_style > 3.5 && facade_style < 4.5) {
        bay_width = 5.4; floor_height = 3.8;
        pane_bounds = vec4(.22, .76, .28, .69);
        wall = vec3(.34, .31, .25);
        glass_amount = .12; occupancy_threshold = .82;
    } else if (facade_style > 4.5 && facade_style < 5.5) {
        bay_width = 3.8; floor_height = 4.65;
        pane_bounds = vec4(.05, .95, .08, .92);
        wall = vec3(.045, .07, .105);
        glass = vec3(.018, .035, .075);
        glass_amount = .90; occupancy_threshold = .72;
    } else if (facade_style > 5.5) {
        bay_width = 3.2; floor_height = 3.55;
        pane_bounds = vec4(.12, .86, .16, .78);
        wall = vec3(.19, .18, .17);
        glass = vec3(.025, .038, .055);
        glass_amount = .55; occupancy_threshold = .38;
    }

    vec2 grid = surface_uv / vec2(bay_width, floor_height);
    vec2 within = fract(grid);
    vec2 aa = min(fwidth(grid) * 1.25, vec2(.12));
    float pane = interval(within.x, pane_bounds.x, pane_bounds.y, aa.x)
               * interval(within.y, pane_bounds.z, pane_bounds.w, aa.y);
    vec2 cell = floor(grid);
    float distance_m = length(camera_position - world_position);
    float occupied = distance_m < 1600.0
        ? step(occupancy_threshold, hash21(cell + facade_style * 31.7))
        : .38;
    float temperature = distance_m < 1600.0
        ? hash21(cell + facade_style * 17.0 + 11.3)
        : .55;
    vec3 window_color = mix(
        vec3(1.0, .42, .12), vec3(.55, .72, 1.0), temperature
    );
    vec3 view_direction = normalize(camera_position - world_position);
    float fresnel = pow(1.0 - max(dot(n, view_direction), 0.0), 4.0);
    vec3 facade = mix(wall, glass, pane * glass_amount);
    facade += glass * fresnel * glass_amount * .35;
    vec3 emission = window_color * pane * occupied
                  * window_radiance_w_m2_sr;

    if (facade_style > 2.5 && facade_style < 3.5) {
        float balcony = 1.0 - smoothstep(
            .035, .09, min(within.y, 1.0 - within.y)
        );
        facade += balcony * vec3(.10, .095, .086);
    }
    if (facade_style > 4.5 && facade_style < 5.5) {
        float column_mod = mod(surface_uv.x, 18.0);
        float column_distance = min(column_mod, 18.0 - column_mod);
        float red_column = 1.0 - smoothstep(.42, .86, column_distance);
        float beam_mod = mod(surface_uv.y, 32.0);
        float beam_distance = min(beam_mod, 32.0 - beam_mod);
        float red_beam = 1.0 - smoothstep(.30, .72, beam_distance);
        facade = mix(
            facade,
            vec3(.38, .018, .009),
            max(red_column, red_beam) * .92
        );
    }
    frag_color = vec4(reflected_radiance(n, facade) + emission, 1.0);
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
uniform sampler2D smoke_state;
uniform sampler3D smoke_state_3d;
uniform sampler2D scene_depth;
uniform int smoke_is_3d;
uniform vec3 camera_position;
uniform mat4 inverse_view_projection;
uniform vec3 volume_min;
uniform vec3 volume_max;
uniform vec4 smoke_xy_bounds;
uniform vec3 smoke_field_min;
uniform vec3 smoke_field_max;
uniform float depth_profile_sigma_m;
uniform float depth_profile_scale;
uniform int camera_inside;
uniform int ray_steps;
uniform float depth_bias_m;
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

    ivec2 depth_pixel = ivec2(gl_FragCoord.xy);
    float opaque_depth = texelFetch(scene_depth, depth_pixel, 0).r;
    if (opaque_depth < 1.0) {
        vec2 depth_size = vec2(textureSize(scene_depth, 0));
        vec2 ndc_xy = gl_FragCoord.xy / depth_size * 2.0 - 1.0;
        vec4 opaque_h = inverse_view_projection
                      * vec4(ndc_xy, opaque_depth * 2.0 - 1.0, 1.0);
        vec3 opaque_world = opaque_h.xyz / opaque_h.w;
        float opaque_distance = dot(opaque_world - camera_position, ray);
        t_far = min(t_far, opaque_distance - depth_bias_m);
    }
    if (t_far <= t_near) discard;

    float step_m = (t_far - t_near) / float(ray_steps);
    float jitter = hash12(gl_FragCoord.xy);
    float transmittance = 1.0;
    vec3 radiance = vec3(0.0);
    for (int i = 0; i < 64; ++i) {
        if (i >= ray_steps) break;
        float distance_m = t_near + (float(i) + jitter) * step_m;
        vec3 world = camera_position + ray * distance_m;
        vec2 state;
        if (smoke_is_3d != 0) {
            vec3 field_uvw = (world - smoke_field_min)
                           / (smoke_field_max - smoke_field_min);
            state = texture(smoke_state_3d, field_uvw).rg;
        } else {
            vec2 field_uv = (world.xy - smoke_xy_bounds.xy)
                          / (smoke_xy_bounds.zw - smoke_xy_bounds.xy);
            state = texture(smoke_state, field_uv).rg;
            float depth_profile = depth_profile_scale * exp(
                -0.5 * world.z * world.z
                / (depth_profile_sigma_m * depth_profile_sigma_m)
            );
            state *= depth_profile;
        }
        float density = max(state.r, 0.0);
        float temperature = max(state.g, 0.0);
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
        lighting_config: LightingConfig | None = None,
        camera_config: PhysicalCameraConfig | None = None,
    ) -> None:
        self.ctx, self.config, self.time_s = ctx, config, 0.0
        self.lighting_config = lighting_config or LightingConfig()
        self.camera_config = camera_config or PhysicalCameraConfig()
        self.frame_index = 0
        physical_fov_deg = vertical_fov_deg(self.camera_config)
        tan_half_fov = math.tan(math.radians(physical_fov_deg) * 0.5)
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        self.quad_buffer = ctx.buffer(quad.tobytes())
        self.background_program = ctx.program(
            vertex_shader=QUAD_VERTEX, fragment_shader=BACKGROUND_FRAGMENT
        )
        self.background_vao = ctx.simple_vertex_array(
            self.background_program, self.quad_buffer, "in_position"
        )
        self.background_program["tan_half_fov"] = tan_half_fov
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
        self.tonemap_program["photon_to_electron"].value = tuple(
            photon_to_electron_scale(self.camera_config)
        )
        self.tonemap_program["analog_gain"] = analog_gain(self.camera_config)
        self.tonemap_program["full_well_electrons"] = (
            self.camera_config.full_well_electrons
        )
        self.tonemap_program["read_noise_electrons"] = (
            self.camera_config.read_noise_electrons
        )
        self.tonemap_program["tan_half_fov"] = tan_half_fov
        self.tonemap_program["aspect"] = config.width / config.height
        self.tonemap_program["sensor_noise_enabled"] = int(
            self.camera_config.enable_sensor_noise
        )
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
        led_budget = led_energy_budget(self.lighting_config)
        self.scene_program["window_radiance_w_m2_sr"] = (
            led_budget.window_radiance_w_m2_sr
        )
        self.scene_program["air_extinction_per_m"] = (
            self.lighting_config.air_extinction_per_m
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
                    [(
                        buffer,
                        "3f 3f 1f 2f 1f",
                        "in_position",
                        "in_normal",
                        "in_surface",
                        "in_surface_uv",
                        "in_facade_style",
                    )],
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
        self.water_spectrum = water_spectrum
        self.water_atmosphere_accumulator_s = 0.0
        self.water_mask_cpu = water_mask
        self.water_mask_bounds_cpu = water_mask_bounds
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
        self.water_program["air_extinction_per_m"] = (
            self.lighting_config.air_extinction_per_m
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
        smoke_vertices = box_vertices(
            (sx0, sy0, sz0), (sx1, sy1, sz1)
        )
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
        self.smoke_buffer = ctx.buffer(
            smoke_vertices.tobytes(), dynamic=True
        )
        self.smoke_index_buffer = ctx.buffer(smoke_indices.tobytes())
        self.smoke_vao = ctx.vertex_array(
            self.smoke_program,
            [(self.smoke_buffer, "3f", "in_position")],
            self.smoke_index_buffer,
            index_element_size=4,
        )
        self.smoke_state_texture = ctx.texture(
            smoke_config.grid_size, components=2, dtype="f4"
        )
        self.smoke_state_texture.filter = moderngl.LINEAR, moderngl.LINEAR
        self.smoke_state_texture.repeat_x = False
        self.smoke_state_texture.repeat_y = False
        self.smoke_program["smoke_state"] = 4
        self.smoke_program["smoke_state_3d"] = 5
        self.smoke_program["scene_depth"] = 6
        self.smoke_program["smoke_is_3d"] = 0
        self.smoke_program["depth_bias_m"] = smoke_config.volume_depth_bias_m
        self.smoke_program["volume_min"].value = (sx0, sy0, sz0)
        self.smoke_program["volume_max"].value = (sx1, sy1, sz1)
        self.smoke_program["smoke_xy_bounds"].value = (sx0, sy0, sx1, sy1)
        self.smoke_program["smoke_field_min"].value = (sx0, sy0, sz0)
        self.smoke_program["smoke_field_max"].value = (sx1, sy1, sz1)
        dz = smoke_config.volume_depth_m / smoke_config.volume_slices
        depth_sigma_m = max(smoke_config.plume_depth_m * 0.5, dz)
        depth_integral = (
            math.sqrt(2.0 * math.pi)
            * depth_sigma_m
            * math.erf(
                smoke_config.volume_depth_m
                / (2.0 * math.sqrt(2.0) * depth_sigma_m)
            )
        )
        self.smoke_program["depth_profile_sigma_m"] = depth_sigma_m
        self.smoke_program["depth_profile_scale"] = (
            smoke_config.plume_depth_m / depth_integral
        )
        # Four sigma on either side retains >99.993% of the normalized
        # Gaussian mass while avoiding fragments in optically empty tails.
        self.smoke_active_depth_m = min(
            smoke_config.volume_depth_m, 8.0 * depth_sigma_m
        )
        self.smoke_program["ray_steps"] = smoke_config.volume_ray_steps
        self.smoke_bounds = np.array(
            [[sx0, sy0, sz0], [sx1, sy1, sz1]], dtype=np.float32
        )
        self.smoke_texture_bounds = self.smoke_bounds.copy()
        self.smoke_revision = -1
        self.hdr_texture = ctx.texture(
            (config.width, config.height), components=4, dtype="f2"
        )
        self.scene_depth_texture = ctx.depth_texture(
            (config.width, config.height)
        )
        self.scene_depth_texture.compare_func = ""
        self.scene_depth_texture.filter = (
            moderngl.NEAREST,
            moderngl.NEAREST,
        )
        self.scene_depth_texture.repeat_x = False
        self.scene_depth_texture.repeat_y = False
        self.hdr_fbo = ctx.framebuffer(
            [self.hdr_texture], self.scene_depth_texture
        )
        self.smoke_fbo = ctx.framebuffer([self.hdr_texture])
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
        self.reflection_sky_accumulator_s = 1.0
        self.reflection_ready = False
        self.reflection_camera_position = np.full(3, np.inf, dtype=np.float32)
        self.reflection_camera_forward = np.zeros(3, dtype=np.float32)
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
            physical_fov_deg, config.width / config.height, .1, 2500
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
        self.smoke_program["inverse_view_projection"].write(
            np.linalg.inv(view_projection).T.astype(np.float32).tobytes()
        )
        self.water_program["camera_position"].value = tuple(camera.position_m)
        self.scene_program["camera_position"].value = tuple(camera.position_m)
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
        self.scene_program["camera_position"].value = tuple(reflected_position)
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
        ambient_illuminance_lux = (
            celestial.twilight_illuminance_lux
            + celestial.moon_illuminance_lux
            + self.lighting_config.calibrated_urban_ambient_illuminance_lux
        )
        self.scene_program["ambient_irradiance_w_m2"] = (
            radiometric_irradiance_from_illuminance(
                ambient_illuminance_lux,
                self.lighting_config.twilight_spectral_luminous_efficacy_lm_w,
            )
        )

    def _update_dynamic_lights(self, world: FireworkWorld) -> np.ndarray:
        count = world.stars.count
        if count:
            radiant_power_w = world.stars.intensity()
            lights = cluster_radiant_lights(
                world.stars.position_m[:count],
                world.stars.color_linear[:count],
                radiant_power_w,
                min(self.lighting_config.dynamic_light_count, 8),
            )
        else:
            radiant_power_w = np.empty(0, dtype=np.float32)
            lights = cluster_radiant_lights(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                radiant_power_w,
                min(self.lighting_config.dynamic_light_count, 8),
            )
        for program in (self.scene_program, self.water_program):
            program["dynamic_light_count"] = lights.count
            program["dynamic_light_position"].write(
                lights.positions_m.tobytes()
            )
            program["dynamic_light_color"].write(lights.colors.tobytes())
            program["dynamic_light_power_w"].write(
                lights.radiant_power_w.tobytes()
            )
        return radiant_power_w

    def _update_water_forcing(
        self, atmosphere: AtmosphereConfig, frame_dt_s: float
    ) -> None:
        self.water_atmosphere_accumulator_s += frame_dt_s
        interval_s = self.water_config.atmosphere_update_interval_s
        if self.water_atmosphere_accumulator_s < interval_s:
            return
        elapsed_s = self.water_atmosphere_accumulator_s
        self.water_atmosphere_accumulator_s %= interval_s
        wind = np.asarray(atmosphere.wind_velocity_mps, dtype=np.float32)
        wind_xz = wind[[0, 2]]
        wind_speed = float(np.linalg.norm(wind_xz))
        wind_from = (
            math.degrees(math.atan2(-float(wind_xz[0]), float(wind_xz[1])))
            % 360.0
        )
        target_config = replace(
            self.water_config,
            wind_speed_mps=max(wind_speed, 0.1),
            wind_direction_deg=wind_from,
            fetch_length_m=estimate_fetch_length_m(
                self.water_mask_cpu,
                self.water_mask_bounds_cpu,
                wind_xz,
            ),
        )
        target = build_directional_spectrum(target_config)
        self.water_spectrum = relax_wave_spectrum(
            self.water_spectrum,
            target,
            elapsed_s,
            self.water_config.wind_response_time_s,
        )
        self.water_program["waves"].write(
            self.water_spectrum.components.tobytes()
        )
        self.significant_wave_height_m = (
            self.water_spectrum.significant_wave_height_m
        )

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
        radiant_power_w = self._update_dynamic_lights(world)
        self._update_water_forcing(world.atmosphere, frame_dt_s)
        self.reflection_accumulator_s += frame_dt_s
        self.reflection_sky_accumulator_s += frame_dt_s
        smoke_revision = smoke.revision if smoke is not None else -1
        fluid_updated = smoke_revision != self.last_rendered_smoke_revision
        camera_changed = (
            float(
                np.linalg.norm(
                    camera.position_m - self.reflection_camera_position
                )
            )
            > 0.025
            or float(
                np.dot(camera.forward, self.reflection_camera_forward)
            )
            < 0.9999985
        )
        reflection_invalid = camera_changed or (
            self.reflection_sky_accumulator_s >= 1.0
        )
        if (
            not self.reflection_ready
            or (
                self.reflection_accumulator_s >= self.reflection_interval_s
                and reflection_invalid
                and not fluid_updated
            )
        ):
            self._render_reflection(camera)
            self.reflection_accumulator_s %= self.reflection_interval_s
            self.reflection_sky_accumulator_s = 0.0
            self.reflection_ready = True
            self.reflection_camera_position[:] = camera.position_m
            self.reflection_camera_forward[:] = camera.forward
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
                - world.stars.velocity_mps[:count]
                * self.camera_config.shutter_time_s
            )
            data[:, 6:9] = world.stars.color_linear[:count]
            data[:, 9] = radiant_power_w
            self.particle_buffer.write(data.tobytes())
            self.ctx.enable(moderngl.BLEND)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.particle_program["time_s"] = self.time_s
            for reflection in (1.0, 0.0):
                self.particle_program["reflection"] = reflection
                self.particle_vao.render(moderngl.POINTS, vertices=count)
        if smoke is not None and smoke.has_visible_smoke():
            self.smoke_fbo.use()
            if smoke.revision != self.smoke_revision:
                render_state_texture = getattr(
                    smoke, "render_state_texture", None
                )
                if render_state_texture is None:
                    smoke_state = np.empty(
                        (
                            smoke.density_kg_m3.shape[0],
                            smoke.density_kg_m3.shape[1],
                            2,
                        ),
                        dtype=np.float32,
                    )
                    smoke_state[:, :, 0] = smoke.density_kg_m3
                    smoke_state[:, :, 1] = smoke.temperature_excess_k
                    self.smoke_state_texture.write(smoke_state.tobytes())
                active_bounds_method = getattr(
                    smoke, "active_render_bounds", None
                )
                active_bounds = (
                    active_bounds_method(
                        self.smoke_active_depth_m
                    )
                    if active_bounds_method is not None
                    else active_slice_bounds(
                        smoke.density_kg_m3,
                        (
                            self.smoke_texture_bounds[0, 0],
                            self.smoke_texture_bounds[0, 1],
                        ),
                        (
                            self.smoke_texture_bounds[1, 0],
                            self.smoke_texture_bounds[1, 1],
                        ),
                        -0.5 * self.smoke_active_depth_m,
                        0.5 * self.smoke_active_depth_m,
                    )
                )
                if active_bounds is not None:
                    active_minimum, active_maximum = active_bounds
                    self.smoke_bounds[:] = (
                        active_minimum,
                        active_maximum,
                    )
                    self.smoke_buffer.write(
                        box_vertices(
                            active_minimum, active_maximum
                        ).tobytes()
                    )
                    self.smoke_program["volume_min"].value = tuple(
                        active_minimum
                    )
                    self.smoke_program["volume_max"].value = tuple(
                        active_maximum
                    )
                    self.smoke_program["camera_inside"] = int(
                        np.all(camera.position_m >= active_minimum)
                        and np.all(camera.position_m <= active_maximum)
                    )
                self.smoke_revision = smoke.revision
            render_state_texture = getattr(
                smoke, "render_state_texture", self.smoke_state_texture
            )
            smoke_is_3d = int(getattr(smoke, "is_3d", False))
            self.smoke_program["smoke_is_3d"] = smoke_is_3d
            if smoke_is_3d:
                self.smoke_program["smoke_field_min"].value = (
                    smoke.x_min, smoke.y_min, smoke.z_min
                )
                self.smoke_program["smoke_field_max"].value = (
                    smoke.x_max, smoke.y_max, smoke.z_max
                )
            render_state_texture.use(5 if smoke_is_3d else 4)
            self.scene_depth_texture.use(6)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.blend_func = (
                moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            )
            self.smoke_vao.render(moderngl.TRIANGLES)
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
        self.frame_index = (self.frame_index + 1) % 1_000_000
        self.tonemap_program["frame_index"] = float(self.frame_index)
        self.tonemap_vao.render(moderngl.TRIANGLE_STRIP)
