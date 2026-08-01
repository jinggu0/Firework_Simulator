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
uniform float wind_speed_mps;
uniform int dynamic_light_count;
uniform vec3 dynamic_light_position[8];
uniform vec3 dynamic_light_color[8];
uniform float dynamic_light_power_w[8];
out vec4 frag_color;
#include "air_extinction.glsl"
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
    float optical_path_m = .42 / max(n_dot_v, .08);
    vec3 absorption_per_m = vec3(.62, .22, .095);
    vec3 water_transmission = exp(-absorption_per_m * optical_path_m);
    vec3 subsurface_scatter = vec3(.000055, .00024, .00031)
                            * (vec3(1.0) - water_transmission);
    vec3 water_body = vec3(.000035, .00011, .00015)
                    * water_transmission + subsurface_scatter;
    float grazing_haze = pow(1.0 - n_dot_v, 3.0) * .0007;
    vec3 radiance = mix(water_body, reflected_radiance, fresnel)
                  + grazing_haze;
    vec2 mask_texel = 1.0 / vec2(textureSize(water_mask, 0));
    float shore_gradient = length(vec2(
        texture(water_mask, mask_uv + vec2(mask_texel.x, 0)).r
      - texture(water_mask, mask_uv - vec2(mask_texel.x, 0)).r,
        texture(water_mask, mask_uv + vec2(0, mask_texel.y)).r
      - texture(water_mask, mask_uv - vec2(0, mask_texel.y)).r
    ));
    float slope = length(fine_gradient + world_normal.xz);
    float crest_foam = smoothstep(.34, .62, slope)
                     * clamp(wind_speed_mps / 9.0, 0.0, 1.0);
    float foam = clamp(shore_gradient * .78 + crest_foam * .24, 0.0, 1.0);
    radiance = mix(radiance, vec3(.010, .012, .013), foam);
    // Fixed-cost GGX reflection from energy-conserving firework clusters.
    float roughness = clamp(.075 + slope * .11 + wind_speed_mps * .004,
                            .075, .24);
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
        vec3 irradiance = dynamic_light_power_w[i]
                        * air_transmittance(
                              dynamic_light_position[i].y, world_position.y,
                              distance_m
                          )
                        / (4.0 * PI * distance_squared);
        radiance += dynamic_light_color[i] * irradiance * brdf * n_dot_l;
    }
    frag_color = vec4(radiance, 1.0);
}
