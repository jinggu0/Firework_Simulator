#version 330
in vec3 world_position;
in vec3 world_normal;
uniform sampler2D water_mask;
uniform vec4 water_mask_bounds;
uniform float sky_ambient_scale;
uniform int static_light_count;
uniform vec3 static_light_position[4];
uniform vec3 static_light_color;
uniform float static_light_power_w;
out vec4 frag_color;
const float PI = 3.14159265359;
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
    for (int i = 0; i < 4; ++i) {
        if (i >= static_light_count) break;
        vec3 displacement = static_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), .5);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float down_lobe = pow(max(light_direction.y, 0.0), 3.0);
        float irradiance = static_light_power_w * 2.0 * down_lobe
                         / (PI * distance_squared);
        ground += vec3(.16, .17, .15) * static_light_color * irradiance
                * max(dot(world_normal, light_direction), 0.0) / PI;
    }
    frag_color = vec4(ground, 1.0);
}
