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
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float value_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 local = fract(p);
    local = local * local * (3.0 - 2.0 * local);
    float a = hash21(cell);
    float b = hash21(cell + vec2(1.0, 0.0));
    float c = hash21(cell + vec2(0.0, 1.0));
    float d = hash21(cell + vec2(1.0, 1.0));
    return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}
void main() {
    vec2 uv = (world_position.xz - water_mask_bounds.xy)
            / (water_mask_bounds.zw - water_mask_bounds.xy);
    if (all(greaterThanEqual(uv, vec2(0.0)))
        && all(lessThanEqual(uv, vec2(1.0)))
        && texture(water_mask, uv).r >= 0.5) discard;
    float broad = value_noise(world_position.xz * .035);
    float grit = value_noise(world_position.xz * 2.9 + 17.0);
    float drainage = value_noise(world_position.xz * .11 - 43.0);
    float sky_light = max(world_normal.y, 0.15);
    vec3 soil = mix(vec3(.00020, .00016, .00011),
                    vec3(.00039, .00043, .00027), broad);
    soil *= mix(.72, 1.08, drainage);
    soil += vec3(.000045, .000040, .000030)
          * smoothstep(.86, .97, grit);
    vec3 ground = soil * sky_light * sky_ambient_scale;
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
