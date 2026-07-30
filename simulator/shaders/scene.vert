#version 330
in vec3 in_position;
in vec3 in_normal;
in float in_surface;
in vec2 in_surface_uv;
in float in_facade_style;
uniform mat4 view_projection;
uniform sampler2D terrain_height;
uniform vec4 terrain_bounds;
uniform float time_s;
uniform vec2 wind_xz;
uniform float wind_speed_mps;
out vec3 world_position;
out vec3 world_normal;
out float surface;
out vec2 surface_uv;
out float facade_style;
void main() {
    vec2 terrain_uv = (in_position.xz - terrain_bounds.xy)
                    / (terrain_bounds.zw - terrain_bounds.xy);
    float base_height = texture(terrain_height, terrain_uv).r;
    vec3 animated_position = in_position;
    vec3 animated_normal = in_normal;
    vec2 wind_direction = length(wind_xz) > .01
                        ? normalize(wind_xz) : vec2(1.0, 0.0);
    float spatial_phase = dot(in_position.xz, vec2(.73, .51));
    float gust = sin(time_s * 1.37 + spatial_phase)
               + .38 * sin(time_s * 2.91 + spatial_phase * 1.83);
    if (in_surface > 15.5) {
        float tip_weight = in_surface_uv.y * in_surface_uv.y;
        float response = 1.0 - exp(-max(wind_speed_mps, 0.0) / 3.2);
        vec2 bend = wind_direction * tip_weight * response
                  * (.16 + .045 * gust);
        animated_position.xz += bend;
        animated_position.y -= length(bend) * .18 * tip_weight;
        animated_normal = normalize(
            in_normal + vec3(-bend.x, .18, -bend.y) * tip_weight
        );
    } else if (in_surface > 10.5 && in_surface < 11.5) {
        float crown_weight = smoothstep(2.0, 11.0, in_position.y);
        vec2 sway = wind_direction * crown_weight
                  * min(wind_speed_mps * .018, .13) * gust;
        animated_position.xz += sway;
    }
    world_position = animated_position + vec3(0.0, base_height, 0.0);
    world_normal = animated_normal;
    surface = in_surface;
    surface_uv = in_surface_uv;
    facade_style = in_facade_style;
    gl_Position = view_projection * vec4(world_position, 1.0);
}
