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
uniform vec3 camera_position;
// Observation distances at which vegetation detail stops being resolvable.
// Derived from the camera optics in simulator/vegetation.py, not chosen.
uniform float blade_full_detail_m;
uniform float blade_cutoff_m;
uniform float tree_sway_cutoff_m;
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
    float observation_distance_m = distance(camera_position, in_position);
    if (in_surface > 15.5) {
        // A blade narrower than the sample spacing is decided by where the
        // pixel centre falls, so it is collapsed toward its base rather than
        // rasterised. Smoothstep keeps the transition from reading as a seam.
        float fade = smoothstep(
            blade_full_detail_m, blade_cutoff_m, observation_distance_m
        );
        float detail = 1.0 - fade;
        if (detail <= 0.0) {
            // Degenerate triangle: the blade produces no fragments at all.
            animated_position.y = in_position.y - in_surface_uv.y * 1e-3;
            world_position = animated_position
                           + vec3(0.0, base_height, 0.0);
            world_normal = in_normal;
            surface = in_surface;
            surface_uv = in_surface_uv;
            facade_style = in_facade_style;
            gl_Position = view_projection * vec4(world_position, 1.0);
            return;
        }
        float tip_weight = in_surface_uv.y * in_surface_uv.y;
        float response = 1.0 - exp(-max(wind_speed_mps, 0.0) / 3.2);
        vec2 bend = wind_direction * tip_weight * response
                  * (.16 + .045 * gust);
        animated_position.xz += bend * detail;
        // Shrink toward the base so the blade leaves the frame by receding
        // into the ground surface that already carries its colour.
        animated_position.y = mix(
            in_position.y - in_surface_uv.y * .5,
            in_position.y - length(bend) * .18 * tip_weight,
            detail
        );
        animated_normal = normalize(
            in_normal + vec3(-bend.x, .18, -bend.y) * tip_weight * detail
        );
    } else if (in_surface > 10.5 && in_surface < 11.5) {
        // Crown geometry stays; only the motion is gated, because a
        // displacement below a pixel cannot be seen but is still computed.
        float sway_detail = 1.0 - smoothstep(
            tree_sway_cutoff_m * .5, tree_sway_cutoff_m,
            observation_distance_m
        );
        if (sway_detail > 0.0) {
            float crown_weight = smoothstep(2.0, 11.0, in_position.y);
            vec2 sway = wind_direction * crown_weight
                      * min(wind_speed_mps * .018, .13) * gust;
            animated_position.xz += sway * sway_detail;
        }
    }
    world_position = animated_position + vec3(0.0, base_height, 0.0);
    world_normal = animated_normal;
    surface = in_surface;
    surface_uv = in_surface_uv;
    facade_style = in_facade_style;
    gl_Position = view_projection * vec4(world_position, 1.0);
}
