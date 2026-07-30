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
