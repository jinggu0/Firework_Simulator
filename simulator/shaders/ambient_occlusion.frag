#version 330
in vec2 uv;
out float frag_occlusion;
uniform sampler2D scene_depth;
uniform float tan_half_fov;
uniform float aspect;
uniform float effect_radius_m;
uniform float strength;
uniform float near_plane_m;
uniform float far_plane_m;
uniform vec2 occlusion_resolution;
const int SAMPLE_COUNT = 8;
const float PI2 = 6.28318530718;

float view_distance_m(float depth) {
    // Keeping the closed form here avoids eight inverse-matrix multiplies per
    // half-resolution fragment. Positive distance points away from the eye.
    float ndc = depth * 2.0 - 1.0;
    return (2.0 * near_plane_m * far_plane_m)
         / (far_plane_m + near_plane_m
            - ndc * (far_plane_m - near_plane_m));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    float centre_raw = texture(scene_depth, uv).r;
    if (centre_raw >= .999999) {
        frag_occlusion = 1.0;
        return;
    }
    float centre = view_distance_m(centre_raw);
    if (centre > 320.0) {
        frag_occlusion = 1.0;
        return;
    }

    vec2 depth_gradient = vec2(dFdx(centre), dFdy(centre));
    vec2 radius_uv = effect_radius_m
                   / max(centre, .2)
                   / vec2(2.0 * tan_half_fov * aspect, 2.0 * tan_half_fov);
    float rotation = hash12(floor(gl_FragCoord.xy)) * PI2;
    float obscurance = 0.0;
    float accepted = 0.0;
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        float ring = .28 + .72 * (float(i) + .5) / float(SAMPLE_COUNT);
        float angle = rotation + PI2 * float(i) / float(SAMPLE_COUNT);
        vec2 offset_uv = vec2(cos(angle), sin(angle)) * radius_uv * ring;
        vec2 sample_uv = uv + offset_uv;
        if (any(lessThan(sample_uv, vec2(0.0)))
            || any(greaterThan(sample_uv, vec2(1.0)))) continue;
        float sample_raw = texture(scene_depth, sample_uv).r;
        if (sample_raw >= .999999) continue;
        float sample_distance = view_distance_m(sample_raw);
        vec2 offset_fragments = offset_uv * occlusion_resolution;
        float expected_plane = centre
                             + dot(depth_gradient, offset_fragments);
        float gap_m = expected_plane - sample_distance;
        float bias_m = .025 + centre * .00015;
        float occluder = smoothstep(bias_m, bias_m + .16, gap_m);
        float range_weight = 1.0 - smoothstep(
            effect_radius_m * .15, effect_radius_m, abs(gap_m)
        );
        obscurance += occluder * range_weight;
        accepted += 1.0;
    }
    float visibility = 1.0 - strength * obscurance / max(accepted, 1.0);
    // Never crush a crevice to black: AO approximates missing indirect sky
    // light, while direct lamps and firework light remain in the same buffer.
    frag_occlusion = clamp(visibility, .48, 1.0);
}
