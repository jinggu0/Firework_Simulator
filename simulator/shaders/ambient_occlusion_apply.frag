#version 330
in vec2 uv;
out vec4 frag_color;
uniform sampler2D ambient_occlusion;
uniform sampler2D scene_depth;
uniform float near_plane_m;
uniform float far_plane_m;

float view_distance_m(float depth) {
    float ndc = depth * 2.0 - 1.0;
    return (2.0 * near_plane_m * far_plane_m)
         / (far_plane_m + near_plane_m
            - ndc * (far_plane_m - near_plane_m));
}

void main() {
    float depth = texture(scene_depth, uv).r;
    if (depth >= .999999) {
        frag_color = vec4(1.0);
        return;
    }

    // Joint bilateral upsampling: the four half-resolution neighbours retain
    // their bilinear footprint, but a neighbour across a facade silhouette or
    // bridge edge receives no weight. Plain linear filtering leaks dark AO
    // into the sky and forms halos exactly where contact depth matters most.
    ivec2 ao_size = textureSize(ambient_occlusion, 0);
    vec2 texel_position = uv * vec2(ao_size) - .5;
    ivec2 base = ivec2(floor(texel_position));
    vec2 fraction = fract(texel_position);
    float centre_distance = view_distance_m(depth);
    float depth_sigma_m = max(.08, centre_distance * .003);
    float weighted_visibility = 0.0;
    float total_weight = 0.0;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            ivec2 coordinate = clamp(
                base + ivec2(x, y), ivec2(0), ao_size - ivec2(1)
            );
            vec2 sample_uv = (vec2(coordinate) + .5) / vec2(ao_size);
            float sample_depth = texture(scene_depth, sample_uv).r;
            if (sample_depth >= .999999) continue;
            float sample_distance = view_distance_m(sample_depth);
            vec2 linear_weight = mix(1.0 - fraction, fraction, vec2(x, y));
            float spatial_weight = linear_weight.x * linear_weight.y;
            float depth_weight = exp2(
                -abs(sample_distance - centre_distance) / depth_sigma_m
            );
            float weight = spatial_weight * depth_weight;
            weighted_visibility += texelFetch(
                ambient_occlusion, coordinate, 0
            ).r * weight;
            total_weight += weight;
        }
    }
    float visibility = total_weight > 1e-5
                     ? weighted_visibility / total_weight
                     : texture(ambient_occlusion, uv).r;
    frag_color = vec4(vec3(visibility), 1.0);
}
