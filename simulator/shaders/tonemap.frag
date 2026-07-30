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
