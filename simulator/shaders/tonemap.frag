#version 330
// Physical Camera Mode display transform: lens, sensor, then display.
//
// Two stages the sensor model was missing are applied here. The lens forms a
// geometrically distorted image rather than an ideal pinhole one, and the
// sensor's own spectral response has to be balanced out — quantum efficiency
// and photon energy together turn a neutral scene into electrons in the ratio
// 0.896 : 1 : 0.579, a yellow-green cast that nothing previously corrected.
//
// Everything upstream of this shader is linear scene-referred radiance;
// everything it emits is display-referred and is no longer valid input to a
// colour metric. See simulator/camera_optics.py for the CPU reference.
uniform sampler2D hdr_texture; uniform sampler2D bloom_texture;
uniform float bloom_strength;
uniform vec3 photon_to_electron;
uniform vec3 white_balance_gain;
uniform vec3 distortion_radial;      // Brown-Conrady k1, k2, k3
uniform vec2 distortion_tangential;  // Brown-Conrady p1, p2
uniform float analog_gain;
uniform float full_well_electrons;
uniform float read_noise_electrons;
uniform float tan_half_fov;
uniform float aspect;
uniform float frame_index;
uniform int sensor_noise_enabled;
in vec2 uv; out vec4 frag_color;

// Must match camera_optics.DISTORTION_INVERSE_ITERATIONS, or the CPU reference
// V-23 predicts the frame from would stop describing this shader.
const int DISTORTION_ITERATIONS = 5;

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

// Which ideal direction landed on this part of the sensor. The frame is
// rendered through an ideal pinhole, so forming the lens's image means
// inverting the Brown-Conrady polynomial, which has no closed form. The
// fixed-point iteration is OpenCV's, and with the shipped identity
// coefficients its first step is exact: radial is 1.0 and tangential is zero,
// so the result is bit-identical to the input and the default camera path is
// unperturbed.
vec2 undistort(vec2 distorted) {
    vec2 ideal = distorted;
    for (int i = 0; i < DISTORTION_ITERATIONS; ++i) {
        float r2 = dot(ideal, ideal);
        float radial = 1.0 + r2 * (distortion_radial.x
                     + r2 * (distortion_radial.y + r2 * distortion_radial.z));
        vec2 tangential = vec2(
            2.0 * distortion_tangential.x * ideal.x * ideal.y
                + distortion_tangential.y * (r2 + 2.0 * ideal.x * ideal.x),
            distortion_tangential.x * (r2 + 2.0 * ideal.y * ideal.y)
                + 2.0 * distortion_tangential.y * ideal.x * ideal.y
        );
        ideal = (distorted - tangential) / radial;
    }
    return ideal;
}

void main() {
    vec2 half_extent = vec2(aspect * tan_half_fov, tan_half_fov);
    vec2 sensor_position = undistort((uv * 2.0 - 1.0) * half_extent);
    vec2 source_uv = sensor_position / half_extent * .5 + .5;
    vec3 hdr = texture(hdr_texture, source_uv).rgb;
    vec3 bloom = texture(bloom_texture, source_uv).rgb * bloom_strength;
    // Natural falloff is a function of the field angle, so it follows the
    // undistorted position rather than the position on the sensor.
    float cos_theta = inversesqrt(1.0 + dot(sensor_position, sensor_position));
    float lens_vignetting = pow(cos_theta, 4.0);
    vec3 electrons = max(hdr + bloom, vec3(0.0))
                   * photon_to_electron * lens_vignetting;
    if (sensor_noise_enabled != 0) {
        // Seeded on the sensor pixel, not the scene position: shot and read
        // noise are properties of the photosite the light landed on.
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
    // White balance sits after the full well because that is where a raw
    // pipeline applies it: a channel that saturates clips *before* the gains,
    // which is why a clipped burst shifts hue instead of staying neutral.
    vec3 normalized_signal = electrons / full_well_electrons
                           * white_balance_gain * analog_gain;
    vec3 mapped = aces(normalized_signal);
    frag_color = vec4(pow(mapped, vec3(1.0/2.2)), 1);
}
