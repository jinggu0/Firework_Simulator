#version 330
// Human Vision Mode display transform.
//
// The sibling tonemap.frag models a camera: fixed aperture, fixed shutter,
// quantum efficiency, full well, read noise. None of that describes an eye.
// This path replaces the sensor stage with the observer state computed in
// simulator/human_vision.py.
//
// It ends, as the camera path does, in a display transform. An SDR monitor
// covers roughly 0.1 to 300 cd/m2; a shell burst is orders of magnitude beyond
// that. What follows is therefore a perceptual compromise, not a reproduction
// of the retinal image. See docs/ARCHITECTURE.md.
uniform sampler2D hdr_texture;
uniform sampler2D bloom_texture;
uniform sampler2D adaptation_texture;
uniform float bloom_strength;

// Observer state, from simulator/human_vision.py.
uniform float pupil_gain;                 // relative to a 3 mm photopic pupil
uniform float cone_fraction;              // 0 rod-only, 1 cone-only
uniform float adapting_luminance_cd_m2;
uniform float glare_constant;             // Stiles-Holladay k
uniform float acuity_e2_deg;              // eccentricity where acuity halves
uniform float maximum_blur_lod;
uniform float gaze_u;
uniform float gaze_v;

uniform float tan_half_fov;
uniform float aspect;

in vec2 uv;
out vec4 frag_color;

const vec3 PHOTOPIC_WEIGHTS = vec3(.2126, .7152, .0722);

vec3 aces(vec3 x) {
    return clamp((x * (2.51 * x + .03)) / (x * (2.43 * x + .59) + .14),
                 0.0, 1.0);
}

// Angle between this pixel's ray and the fixation ray, in degrees.
float eccentricity_deg() {
    vec2 scale = vec2(aspect * tan_half_fov, tan_half_fov);
    vec2 here = (uv * 2.0 - 1.0) * scale;
    vec2 gaze = (vec2(gaze_u, gaze_v) * 2.0 - 1.0) * scale;
    vec3 a = normalize(vec3(here, 1.0));
    vec3 b = normalize(vec3(gaze, 1.0));
    return degrees(acos(clamp(dot(a, b), -1.0, 1.0)));
}

void main() {
    float eccentricity = eccentricity_deg();

    // Peripheral acuity. Cortical magnification gives resolvable frequency
    // 1 / (1 + e / E2), so detail is dropped by sampling a coarser mip rather
    // than by blurring the whole image uniformly.
    float acuity = 1.0 / (1.0 + eccentricity / acuity_e2_deg);
    float lod = maximum_blur_lod * (1.0 - acuity);
    vec3 scene = textureLod(hdr_texture, uv, lod).rgb;
    vec3 bloom = texture(bloom_texture, uv).rgb * bloom_strength;

    // Disability glare. Light scattered in the ocular media veils the retinal
    // image, which is why a burst washes out its surroundings rather than
    // simply looking bright. The Stiles-Holladay term falls as 1/theta^2,
    // which is far wider than the bloom kernel, so a heavily reduced mip of
    // the bloom stands in for that wide tail. This is an approximation to the
    // CIE glare equation, not an evaluation of it.
    vec3 wide_field = textureLod(bloom_texture, uv, 5.0).rgb;
    float veiling = glare_constant * dot(wide_field, PHOTOPIC_WEIGHTS) * .01;

    // The pupil is the eye's aperture control and replaces the camera's
    // f-number: dark-adapted, it gathers roughly five times the light.
    vec3 retinal = (scene + bloom + vec3(veiling)) * pupil_gain;

    // Local adaptation. Dividing by the locally adapted level rather than by a
    // single global exposure is what produces an afterimage: where a burst has
    // just faded, the local level is still high and the region reads dark.
    float local_adaptation = texture(adaptation_texture, uv).r;
    float reference = max(
        mix(local_adaptation, adapting_luminance_cd_m2, .35), 1e-6
    );
    vec3 normalized = retinal / (reference * 6.0 + 1e-5);

    // Mesopic mixing. There is one rod photopigment, so rod vision carries no
    // hue: as the cone contribution falls the image collapses toward its
    // luminance. The Purkinje shift, which would additionally favour short
    // wavelengths, is not modelled here because it needs the tabulated
    // scotopic luminous efficiency V'(lambda), which this project does not
    // hold. Only the achromatic collapse is applied.
    float luminance = dot(normalized, PHOTOPIC_WEIGHTS);
    vec3 mesopic = mix(vec3(luminance), normalized, cone_fraction);

    vec3 mapped = aces(mesopic);
    frag_color = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
