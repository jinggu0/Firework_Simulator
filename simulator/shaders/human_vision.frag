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
uniform float chromatic_degree;           // CIECAM02 D, 0 none, 1 complete
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

// Linear sRGB to CAT02 cone response and back, from simulator/color.py. GLSL
// mat3 literals are column-major, so these are the columns; a test extracts
// them and compares against the Python matrices, because a transposed cone
// matrix produces a plausible-looking image rather than an obvious failure.
const mat3 LINEAR_SRGB_TO_LMS = mat3(
    0.39040536, 0.07084159, 0.02310818,
    0.54994112, 0.96317176, 0.12802080,
    0.00892632, 0.00135775, 0.93624512
);
const mat3 LMS_TO_LINEAR_SRGB = mat3(
     2.85846767, -0.21018227, -0.04181200,
    -1.62878773,  1.15820086, -0.11816935,
    -0.02489104,  0.00032428,  1.06866637
);

// Chromatic adaptation: the observer discounts the illuminant, so a scene lit
// warm does not read uniformly orange. Von Kries gain control in the cone
// channels, taking the field's own average chromaticity to the display white.
//
// The adapting white arrives normalised to unit luminance and the display
// white is unit luminance by construction, so the adapted white is a convex
// combination of two unit-luminance whites. Luminance is linear in LMS, so it
// is preserved exactly and brightness adaptation is left undisturbed.
vec3 chromatically_adapt(vec3 colour, vec3 adapting_white) {
    // Renormalised on read as well as on write, so the luminance-preserving
    // property holds however the state was stored.
    vec3 white = adapting_white
               / max(dot(adapting_white, PHOTOPIC_WEIGHTS), 1e-6);
    vec3 display_lms = LINEAR_SRGB_TO_LMS * vec3(1.0);
    vec3 field_lms = LINEAR_SRGB_TO_LMS * white;
    vec3 gain = chromatic_degree * (display_lms / max(field_lms, vec3(1e-6)))
              + (1.0 - chromatic_degree);
    return LMS_TO_LINEAR_SRGB * ((LINEAR_SRGB_TO_LMS * colour) * gain);
}

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
    // Blended from two explicit integer levels rather than left to hardware
    // trilinear. This driver implements GL_LINEAR_MIPMAP_LINEAR as
    // *brilinear*: the level weight is clamp((frac - 1/6) / (2/3), 0, 1), so
    // it snaps to the nearer level over the outer sixth of each transition and
    // a requested 0.75 is actually applied as 0.875. That is measured rather
    // than assumed — V-24 pinned the curve at five points — and it made the
    // peripheral blur a property of the driver instead of a property of the
    // eye, which a reconstruction cannot have. At an integer level the
    // hardware blend weight is zero and the fetch is exact, so two taps and a
    // mix put the interpolation somewhere portable and predictable.
    float level = floor(lod);
    vec3 scene = mix(textureLod(hdr_texture, uv, level).rgb,
                     textureLod(hdr_texture, uv, level + 1.0).rgb,
                     lod - level);
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
    vec4 adaptation = texture(adaptation_texture, uv);
    float reference = max(
        mix(adaptation.a, adapting_luminance_cd_m2, .35), 1e-6
    );
    vec3 normalized = retinal / (reference * 6.0 + 1e-5);

    // Chromatic adaptation acts on the cones, so it is applied to the cone
    // signal and not to the rod one. The rod channel below therefore keeps the
    // unadapted luminance: rods have a single photopigment and no gain control
    // that could discount a hue they cannot see in the first place.
    vec3 adapted = chromatically_adapt(normalized, adaptation.rgb);

    // Mesopic mixing. There is one rod photopigment, so rod vision carries no
    // hue: as the cone contribution falls the image collapses toward its
    // luminance — and with it, so does everything chromatic adaptation just
    // did. The Purkinje shift, which would additionally favour short
    // wavelengths, is not modelled here because it needs the tabulated
    // scotopic luminous efficiency V'(lambda), which this project does not
    // hold. Only the achromatic collapse is applied.
    float luminance = dot(normalized, PHOTOPIC_WEIGHTS);
    vec3 mesopic = mix(vec3(luminance), adapted, cone_fraction);

    // Von Kries gains are not gamut preserving; a saturated colour can leave
    // the cone round trip slightly negative.
    vec3 mapped = aces(max(mesopic, vec3(0.0)));
    frag_color = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
