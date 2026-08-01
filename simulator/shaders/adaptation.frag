#version 330
// Retinal adaptation state, advanced one frame at a time.
//
// Two states share this buffer because they share a ping-pong and a time step,
// but they are different mechanisms on different scales:
//
//   .a   local adapting luminance. A single global exposure cannot produce an
//        afterimage, because an afterimage is local: the patch of retina a
//        burst fell on stays bleached while its surroundings do not.
//
//   .rgb the adapting white, near-global and much slower. Chromatic adaptation
//        discounts the illuminant, and the illuminant is a property of the
//        whole field rather than of one patch, so it is pooled from the top of
//        the mip chain. Stored normalised to unit luminance: brightness is the
//        other channel's job and the two must not interfere.
uniform sampler2D hdr_texture;
uniform sampler2D previous_adaptation;
uniform float light_response;       // 1 - exp(-dt / light adaptation time)
uniform float dark_response;        // 1 - exp(-dt / dark adaptation time)
uniform float chromatic_response;   // 1 - exp(-dt / chromatic adaptation time)
uniform float pooling_lod;          // mip level standing in for the pooling area
uniform float global_pooling_lod;   // mip level that covers the whole field
in vec2 uv;
out vec4 frag_color;
const vec3 PHOTOPIC_WEIGHTS = vec3(.2126, .7152, .0722);
void main() {
    // Adaptation pools over roughly a degree of visual angle rather than a
    // single receptor, so the current level is read from a reduced mip.
    vec3 pooled = textureLod(hdr_texture, uv, pooling_lod).rgb;
    float current = dot(pooled, PHOTOPIC_WEIGHTS);
    vec4 previous = texture(previous_adaptation, uv);
    // Asymmetric: bleaching is fast, recovery is slow. This is what leaves a
    // dark afterimage where a burst was once it has faded.
    float response = current > previous.a ? light_response : dark_response;
    float luminance = max(previous.a + (current - previous.a) * response, 0.0);

    // Sampled at a fixed point, not at this texel's uv: the adapting white is
    // one value for the whole field, so every texel must arrive at the same
    // one rather than happen to because the mip is a single texel.
    vec3 field = textureLod(hdr_texture, vec2(0.5), global_pooling_lod).rgb;
    float field_luminance = dot(field, PHOTOPIC_WEIGHTS);
    // A black field carries no chromaticity to adapt to, so the observer holds
    // the white they had rather than diverging.
    vec3 target = field_luminance > 1e-9
                ? field / field_luminance
                : previous.rgb;
    // Symmetric, unlike the luminance channel: no directional asymmetry is
    // reported for chromatic adaptation. Both endpoints carry unit luminance
    // and luminance is linear, so the interpolation keeps unit luminance too.
    vec3 white = previous.rgb + (target - previous.rgb) * chromatic_response;
    // Renormalised every step. The state is stored as half floats and read
    // back into the next step, so a systematic rounding of a few 1e-4 is
    // amplified by 1 / chromatic_response — about eighteen — and the white
    // drifts off unit luminance. Measured at 3.7e-3 before this line existed.
    white /= max(dot(white, PHOTOPIC_WEIGHTS), 1e-6);
    frag_color = vec4(white, luminance);
}
