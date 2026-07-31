#version 330
// Local retinal adaptation state, advanced one frame at a time.
//
// A single global exposure cannot produce an afterimage, because an afterimage
// is local: the patch of retina a burst fell on stays bleached while its
// surroundings do not. This buffer holds that per-region state.
uniform sampler2D hdr_texture;
uniform sampler2D previous_adaptation;
uniform float light_response;   // 1 - exp(-dt / light adaptation time)
uniform float dark_response;    // 1 - exp(-dt / dark adaptation time)
uniform float pooling_lod;      // mip level standing in for the pooling area
in vec2 uv;
out vec4 frag_color;
void main() {
    // Adaptation pools over roughly a degree of visual angle rather than a
    // single receptor, so the current level is read from a reduced mip.
    vec3 pooled = textureLod(hdr_texture, uv, pooling_lod).rgb;
    float current = dot(pooled, vec3(.2126, .7152, .0722));
    float previous = texture(previous_adaptation, uv).r;
    // Asymmetric: bleaching is fast, recovery is slow. This is what leaves a
    // dark afterimage where a burst was once it has faded.
    float response = current > previous ? light_response : dark_response;
    frag_color = vec4(max(previous + (current - previous) * response, 0.0),
                      0.0, 0.0, 1.0);
}
