#version 330
uniform sampler2D hdr_texture;
in vec2 uv; out vec4 frag_color;
void main() {
    vec3 color = texture(hdr_texture, uv).rgb;
    float brightness = max(max(color.r, color.g), color.b);
    float soft = clamp((brightness - .035) / .10, 0.0, 1.0);
    frag_color = vec4(color * soft, 1.0);
}
