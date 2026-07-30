#version 330
uniform sampler2D source_texture; uniform vec2 direction;
in vec2 uv; out vec4 frag_color;
void main() {
    vec2 texel = 1.0 / vec2(textureSize(source_texture, 0));
    vec3 color = texture(source_texture, uv).rgb * .227027;
    color += texture(source_texture, uv + direction * texel * 1.384615).rgb * .316216;
    color += texture(source_texture, uv - direction * texel * 1.384615).rgb * .316216;
    color += texture(source_texture, uv + direction * texel * 3.230769).rgb * .070270;
    color += texture(source_texture, uv - direction * texel * 3.230769).rgb * .070270;
    frag_color = vec4(color, 1.0);
}
