#version 330
in vec2 in_xz;
uniform mat4 view_projection;
uniform sampler2D terrain_height;
uniform vec4 terrain_bounds;
out vec3 world_position;
out vec3 world_normal;
void main() {
    vec2 uv = (in_xz - terrain_bounds.xy) / (terrain_bounds.zw - terrain_bounds.xy);
    ivec2 dimensions = textureSize(terrain_height, 0);
    vec2 metres_per_texel = (terrain_bounds.zw - terrain_bounds.xy)
                          / vec2(dimensions);
    float height = texture(terrain_height, uv).r;
    float left = textureOffset(terrain_height, uv, ivec2(-1, 0)).r;
    float right = textureOffset(terrain_height, uv, ivec2(1, 0)).r;
    float back = textureOffset(terrain_height, uv, ivec2(0, -1)).r;
    float front = textureOffset(terrain_height, uv, ivec2(0, 1)).r;
    world_normal = normalize(vec3(
        -(right - left) / (2.0 * metres_per_texel.x),
        1.0,
        -(front - back) / (2.0 * metres_per_texel.y)
    ));
    world_position = vec3(in_xz.x, height - 0.04, in_xz.y);
    gl_Position = view_projection * vec4(world_position, 1.0);
}
