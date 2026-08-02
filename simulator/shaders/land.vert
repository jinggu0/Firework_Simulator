#version 330
in vec2 in_xz;
uniform mat4 view_projection;
uniform sampler2D terrain_height;
uniform vec4 terrain_bounds;
out vec3 world_position;
out vec3 world_normal;
void main() {
    ivec2 dimensions = textureSize(terrain_height, 0);
    vec2 dimension_f = vec2(dimensions);
    vec2 grid_size = max(dimension_f - 1.0, vec2(1.0));
    vec2 geographic_uv = clamp(
        (in_xz - terrain_bounds.xy) / (terrain_bounds.zw - terrain_bounds.xy),
        0.0, 1.0
    );
    // The CPU height field includes both geographic bounds as texel centres.
    // Half-texel addressing keeps navigation collision and rendering on the
    // same bilinear surface instead of shifting the GPU by half a DEM cell.
    vec2 uv = (geographic_uv * grid_size + 0.5) / dimension_f;
    vec2 metres_per_texel = (terrain_bounds.zw - terrain_bounds.xy)
                          / grid_size;
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
