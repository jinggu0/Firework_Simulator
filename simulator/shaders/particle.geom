#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
in vec4 trail_start_clip[];
in vec3 particle_color[];
in float particle_power[];
out vec3 trail_color;
out float trail_power;
out vec2 trail_coordinate;
uniform vec2 viewport_size;
void emit_trail_vertex(vec4 clip, vec2 offset_ndc, vec2 coordinate) {
    gl_Position = clip;
    gl_Position.xy += offset_ndc * clip.w;
    trail_color = particle_color[0];
    trail_power = particle_power[0];
    trail_coordinate = coordinate;
    EmitVertex();
}
void main() {
    vec4 start_clip = trail_start_clip[0];
    vec4 end_clip = gl_in[0].gl_Position;
    if (start_clip.w <= 0.0 || end_clip.w <= 0.0) return;
    vec2 start_ndc = start_clip.xy / start_clip.w;
    vec2 end_ndc = end_clip.xy / end_clip.w;
    vec2 delta_pixels = (end_ndc - start_ndc) * viewport_size * .5;
    float length_pixels = length(delta_pixels);
    vec2 direction = length_pixels > .1
                   ? delta_pixels / length_pixels : vec2(0.0, 1.0);
    vec2 perpendicular = vec2(-direction.y, direction.x);
    float half_width = clamp(1.15 + 900.0 / max(end_clip.w, 1.0), 1.2, 5.5);
    vec2 offset = perpendicular * half_width * 2.0 / viewport_size;
    if (length_pixels < 1.0) {
        vec2 extension = direction * (1.0 - length_pixels) * 2.0 / viewport_size;
        start_clip.xy -= extension * start_clip.w;
    }
    emit_trail_vertex(start_clip, -offset, vec2(0.0, -1.0));
    emit_trail_vertex(start_clip,  offset, vec2(0.0,  1.0));
    emit_trail_vertex(end_clip,   -offset, vec2(1.0, -1.0));
    emit_trail_vertex(end_clip,    offset, vec2(1.0,  1.0));
    EndPrimitive();
}
