#version 330
in vec3 in_position;
uniform mat4 view_projection;
out vec3 surface_position;
void main() {
    surface_position = in_position;
    gl_Position = view_projection * vec4(in_position, 1.0);
}
