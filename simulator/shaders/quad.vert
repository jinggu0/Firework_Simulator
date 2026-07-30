#version 330
in vec2 in_position;
out vec2 uv;
void main() { uv = in_position * .5 + .5; gl_Position = vec4(in_position, 0, 1); }
