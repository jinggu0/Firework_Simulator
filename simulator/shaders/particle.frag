#version 330
in vec3 trail_color; in float trail_power; in vec2 trail_coordinate;
out vec4 frag_color;
void main() {
    float cross_section = exp(-trail_coordinate.y * trail_coordinate.y * 3.4);
    float longitudinal = mix(.62, 1.0, trail_coordinate.x);
    float radiance = cross_section * longitudinal * trail_power * .0045;
    frag_color = vec4(trail_color * radiance, 1);
}
