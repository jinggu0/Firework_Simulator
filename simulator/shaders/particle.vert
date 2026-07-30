#version 330
in vec3 in_position; in vec3 in_trail_start; in vec3 in_color; in float in_power;
uniform mat4 view_projection; uniform float reflection; uniform float time_s;
out vec4 trail_start_clip; out vec3 particle_color; out float particle_power;
void main() {
    vec3 p = in_position;
    vec3 start = in_trail_start;
    if (reflection > .5) {
        p.y = -p.y;
        start.y = -start.y;
        p.x += sin(p.x * .12 + time_s * 2.1) * (.7 + .012 * abs(p.y));
        start.x += sin(start.x * .12 + time_s * 2.1)
                 * (.7 + .012 * abs(start.y));
    }
    gl_Position = view_projection * vec4(p, 1);
    trail_start_clip = view_projection * vec4(start, 1);
    particle_color = in_color;
    particle_power = in_power * (reflection > .5 ? .13 : 1.0);
}
