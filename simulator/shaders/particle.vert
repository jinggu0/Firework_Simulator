#version 330
in vec3 in_position; in vec3 in_trail_start; in vec3 in_color; in float in_power;
uniform mat4 view_projection; uniform float reflection; uniform float time_s;
uniform vec3 camera_position;
out vec4 trail_start_clip; out vec3 particle_color; out float particle_power;
#include "air_extinction.glsl"
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
    // Stars are additive over a scene the deferred haze pass has already
    // closed, so they take their own path transmittance here and no airlight:
    // the airlight along the line of sight is already in the pixel behind
    // them. Folding it into the colour rather than a new varying keeps the
    // geometry stage's interface unchanged, and is exact because the fragment
    // radiance is linear in this colour.
    //
    // The mirrored draw is the specular path camera -> water -> star, whose
    // length the mirrored image point reproduces exactly; taking heights as
    // magnitudes inside the profile integral keeps that path above the datum.
    float path_m = distance(camera_position, p);
    particle_color = in_color
                   * air_transmittance(camera_position.y, p.y, path_m);
    particle_power = in_power * (reflection > .5 ? .13 : 1.0);
}
