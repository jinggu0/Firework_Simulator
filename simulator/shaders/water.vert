#version 330
in vec2 in_xz;
uniform mat4 view_projection;
uniform vec4 waves[32];
uniform float phases[32];
uniform float time_s;
uniform float choppiness;
out vec3 world_position;
out vec3 world_normal;
out vec4 reflection_clip;
uniform mat4 reflection_view_projection;
void main() {
    vec2 displaced_xz = in_xz;
    float height = 0.0;
    vec2 gradient = vec2(0.0);
    for (int i = 0; i < 32; ++i) {
        vec2 direction = waves[i].xy;
        float k = waves[i].z;
        float amplitude = waves[i].w;
        float omega = sqrt(9.80665 * k);
        float theta = k * dot(direction, in_xz) - omega * time_s + phases[i];
        float wavelength = 6.28318530718 / k;
        float geometry_weight = smoothstep(1.8, 7.0, wavelength);
        height += amplitude * sin(theta) * geometry_weight;
        gradient += amplitude * k * direction * cos(theta) * geometry_weight;
        displaced_xz += choppiness * amplitude * direction * cos(theta)
                      * geometry_weight;
    }
    world_position = vec3(displaced_xz.x, height, displaced_xz.y);
    world_normal = normalize(vec3(-gradient.x, 1.0, -gradient.y));
    reflection_clip = reflection_view_projection
                    * vec4(world_position, 1.0);
    gl_Position = view_projection * vec4(world_position, 1.0);
}
