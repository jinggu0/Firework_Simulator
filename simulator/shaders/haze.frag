#version 330
// Aerial perspective over the opaque scene, as a deferred composite.
//
// Koschmieder's relation, which is what "visibility" means, is
//
//     L = L_object * T + L_air * (1 - T),
//
// with L_air the horizon sky radiance and T the transmittance over the path.
// The airlight texture is that horizon radiance, rendered by the sky pass
// itself, so an object at the visibility range sits at the 2% contrast the
// definition names, and one at infinity is exactly the sky.
//
// The two terms are applied as two draws of this shader because the blend
// pipeline carries one scalar alpha and the transmittance is per channel:
// stage 0 multiplies the target by T, stage 1 adds the airlight. Stage 0 must
// run first — the multiply would otherwise scale the airlight it just added.
// Being deferred makes it independent of the order the opaque passes drew in,
// which a blend on the geometry itself would not be.
in vec2 uv;
out vec4 frag_color;
uniform sampler2D scene_depth;
uniform sampler2D airlight;
uniform mat4 inverse_view_projection;
uniform vec3 camera_position;
uniform float haze_stage;
#include "air_extinction.glsl"

void main() {
    float depth = texelFetch(scene_depth, ivec2(gl_FragCoord.xy), 0).r;
    // Nothing was drawn here, so the pixel is sky: it already carries the
    // airlight of an infinite path and must not be attenuated again.
    if (depth >= 1.0) discard;

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 homogeneous = inverse_view_projection
                     * vec4(ndc, depth * 2.0 - 1.0, 1.0);
    vec3 world_position = homogeneous.xyz / homogeneous.w;
    vec3 offset = world_position - camera_position;
    vec3 transmittance = air_transmittance(
        camera_position.y, world_position.y, length(offset)
    );
    vec3 scattered_fraction = 1.0 - transmittance;
    frag_color = vec4(
        mix(
            scattered_fraction,
            texture(airlight, uv).rgb * scattered_fraction,
            haze_stage
        ),
        1.0
    );
}
