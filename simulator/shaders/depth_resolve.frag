#version 330
// Resolve a multisampled depth buffer into the sampleable single-sample one.
//
// glBlitFramebuffer refuses a depth blit between buffers of different sample
// counts, and the ambient occlusion, haze and smoke passes all read scene
// depth as a texture. So depth is resolved by drawing over the frame.
//
// Sample zero is taken rather than an average. A depth buffer does not store a
// linear quantity, and averaging the two sides of a silhouette would place a
// surface where there is nothing; sample zero is always a depth some part of
// the pixel really has.
//
// It is not the depth a single-sample render would have written. Sample zero
// sits at a standard multisample position, not at the pixel centre, so on a
// slope the two differ by the depth gradient across that offset — measured at
// up to 0.00225 of the depth range on the facade_landmark view. The passes
// that read this buffer reconstruct world position from it, so they see the
// scene shifted by that much inside the pixel, never a surface that is absent.
uniform sampler2DMS depth_multisample;
in vec2 uv;
void main() {
    gl_FragDepth = texelFetch(
        depth_multisample, ivec2(gl_FragCoord.xy), 0
    ).r;
}
