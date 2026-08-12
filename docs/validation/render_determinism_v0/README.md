# V0 scene residual shader instrumentation

This validation step localises the remaining scene-owned determinism residual
without changing the shipped fragment shader. `tools/probe_render_determinism.py`
temporarily substitutes one diagnostic expression while the OpenGL program is
created, restores the normal shader loader immediately afterwards, freezes
simulation time, and compares the known half-open GL rectangle
`(598, 380, 600, 384)`.

The production baseline remains an eight-pixel block with a maximum linear-HDR
difference near `1.204e-3`. Raw attributes and independently exposed shading
inputs are stable, but the apparent first varying intermediate moves when the
diagnostic return point or helper-function boundary changes. Five plausible
runtime changes were tested and rejected because none removed the final
residual over long repeated captures. The shipped `scene.frag` therefore stays
unchanged.

The measurements were made on Intel Arc 140V, OpenGL 4.3, driver
32.0.101.8626. The supported conclusion is compiler/driver optimisation
sensitivity, not a confirmed defect in a particular BRDF term. Closing the
mechanism requires either the same probe on another driver/GPU or a minimal
standalone shader reproducer.

V0-11 adds that standalone reproducer in
`tools/probe_facade_shader_reproducer.py`. It uses the captured input ranges
from the known pixels and cumulatively restores the view vector, Fresnel,
environment reflection, helper boundary, zero-count light loops, and final
emission in a 2x4 RGBA16F target. On the Intel Arc 140V driver above, all eight
stages produced exactly one state over 4,096 draws each. A same-session scene
probe still varied in 27 of 95 comparisons, so the negative standalone result
is meaningful: the remaining trigger needs actual pass, interpolation,
raster/depth, viewport-position, or whole-program state.

The machine-readable result is `standalone_intel_arc_140v.json`. The next
isolation ladder should add the 1280x720 viewport and original pixel location,
actual facade triangle interpolation, depth attachment/test, and the complete
scene program in that order.

Representative commands:

```powershell
python -m tools.probe_render_determinism --view water_reflection --iterations 96 --region 598,380,600,384
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term raw_normal
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term radiance_environment
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term radiance_after_dynamic --static-light-count 0
python -m tools.probe_facade_shader_reproducer --iterations 4096 --output docs/validation/render_determinism_v0/standalone_intel_arc_140v.json
```

Rows are OpenGL rows counted from the bottom. Image-order captures use
`height - 1 - row` instead.
