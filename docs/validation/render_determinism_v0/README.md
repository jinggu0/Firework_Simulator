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

Representative commands:

```powershell
python -m tools.probe_render_determinism --view water_reflection --iterations 96 --region 598,380,600,384
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term raw_normal
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term radiance_environment
python -m tools.probe_render_determinism --view water_reflection --iterations 192 --scene-term radiance_after_dynamic --static-light-count 0
```

Rows are OpenGL rows counted from the bottom. Image-order captures use
`height - 1 - row` instead.
