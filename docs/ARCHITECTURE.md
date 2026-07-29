# Yeouido Fireworks Simulator

## Fidelity contract

The simulator keeps physical state in SI units and linear-light rendering
values. Artistic constants are permitted only as explicitly identified
calibration values, so measured weather, shell, terrain, camera, and spectral
data can replace them without redesigning the engine.

## Frame model

- Display: 60 Hz target, 16.67 ms budget.
- Ballistics and burning stars: fixed 120 Hz.
- Smoke velocity/density/temperature fields: planned fixed 30 Hz with temporal
  interpolation.
- Water spectrum: analytic deep-water phase evaluation at 60 Hz; a future FFT
  spectrum will update displacement at 15–30 Hz with 60 Hz shading.
- Rendering: linear RGBA16F followed by camera exposure and tone mapping.

Simulation results must not depend on display frame rate. Random generation uses
a recorded seed so a performance capture can be compared against reference
footage frame by frame.

## Coordinate system

The runtime coordinate system is right-handed and measured in metres:

- +X: local east
- +Y: up
- +Z: local south

Geospatial source data will be transformed from WGS84 to a local ENU origin
near the 2024 launch barges. Camera calibration and launch coordinates belong
in scenario data rather than source code.

## Render passes

1. Sky radiance and astronomical lighting
2. Terrain, bridges, buildings, and emissive city lights
3. Spectral water displacement and reflection
4. Firework shells, stars, sparks, and trails
5. Participating-media smoke and light scattering
6. HDR bloom, camera exposure, sensor response, and tone mapping

The water surface now uses a discrete directional wind-wave spectrum with
deep-water dispersion. Wave energy is derived from wind speed, direction, and
fetch rather than hand-authored amplitudes. The current reflected firework
displacement remains a bounded prototype; it will be replaced by reflected
scene radiance sampled at the displaced water normal.

The default provisional wind input produces a significant wave height of about
0.05 m. It is a safe development condition, not a claim about the event night.
The value must be regenerated from verified October 5, 2024 observations.

## Performance rules

- No per-particle Python objects.
- Particle state uses contiguous structure-of-arrays storage.
- Large visual particle sets migrate to compute shaders before production.
- Reflections use perceptual importance and temporal caching.
- Every feature ships with a GPU/CPU timing measurement and a scalable quality
  control; physical trajectory parameters are never reduced for performance.

On the initial development laptop, the terrain-free water and firework vertical
slice renders uncapped at about 2.0 ms per frame. V-synced measurements report
approximately 58 FPS because of the display/SDL swap interval; the uncapped
measurement confirms substantial GPU/CPU headroom above the 60 FPS workload.

## Initial geospatial scene

The first reproducible scene snapshot is derived from OpenStreetMap geometry in
the 37.515–37.545 N, 126.910–126.960 E bounding box. WGS84 coordinates are
converted to local East-Up-South metres before rendering. It currently contains
about 93,500 building vertices and 6,100 bridge-deck vertices. Explicit OSM
heights are retained; `building:levels` is converted at 3.2 m per level, and
features with neither value receive a clearly provisional 12 m fallback.

The snapshot provides measured plan geometry, not a final October 2024 scene.
Known gaps are terrain elevation, exact bridge structures, facade materials,
shoreline clipping, vegetation, and dated construction state. These must be
validated against Seoul/VWorld data and event footage. OSM attribution and
licensing are recorded in `assets/ATTRIBUTION.md`.

With this scene, water, and the initial firework active, the uncapped
development-machine measurement is approximately 1.7 ms per frame. This is a
pipeline headroom measurement, not a guarantee for the final volumetric and
reflection workload.

## Required reference datasets

- October 5, 2024 hourly and, where possible, sub-hourly weather observations
- Launch-barge positions, firing timeline, shell types, and calibrated video
- Terrain/building/bridge geometry and nighttime emissive references
- Han River wind-wave observations or a defensible wind-driven spectrum
- Solar/lunar ephemeris, cloud cover, atmospheric visibility, and sky luminance
- Camera position, focal length/FOV, exposure, white balance, and sensor curve
