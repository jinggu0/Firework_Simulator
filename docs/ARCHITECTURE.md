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
- Water spectrum: analytic deep-water phase evaluation at 60 Hz. Long waves
  displace the mesh while sub-grid ripples contribute to the fragment normal.
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

The water surface uses a 32-component, fetch-limited JONSWAP wind-wave
spectrum with deep-water dispersion. Wave energy is derived from the stored
October 5, 2024 wind observation and from an upwind ray traced through the
historical Han River mask, rather than from hand-authored amplitudes. At show
start, the imported 10 m wind is 0.471 m/s from 315 degrees. The river mask
gives a 2,094.7 m effective fetch at the scene origin, and the resulting
significant wave height is 0.0130 m.

The surface is split into a 1,200 x 900 m near grid and a 5,000 x 4,000 m far
grid. The far grid omits the near patch, preventing overlap and z-fighting.
Wavelengths too short for the mesh sampling density are evaluated only in the
fragment normal, avoiding aliased geometry while retaining fine reflected
highlights.

Water Fresnel reflection samples a half-resolution HDR planar-reflection pass
containing the astronomical sky, terrain, 2024 building geometry, and bridges.
The reflection camera is mirrored around the water datum and the projected
sample is perturbed by the same spectrum-derived normal used for shading.
This pass is cached at 30 Hz and scheduled away from a newly completed fluid
step to keep CPU/GPU spikes under the 60 Hz frame budget. Resolution scale and
update rate are explicit `RenderConfig` quality controls; the fidelity default
is half resolution at 30 Hz.

The water datum is the median DEM elevation beneath the river mask (5.01 m in
the source elevation model), not a verified October 5 gauge observation.
Exact tide-controlled river stage and local wakes remain calibration inputs
until timestamped gauge and vessel records are available.

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

The reproducible scene snapshot uses the OpenStreetMap historical state at
2024-10-05 10:20 UTC (19:20 KST) in the 37.515–37.545 N,
126.910–126.960 E bounding box. WGS84 coordinates are converted to local
East-Up-South metres before rendering. The shipped asset contains 99,570
building vertices, 6,330 bridge-deck vertices, 68,244 road vertices, and 5,469
park/grass/forest vertices. Explicit OSM heights are retained;
`building:levels` is converted at 3.2 m per level, and features with neither
value receive a clearly provisional 12 m fallback.

The snapshot now guarantees dated plan geometry but not survey accuracy. Known
gaps are untagged building heights, exact bridge superstructures, facade
materials, individual trees, and temporary event structures. The national
daily GIS building integration dataset and 1–5 m NGII DEM require authenticated
downloads and can replace these inputs when supplied. OSM attribution and
licensing are recorded in `assets/ATTRIBUTION.md`.

## Landmark geometry and facades

Static scene vertices carry world position, normal, surface kind, metric
surface coordinates, and a facade family. Buildings remain true 3D meshes in
the same terrain-displaced world and depth buffer as water, smoke, fireworks,
roads, and bridges. The existing free camera traverses that world with
W/A/S/D, Q/E vertical motion, Shift sprint, and mouse yaw/pitch; it does not
switch to a 2D map or billboard representation.

Historical OSM building and `building:part` ways select seven facade families:
generic office, blue curtain wall, 63 City gold glass, residential,
institutional stone, Parc.1 glass with red expressed structure, and hotel.
Metric facade coordinates produce anti-aliased floor slabs, bays, mullions,
balconies, deterministic occupied windows, view-dependent glass Fresnel, and
night emission in one batched building draw. This retains individual building
geometry without issuing a material draw call per building.

Named historical features receive targeted treatment. The 252 m 63 City mesh
uses sixteen tapered vertical bands to reproduce its narrowing upper
silhouette. IFC and FKI towers use a blue crystalline curtain-wall response.
The 322 m and 256 m Parc.1 towers expose red vertical columns and beams.
Residential buildings receive smaller floor spacing and balcony bands.
`min_height` is preserved for elevated parts, and OSM dome roof parts generate
curved 3D geometry, including the National Assembly dome data.

This is appearance reconstruction, not a claim of photogrammetric identity.
Facade panel dimensions are family calibrations; deterministic window
occupancy is not measured room-by-room lighting from October 5, 2024. Exact
signage, curtain-wall reflectance spectra, rooftop equipment, trees, temporary
event installations, and sub-metre silhouettes require licensed
timestamp-matched photography or survey/photogrammetry.

The Han River surface is clipped by a 1024 × 1024 geographic coverage mask
generated from OSM multipolygon relation 152336. Both outer riverbanks and
inner land/island rings are preserved. The mask covers a 5 × 4 km local area,
which gives roughly 4–5 m source-mask resolution before GPU interpolation.
This removes water from Yeouido park and other land while keeping the wave
simulation independent of shoreline complexity. The current land pass is a
flat provisional surface; surveyed terrain elevations remain required.

With this scene, water, and the initial firework active, the uncapped
development-machine measurement is approximately 1.7 ms per frame. This is a
pipeline headroom measurement, not a guarantee for the final volumetric and
reflection workload.

After geographic shoreline masking and the provisional land pass, the same
uncapped workload measures approximately 2.6 ms per frame (about 383 FPS).

## Terrain elevation

The initial terrain height map samples Mapzen/AWS Terrarium elevation tiles at
zoom 12 into a 512 × 512 local grid. The median DEM elevation under the Han
River mask is 5.01 m and becomes simulation `y=0`; stored land elevations are
relative to that datum. The current 5 × 4 km scene ranges from -1 m to about
87 m relative height.

Terrain is sampled in the GPU vertex stage. Buildings and bridge decks sample
the same texture at their footprint positions, preventing detached or buried
geometry. Finite differences over the terrain texture provide land normals for
night-sky shading.

This DEM is a regional bare-earth baseline, not a survey of individual
embankments. Its effective source resolution is roughly 30 m in this area and
cannot resolve narrow levees, stairs, roads, or the exact riverside viewing
surface. Those features require a higher-resolution Seoul/VWorld dataset or
manual photogrammetric calibration.

With terrain displacement and terrain-sampled buildings enabled, the uncapped
development-machine run measures approximately 1.94 ms per frame (about
515 FPS). Timing varies with driver scheduling, so the frame time—not the
uncapped FPS headline—is retained for later regression comparisons.

## Camera

The runtime camera is a six-degree-position, yaw/pitch free camera. Horizontal
movement follows camera heading while elevation remains independently
controllable. Input velocity uses frame-rate-independent exponential
acceleration and damping; diagonal input is normalized.

The view-projection matrix and camera position are refreshed for every render
pass each frame, keeping terrain, water Fresnel response, buildings, and
fireworks in one coordinate frame. Adding the dynamic camera path measures
about 1.69 ms per uncapped frame on the development machine.

## Event environment timeline

The official show interval is October 5, 2024, 19:20–20:30 KST. The runtime
clock starts at 19:20 and samples hourly Seoul station 47108 observations using
linear interpolation. The initial show sample is approximately 18.83 °C,
55% relative humidity, 1021.13 hPa, and 0.47 m/s horizontal wind after vector
interpolation.

Meteorological wind bearings describe the direction wind comes from. They are
converted to East-Up-South velocity vectors before interpolation, avoiding the
360°/0° discontinuity. Moist-air density is recalculated from temperature,
pressure, and humidity for ballistic drag.

The station wind is treated as the 10 m reference. A provisional neutral
vertical profile scales it to 1.4× at 100 m, with logarithmic interpolation by
shell or star altitude. This is more defensible than applying one velocity at
all heights, but it is not a measured event-night sounding and must be replaced
if local lidar or radiosonde data becomes available.

The environment timeline and altitude-dependent wind path bring the current
uncapped full-frame measurement to approximately 2.36 ms (about 424 FPS) on
the development machine.

## Astronomical illumination

Astronomy Engine supplies topocentric apparent Sun and Moon coordinates at the
event observer location. At 19:20 KST the apparent Sun is at azimuth 275.13°
and altitude -14.28°. The Moon is at azimuth 248.81°, altitude -2.56°, and
6.23% illuminated, so direct moonlight is zero during the show start.

The sky shader reconstructs a world-space ray from the free camera for every
pixel. It combines directional western twilight, a dark zenith, cloud-enhanced
urban horizon glow, and a photometrically scaled lunar disc when above the
horizon. The same ambient scale reaches water Fresnel reflection and land
lighting. Celestial positions update at 1 Hz; camera-dependent sky rays remain
60 Hz.

Twilight currently uses log-interpolated illuminance anchors at solar
altitudes 0°, -6°, -12°, and -18°. This captures the rapid event-time decay but
does not replace spectral atmospheric scattering. Absolute colour and
luminance still require calibration against a RAW/reference camera exposure.

The astronomical sky path measures approximately 1.90 ms per uncapped frame
(about 525 FPS) in the current development workload.

## Star exposure and bloom

Burning stars now render as exposure-integrated trails. Each GPU primitive
spans from the current physical position back along velocity by the configured
1/60 s shutter time. A geometry shader expands the projected segment into a
subpixel Gaussian ribbon, preserving direction and apparent angular width
without adding CPU particles.

Star power uses smooth ignition, a slowly shrinking luminous surface, and
smooth final extinction. Small deterministic combustion variation avoids
frame-random flicker while keeping replay output stable.

The linear HDR target is downsampled to half resolution, soft-thresholded, and
Gaussian blurred in two separable passes. Bloom is added before exposure and
ACES tone mapping, so overlapping stars naturally overexpose the burst core
instead of being clamped individually.

The new trail and bloom path measures about 2.21 ms per uncapped frame (about
453 FPS) with the 8,000-star development shell. The current shell still treats
all 8,000 particles as one star class; production shells need separate primary
stars, fine sparks, and embers with measured counts and burn laws.

## Post-blast smoke fluid

The first plume solver is a 64 x 36 vertical East-Up slice covering 640 x 360
m. It advances at a fixed 30 Hz independently of the display rate. Smoke mass
and temperature excess occupy cell centres; horizontal and vertical velocity
occupy their respective faces on a Marker-and-Cell grid.

Each step uses midpoint semi-Lagrangian backtracing, molecular diffusion,
Boussinesq thermal buoyancy, particulate loading, vorticity confinement, and a
Jacobi pressure projection. Automated checks require the projection to reduce
velocity divergence, preserve the injected smoke mass to discretization
tolerance, carry smoke downwind, and move a heated plume upward.

The shell's provisional 85 g burst charge emits 12% particulate mass. Of the
charge's provisional 3 MJ/kg chemical energy, 18% enters the slow plume as
residual heat. These parameters are separated from the solver because exact
values depend strongly on shell construction and composition and must be
replaced by event shell records or experiments.

The visible stars now contribute throughout their burn rather than creating
smoke only at the central burst. A provisional 0.78 kg star-composition mass is
divided across the shell's stars. Constant radial regression of a spherical
star gives the cumulative burned fraction `1 - (1 - t / burn_time)^3`.
Successive 30 Hz samples difference this value, so the emitted mass is
independent of display frame rate and remains exactly bounded by the initial
fuel mass. The default smoke yield is 24%; 6% of a provisional 4.2 MJ/kg
reaction energy enters the resolved plume as sensible heat.

Star products are accumulated over four 120 Hz ballistic steps and deposited
into the containing finite-volume smoke cell at 30 Hz. This nearest-cell source
is conservative and avoids fabricating sub-grid detail at the current 10 m
cell size. The GPU's linear field sampling and the following fluid advection
provide visual continuity. Stars that extinguish between fluid steps settle
their final partial burn before particle compaction, preventing lost mass or
energy.

The incompressible solver deliberately begins after the rapid compressible
shock phase. Applying incompressibility to the detonation itself would be
physically wrong; the separate blast-acoustics module supplies the short
pressure and acoustic transient.

The renderer reconstructs the slice into a 64 x 36 x 24 density and
temperature volume spanning 120 m in depth. A Gaussian depth profile is
discretely normalized so integrating every reconstructed column gives exactly
the original 2D smoke mass and sensible heat. This removes the former flat
plane without silently creating or destroying source material.

For every visible pixel, the GPU intersects the camera ray with the volume and
integrates up to 40 jittered samples. Each segment applies Beer-Lambert
extinction with a provisional 4.5 m2/g fine-aerosol mass-extinction
coefficient, front-to-back transmittance, temperature-dependent source colour,
and early termination at high opacity. Integration distance is measured in
metres, so oblique views naturally produce a longer optical path. Front/back
face selection also supports a camera located inside the volume.

The volume bounding geometry uses the opaque scene depth buffer, so terrain or
buildings entirely in front of the plume occlude it. Geometry entering the
volume after its front boundary cannot yet terminate an individual ray; exact
interior occlusion requires sampling a copied depth texture or ray tracing the
scene geometry.

This is a view-dependent 3D participating medium but not yet a 3D fluid solve.
The Gaussian depth assumption is symmetric and cannot produce independent
crosswind vortices. Production quality still requires a sparse 3D GPU MAC
grid, combustion-species calibration, multiple scattering, and direct light
injection from individual burning stars.

The volume texture retains its physical domain, but its rendered proxy is
contracted after every fluid update to the conservative AABB of optically
active voxels plus one-cell interpolation margin. Empty portions of the
640 x 360 x 120 m domain therefore create neither fragments nor 40-sample ray
loops. Texture coordinates still reference the complete domain, so this is
empty-space skipping rather than a change to density, temperature, or optical
path length.

## p95 bottleneck analysis

`python -m tools.profile_runtime --frames 120` runs three repeatable GL
timestamp-query cases with 8,000 held stars: full moving-camera rendering,
moving-camera rendering without smoke, and full static-camera rendering.
After active-volume contraction, the full moving case measures 4.925 ms GPU
p95 and 6.267 ms CPU-submit p95. The no-smoke GPU p95 is 4.878 ms and the
static-camera GPU p95 is 4.640 ms. The small residuals show that neither the
contracted plume nor the camera-invalidated planar reflection is now the
dominant visual p95 cost. Before contraction, the equivalent full visual GPU
p95 was 7.544 ms.

The reflection texture is redrawn at up to 30 Hz while the camera moves, but a
stationary camera reuses static city geometry and refreshes only for the
one-second sky-lighting invalidation. Animated water normals continue to
distort the cached radiance every 60 Hz frame. Far facades also replace
per-window occupancy hashes with filtered average emission beyond 1,600 m,
limiting shader aliasing and work without replacing geometry with billboards.

A stricter 240-frame integrated run calls GPU finish every frame while moving
the camera and updating 8,000 stars, 120 Hz ballistics, 30 Hz fluid, smoke
uploads, water, reflection, bloom, and tone mapping. It measures 13.325 ms mean
(75.0 FPS), 20.122 ms p95, and 21.936 ms p99. Physics averages 6.634 ms and the
visual section 6.690 ms. The mean 60 FPS budget is met, but the integrated p95
still exceeds 16.67 ms because the CPU fluid solve lands on alternating
display frames.

The next latency work is therefore ordered as follows: migrate pressure
projection and advection to a GPU MAC grid without reducing physical
iterations; double-buffer volume reconstruction/uploads so they overlap the
previous frame; then add depth-pyramid interior smoke occlusion. Dynamic
resolution, if required, applies only to volumetric radiance and reflection,
never to trajectory, terrain, building geometry, or the fixed physics clocks.

## Delayed blast acoustics

Light propagation is treated as instantaneous at scene scale, while every
burst now creates a separately propagating pressure event. Moist-air sound
speed is calculated from temperature, pressure, and relative humidity using
the Buck saturation-vapour-pressure relation and the moist-air gas constant.
Wind projected along the source-to-listener ray changes effective propagation
speed. The listener follows the free camera, so moving the camera changes both
arrival time and stereo direction.

The first subsonic radius is found from the Sedov-Taylor strong-shock solution
`R = beta (E t^2 / rho)^(1/5)` with `beta = 1.033`. A provisional 32% of burst
chemical energy drives this early blast. After shock velocity falls to local
sound speed, the remaining distance propagates acoustically. This avoids both
instant sound and the physically invalid use of the incompressible smoke
solver for the detonation.

Far-field acoustic energy is provisionally 1% of burst chemical energy.
Spherical spreading, duration broadening with range, air absorption, density,
and effective sound speed determine RMS pressure and SPL. At the default
camera, the development shell's approximately 159 m burst altitude predicts
about 0.787 s flash-to-boom delay, 100.5 dB RMS SPL, and 3.00 Pa peak
pressure. These are model outputs, not
measurements of the exact 2024 shell.

Audio renders at 48 kHz as a Friedlander pressure impulse plus a deterministic
32-180 Hz decaying tail. Equal-power stereo panning follows camera-relative
source direction. The pressure-to-digital mapping reserves 20 Pa as full scale
to prevent numerical clipping, but consumer speaker gain is not calibrated;
displayed SPL remains the physical prediction while playback loudness remains
device-dependent.

Band-limited tail noise is prepared once. Final per-arrival synthesis costs
about 4.7 ms on the development machine. It is dispatched to one audio worker
when the wavefront is within 120 ms, then the prepared buffer is released on
the exact 120 Hz arrival step. This prevents an audio FFT or PCM conversion
from stalling the render frame. Remaining limitations are shell directivity,
frequency-dependent ISO
9613 absorption, water/ground reflection, bridge and building echoes,
temperature-gradient refraction, and measured event-shell acoustic energy.

The prepared audio buffer adds no synchronous waveform synthesis to its
arrival frame. End-to-end timing is recorded with the current smoke renderer
above; the remaining frame-time tail is dominated by the CPU fluid quality
tier and reinforces the planned migration to a 3D GPU solver.

## Required reference datasets

- October 5, 2024 hourly and, where possible, sub-hourly weather observations
- Launch-barge positions, firing timeline, shell types, and calibrated video
- Terrain/building/bridge geometry and nighttime emissive references
- Han River wind-wave observations or a defensible wind-driven spectrum
- Solar/lunar ephemeris, cloud cover, atmospheric visibility, and sky luminance
- Camera position, focal length/FOV, exposure, white balance, and sensor curve
