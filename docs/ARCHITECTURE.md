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

The spectrum is no longer frozen after startup. Every two seconds the current
10 m wind vector and river-mask fetch generate a target spectrum. Component
directions, wave numbers, and energy relax toward it with a 180 s wind-sea
response time while phases remain continuous, preventing a non-physical pop
when the weather timeline crosses a sample. Lunar position affects incident
radiance and the reflected direction. It does not create local short waves:
the lunar equilibrium tide is not an appropriate forcing term for the visible
wind-wave band on this regulated freshwater reach.

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

The default plume solver is now a depth-resolved 32 x 24 x 10 East-Up-North
grid covering 400 x 300 x 120 m around the launch site. Its approximately
12 m cells are an intentionally bounded launch-domain brick: the scale resolves
the event's large coherent plume motion while keeping the complete coupled
simulation inside a laptop's 16.67 ms frame budget. Smoke mass and temperature
excess occupy cell centres; all three velocity components occupy their
respective faces on a three-dimensional Marker-and-Cell grid.

Each step uses midpoint semi-Lagrangian backtracing, molecular diffusion,
Boussinesq thermal buoyancy, particulate loading, vorticity confinement, and a
Jacobi pressure projection. Automated checks require the projection to reduce
velocity divergence, preserve the injected smoke mass to discretization
tolerance, carry smoke through the measured crosswind direction, and move a
heated plume upward.

OpenGL 4.3 compute shaders advance the 3D grid at 30 Hz while rendering and
camera response remain 60 Hz. Density/temperature use RG32F 3D textures;
the staggered U, V, and W fields, pressure, and divergence use R32F 3D
textures. A 3D curl field supplies vorticity confinement. Twenty-four Jacobi
iterations per step are evaluated as twelve ping-pong dispatches by
algebraically expanding two iterations in each compute invocation. This keeps
the same 720 pressure iterations per simulated second as the fallbacks.
Pressure is warm-started and its arbitrary gauge is pinned to prevent
constant-mode drift. GPU state stays resident; the CPU uploads only a compact
conservative source field, and readback is reserved for diagnostics.

The application requests OpenGL 4.3 first. If compute shaders are unavailable,
it automatically selects the existing OpenGL 3.3 64 x 36 two-dimensional MAC
solver at 60 Hz and twelve pressure iterations per step. If GPU fluid creation
also fails, the 30 Hz NumPy solver with twenty-four iterations remains the
last compatibility path. All three therefore retain the same pressure-work
rate and physical coefficients rather than changing the model to recover FPS.

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
into the containing 3D finite-volume cell at 30 Hz. `bincount` reductions
conserve all accepted particulate mass and sensible heat without an
8,000-particle GPU dispatch. The spherical burst source is a volume-normalized
3D Gaussian rather than the former depth profile. The temperature field is
bounded at the configured 850 K excess, matching the CPU solver's unresolved
sub-grid energy policy. GPU trilinear sampling and subsequent advection provide
visual continuity without fabricating smaller simulated eddies. Stars that
extinguish between fluid steps settle their final partial burn before particle
compaction, preventing lost mass or energy.

The incompressible solver deliberately begins after the rapid compressible
shock phase. Applying incompressibility to the detonation itself would be
physically wrong; the separate blast-acoustics module supplies the short
pressure and acoustic transient.

The renderer directly samples the resident two-channel 3D state texture. Every
ray-march sample therefore sees the independently advected crosswind density
and temperature instead of reconstructing depth from a symmetric Gaussian.
The OpenGL 3.3 and CPU fallbacks retain the normalized analytic-depth path so
older laptops still render a volumetric plume.

For every visible pixel, the GPU intersects the camera ray with the volume and
integrates up to 40 jittered samples. Each segment applies Beer-Lambert
extinction with a provisional 4.5 m2/g fine-aerosol mass-extinction
coefficient, front-to-back transmittance, temperature-dependent source colour,
and early termination at high opacity. Integration distance is measured in
metres, so oblique views naturally produce a longer optical path. Front/back
face selection also supports a camera located inside the volume.

The opaque pass now writes a sampleable full-resolution depth texture rather
than an inaccessible renderbuffer. Before smoke compositing, the renderer
rebinds the HDR colour target without that depth attachment; this avoids a
texture feedback loop while preserving the completed terrain, building, and
water depths. The smoke fragment reconstructs the opaque world position with
the inverse view-projection matrix and clamps its ray exit distance to 0.20 m
before that surface. Geometry in front of the volume rejects the ray,
geometry inside it terminates integration exactly, and geometry behind it does
not alter the plume. Firework stars remain emissive participants rather than
opaque depth writers.

A depth pyramid is deliberately not built for this operation. The ray marcher
needs only the nearest opaque hit for its pixel, so one exact full-resolution
fetch is both cheaper and more accurate than an extra hierarchy construction
pass. Hierarchical depth remains useful only if a future adaptive marcher
performs multiple screen-space visibility queries per ray.

The 3D field retains its complete physical domain, but its rendered proxy is
restricted to conservative source bounds and then expanded each step by the
maximum modeled wind and buoyant-growth envelope. Empty portions of the
400 x 300 x 120 m domain therefore create neither fragments nor 40-sample ray
loops. Texture coordinates still reference the complete 3D domain, so this is
empty-space skipping rather than a change to density, temperature, or optical
path length. Future production calibration still requires adaptive sparse
bricks outside the launch domain, combustion-species measurements, multiple
scattering, and direct shadowing from individual burning stars.

## Radiometric light sources and physical camera

All render-light inputs are classified as electric LED, combustion, or
celestial sources. Electric windows begin with a 35 W input fixture and pass
through driver, electrical injection, internal quantum, package extraction,
phosphor, and luminaire optical efficiencies. Interior utilization, visible
window transmittance, emitting area, and Lambertian exitance then produce
window radiance in W/(m2 sr). The default chain produces 14.93 W of luminaire
radiant power, 4,478 lm, and 0.1746 W/(m2 sr) at the window. These are
calibratable fixture assumptions, not a claim that every 2024 window used that
lamp or was occupied.

A burning star no longer receives an arbitrary fixed brightness. Its assigned
composition mass times 4.2 MJ/kg gives chemical energy; a provisional measured
fireball radiative fraction of 0.15 partitions radiant energy. The normalized
ignition, regressing-surface, and extinction curve is integrated, so peak
radiant power is solved from the required time integral. The small
deterministic burn fluctuation has a 0.97 mean and is included in that energy
normalization.

Each frame, 8,000 star sources are reduced to at most eight octant clusters.
Power-weighted position and spectral RGB are computed with vectorized
histograms, and total radiant power is conserved exactly to floating-point
precision. The fixed light count bounds fragment cost. Building irradiance is
`P exp(-sigma d)/(4 pi d^2)` and returns Lambertian reflected radiance
`albedo E cos(theta)/pi`. Water applies the same irradiance to a GGX
microfacet BRDF with the Fresnel reflectance of water. This makes a burst
illuminate actual building normals and produce view-dependent highlights on
the animated river rather than adding a screen-space glow.

The free camera projection is derived from a 36 x 20.25 mm sensor and 24 mm
focal length (45.75 degree vertical field of view). Linear scene radiance then
passes through an f/2.8 aperture, 90% lens transmission, 1/60 s shutter, 5.9 um
pixel area, wavelength-dependent RGB photon energy and quantum efficiency,
45,000-electron full well, ISO 800 analogue gain, Poisson shot-noise
approximation, read noise, and cos^4 lens falloff before ACES display mapping.
Bloom is evaluated before the sensor stage as a compact optical point-spread
approximation.

This is a physically dimensioned real-time transport path, not a declaration
of perceptual identity with unaided human vision. It does not yet include
measured spectral power distributions, wavelength-resolved material BRDFs,
dynamic shadow maps for the clustered lights, lens aberration/diffraction,
sensor colour matrices, retinal adaptation, or smoke multiple scattering.
Those require timestamped source/material/camera measurements and are explicit
next calibration inputs.

## p95 bottleneck analysis

`python -m tools.profile_runtime --frames 120` runs three repeatable GL
timestamp-query cases with 8,000 held stars: full moving-camera rendering,
moving-camera rendering without smoke, and full static-camera rendering.
With physical sensor response and eight clustered radiometric lights enabled,
a 240-frame run measures 5.113 ms GPU p95 and 6.099 ms CPU-submit p95 for the
full moving-camera case. The no-smoke GPU p95 is 4.779 ms. The complete visual
path therefore remains comfortably inside the 16.67 ms 60 Hz budget.

The initial Python octant clustering implementation cost 6.58 ms per 8,000
sources. Replacing repeated boolean masks and weighted averages with
`bincount` power moments reduces it to 1.52 ms while preserving source power.
The fluid advection path also caches face-coordinate grids and reuses one
centred velocity reconstruction for all midpoint backtraces; it changes no
grid, time step, coefficient, or solver iteration.

The reflection texture is redrawn at up to 30 Hz while the camera moves, but a
stationary camera reuses static city geometry and refreshes only for the
one-second sky-lighting invalidation. Animated water normals continue to
distort the cached radiance every 60 Hz frame. Far facades also replace
per-window occupancy hashes with filtered average emission beyond 1,600 m,
limiting shader aliasing and work without replacing geometry with billboards.

A stricter comparison uses separate processes to avoid cross-context driver
contamination:
`python -m tools.profile_runtime --frames 240 --fluid-backend 3d
--integrated-only`, followed by the equivalent `2d` and `cpu` commands. Each
case blocks on GPU completion while moving the camera and updating 8,000
stars, 120 Hz ballistics, fluid evolution, evolving water, reflections,
physical sensor noise, bloom, and tone mapping.

The final 240-frame 3D run measures 10.816 ms mean, 13.431 ms p95, and
14.700 ms p99. Physics p95 is 5.805 ms and visual p95 is 8.223 ms, leaving
3.24 ms of p95 margin inside the 16.67 ms 60 Hz budget. A longer 720-frame
run improves to 12.873 ms p95 and 13.734 ms p99. In the same isolated
240-frame procedure, the OpenGL 3.3 2D fallback measures 13.028 ms p95. The
NumPy fallback is machine-load sensitive and measures 29.592 ms p95 in the
recorded run, which confirms that it is a correctness compatibility path
rather than the target quality tier.

The key latency decision is to advance the inertial plume at 30 Hz while
volumetric ray marching, water, camera, particles, and exposure stay at 60 Hz.
Doubling pressure iterations per fluid step preserves 720
iterations/s, and the real 3D field removes the former Gaussian-depth visual
assumption. A threaded NumPy double-buffer remains rejected because it
competes with rendering for memory bandwidth. With exact interior depth
termination enabled, a 120-frame timestamp-query run records 8.096 ms GPU p95
and 13.681 ms CPU-submit p95 for the full moving-camera case. Both remain
inside the 16.67 ms display budget despite concurrent host-load variance.
The next latency work is ordered as follows: add active sparse-brick dispatch
beyond the current launch domain, add GPU source-reduction diagnostics, then
calibrate smoke lighting and multiple scattering. Dynamic resolution, if
required, applies only to volumetric radiance and reflection, never to
trajectory, terrain, building geometry, or the fixed physics clocks.

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
above. The remaining fidelity boundary is the 2D-to-analytic-depth plume
assumption, reinforcing the planned migration to a sparse 3D GPU solver.

## Model references

Model constants and equations are anchored to the following primary or
technical references:

- [NIST SI photometry and the 683 lm/W definition](https://www.nist.gov/pml/special-publication-330/sp-330-appendix-1)
- [NIST candela realization and inverse-square photometry](https://www.nist.gov/pml/sensor-science/optical-radiation/realization-candela)
- [US DOE LED energy and efficacy fundamentals](https://www.energy.gov/cmei/ssl/led-basics)
- [Physically Based Rendering: film and imaging pipeline](https://www.pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Film_and_the_Imaging_Pipeline)
- [Fireball radiative-fraction measurements](https://publications.iafss.org/publications/fss/6/1125/view/fss_6-1125.pdf)
- [Pyrotechnic-star burning-time measurements](https://www.jes.or.jp/mag/stem/Vol.67/No.1.09.html)

## Required reference datasets

- October 5, 2024 hourly and, where possible, sub-hourly weather observations
- Launch-barge positions, firing timeline, shell types, and calibrated video
- Terrain/building/bridge geometry and nighttime emissive references
- Han River wind-wave observations or a defensible wind-driven spectrum
- Solar/lunar ephemeris, cloud cover, atmospheric visibility, and sky luminance
- Camera position, focal length/FOV, exposure, white balance, and sensor curve
