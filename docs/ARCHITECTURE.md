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

Geospatial source data is transformed from WGS84 to a local ENU origin using
`simulator/geodesy.py`. The transform is bidirectional: `to_geodetic` inverts a
local position back to WGS84 so simulated positions can be compared against
externally reconstructed ones in a shared frame. Observer position, epoch,
launch coordinates, and seeds live in scenario data rather than source code —
see "Scenario, time, and provenance" below.

## Scenario, time, and provenance

`assets/scenario_yeouido_2024-10-05.json` is the single authority for what is
being reconstructed, where it is observed from, and when. `simulator/scenario.py`
loads and validates it; `simulator/app.py` no longer contains an observer
position, a launch coordinate, or a seed literal.

Three clocks are kept distinct by `simulator/timebase.py`:

- **Absolute event time** — timezone-aware, stored UTC, displayed `Asia/Seoul`.
- **Playback time** — seconds since the playback origin, which is the show
  start. Position is held as an integer fixed-step count, so 12,000 steps of
  1/120 s land on exactly 100 s with no accumulated float drift.
- **Wall time** — used only to decide how many fixed steps to run, never how
  long a step is.

`PlaybackMode.REALTIME` reproduces the previous accumulator behaviour, including
its 0.25 s clamp and 8-step catch-up cap, so simulated time may lag wall time
under load. `PlaybackMode.DETERMINISTIC` advances exactly one step per frame and
ignores wall time; this is the mode in which a capture can be compared against
reference footage frame by frame, and it is exercised by the V-02 replay test.

There is no default epoch. A scenario without a timezone-aware
`reference_epoch` fails to load, and a timestamp without a UTC offset is
rejected rather than being reinterpreted in the host machine's local timezone.
The previous code path defaulted the event timestamp to POSIX zero whenever the
weather asset was absent, which silently evaluated the sky at
1970-01-01T00:00:00Z.

`simulator/provenance.py` makes the source-confidence grade a runtime value
rather than prose. Grades are A (measured), B (reconstructed), C (modelled),
D (artistic), U (unverified). Grades A and B require a named source. A derived
quantity can query `Provenance.worst_grade` so it cannot claim stronger
evidence than its weakest input.

The shipped scenario carries **empty** `launch_sites` and `events` lists with
grade-U records. No dated source publishing 2024-10-05 barge coordinates or a
firing timeline has been located, and populating either with an estimate would
convert an unknown into an apparent measurement. Until those datasets arrive,
the simulator has no historical performance to replay.

Random draws come from a named seed registry derived from the scenario's master
seed by `blake2b`, so adding a new stochastic subsystem cannot perturb the
sequence an existing one sees.

`python -m tools.scenario_report` prints the whole resolved state — clock,
derived seeds, observers in both WGS84 and local East-Up-South metres with the
round-trip residual, celestial and atmospheric samples, provenance counts, and
the list of missing datasets — as JSON, with no OpenGL context required.

## Astrometry and atmospheric optics

`simulator/starcatalogue.py` reads a real catalogue, propagates each star from
J2000 by its proper motion, applies annual aberration from the Earth's
barycentric velocity, and transforms the whole set into apparent horizontal
coordinates with one 3x3 multiply per frame rather than one library call per
star. Refraction is applied afterwards because it depends only on the resulting
altitude.

The transform is validated against Astronomy Engine's own per-star path: the
two agree to **0.037 arcsecond**. Omitting aberration moves that to 18
arcseconds, which is the classical aberration constant and why the term is
applied rather than documented away.

The catalogue is stored in J2000 equatorial coordinates and the sky shader
receives the East-Up-South to equatorial rotation as three row uniforms, so the
sky rotates by transforming the view ray instead of re-rasterising the texture
every second.

`python -m tools.import_star_catalogue` fetches the Yale Bright Star Catalogue
(Hoffleit & Warren 1991) from the CDS archive and writes 8,404 stars to
magnitude 6.5 with a provenance sidecar recording URL, checksum, retrieval time,
and citation. **The result is not committed**: the CDS ReadMe for this catalogue
states no explicit licence, so redistributing a derived copy is the repository
owner's decision. Until the importer is run, the renderer falls back to
`procedural_star_catalogue` — 3,500 invented positions, confidence grade D — and
records that it is doing so in `Renderer.star_catalogue_is_measured`.

Star colour comes from the catalogue's B-V index through the Ballesteros (2012)
temperature relation and the existing black-body path. That relation reproduces
the Sun at 5778 K and an A0V star near 10,000 K, which is why a measured colour
index can drive the renderer directly instead of an invented hue.

Sun and Moon are joined by Mercury through Neptune, each with apparent
magnitude and illuminated fraction. At the reference epoch only Venus (altitude
0.5 degrees, magnitude -3.90) and Saturn (altitude 26.5 degrees, magnitude
+0.56) are both above the horizon and naked-eye; Neptune is up at magnitude
7.81 and is suppressed by the visibility gate.

`simulator/atmosphere.py` replaces a hardcoded `exp(-0.12 * (air_mass - 1))`,
which had no recorded source and was duplicated between Python and GLSL, with a
wavelength-resolved optical depth. Molecular scattering uses the Bodhaine et al.
(1999) fit and reproduces published optical depths to **0.06 percent** across
400-700 nm; it scales linearly with the observed station pressure, so
extinction now tracks the weather timeline. Aerosol uses Ångström turbidity,
whose form is standard but whose parameters are a grade C urban estimate. Ozone
is present as a named, wired-in term that **defaults to zero and is graded U**,
because neither the Chappuis-band cross-sections nor the column amount for the
event is held; folding it into the aerosol term would make an absent
measurement look like a modelled one.

The shader keeps the zenith-relative form of the extinction deliberately. The
absolute zenith extinction at the documented turbidity is 0.39 magnitudes;
applying it would dim every star by a further 30 percent and invalidate the
separately calibrated star radiance scale. The calibration and the physics stay
separate until a measured sky luminance is available to recalibrate against,
and the absolute figure is reported by the validation harness instead.

## Shell library and performance scheduler

`simulator/shells.py` replaces the single global shell config with a library of
profiles. A profile carries ballistics, a break pattern, an optical description,
an energy budget, and its own confidence grade. Break patterns are star
**emission distributions**, not new solvers, so they reuse the existing
structure-of-arrays integration and the energy-conserving radiant power solve.

Eleven patterns are implemented — peony, chrysanthemum, willow, palm, ring,
crossette, horsetail, comet, mine, fan, waterfall — across seventeen profiles,
plus strobe and crackle temporal modulation, colour-changing stars, and
secondary breaks for crossette and multi-break shells.

Nothing in this module describes a chemical composition, formulation, or
manufacturing procedure. Colour is either a dominant emission wavelength or a
colour temperature; energy is a specific energy in J/kg; a break is a velocity
distribution.

Coloured stars radiate in narrow bands, so a colour temperature cannot describe
them. `simulator/color.py` maps a dominant emission wavelength to linear RGB
through the CIE 1931 colour matching functions, using the analytic multi-lobe
Gaussian fit of Wyman, Sloan and Shirley (JCGT 2(2), 2013), then the standard
XYZ to linear sRGB matrix. Incandescent effects keep the black-body path, which
is physically correct for them. Both normalise to a peak channel of 1.0, so the
vector carries hue while radiant power stays a separate solved quantity. A
three-channel representation cannot express that a deep blue line delivers far
less luminance per watt than a green one; that needs a spectral renderer.

Secondary breaks draw from their own declared composition mass, so a crossette
cannot inflate its parent shell's energy or smoke budget. Carriers expire over a
spread of steps, so each released batch takes a share of the declared mass
proportional to the carriers it represents — allocating the full budget per
batch silently multiplied a crossette's emitted mass about sixfold before this
was corrected.

Combustion coefficients are stored per star rather than per world. A show mixing
shell types would otherwise attribute every star's smoke and heat to whichever
profile the world was constructed with.

`simulator/show.py` turns a scenario's `events` into launches keyed to absolute
event time. It holds no randomness: every launch follows from the event record
and the clock, which is what allows a replay to be compared frame by frame
against a recording. Seeking moves the cursor without firing, so jumping forward
does not dump the whole show into one frame. A per-shot record may override the
profile's calibre, muzzle velocity, or fuse delay; fields left unset keep the
archetype's value, so "not recorded" stays distinguishable from "recorded as
zero".

Launch geometry is a site position plus a tube azimuth and elevation, replacing
the previous fixed vertical launch from the origin. An elevation of 90 degrees
reproduces the old behaviour exactly.

**Every shipped profile is confidence grade D.** No measured shell record for
the 2024-10-05 performance has been obtained. The profiles are archetypes that
reproduce each named effect's documented visual behaviour; they are not a claim
about any shell that was actually fired. The historical scenario therefore
carries an empty firing timeline, and
`assets/scenario_demo_synthetic.json` — a twenty-two shot sequence covering
every pattern — is labelled a synthetic demonstration throughout.

## Validation harness

`python -m tools.validate --summary` runs the reconstruction validation suite
and prints each metric's status and residuals; `--include-performance` adds the
frame-budget measurement. Exit code is non-zero only for `FAIL` or `ERROR`.

Metrics are declared in `simulator/validation/catalogue.py` with their tolerance
and the physical reason for that tolerance, and implemented in
`simulator/validation/metrics.py`. A metric whose reference dataset is absent
reports `NO_REFERENCE` and never `PASS`; nine of seventeen are in that state
because the firing timeline, launch coordinates, reference footage, and river
gauge records have not been obtained.

Everything except the frame budget and the frame-difference metric runs without
an OpenGL context. The frame budget executes `tools.profile_runtime` in a
separate process, so the harness itself has no GL dependency and records the
machine, platform, backend, frame count, and date alongside the number — a frame
time without that context is not comparable across machines.

Currently passing: deterministic replay (clock, ballistics, burst events, and
acoustic arrival compare bit-exactly), geodetic round trip (2.2e-9 m worst
residual), blast propagation against the analytic Sedov-Taylor solution
(exponent 0.40000), combustion and plume conservation (radiated energy closes to
2.9e-4, pressure projection removes 88% of RMS divergence), and two
astronomical transform cross-checks. Reported without a gate: the CPU
simulation-state footprint, 32.9 MiB at the shipped 250,000-particle capacity.

Colour and brightness comparison reads the linear RGBA16F buffer through
`simulator/validation/capture.py`, before exposure and tone mapping, and stores
frames as `.npy`. A gamma-encoded screenshot has already lost the quantity being
validated, and an SDR display cannot represent firework luminance in any case.

## Renderer structure

`simulator/renderer.py` was 1,891 lines: 904 of GLSL held as Python string
constants, a 463-line constructor that allocated every program, buffer,
texture, and framebuffer in the engine, and a 190-line draw method. It is now
382 lines and does one job — coordinating passes.

GLSL lives in `simulator/shaders/*.glsl`, loaded and cached by
`simulator/shaders/__init__.py`. `shaders.program()` wraps compilation so a
driver error names the files involved; a raw error reports a line number
against an anonymous string, which is not enough to locate the fault among
sixteen shaders.

`simulator/passes/` holds one module per pass, each owning its own programs,
buffers, textures, and draw call:

| Module | Owns |
|---|---|
| `targets.py` | HDR colour and sampleable depth, reflection, bloom attachments |
| `sky.py` | Background program, cloud noise, star catalogue, celestial frame |
| `scene.py` | Static city batches, reflection subset, luminaire positions |
| `water.py` | Near/far grids, JONSWAP spectrum, wind relaxation |
| `land.py` | Terrain-displaced ground plane |
| `particles.py` | Star trails and the reusable staging buffer |
| `smoke.py` | Volume proxy box, state texture, active-bounds shrinking |
| `post.py` | Bloom and the display transform |

The dependency runs one way — the coordinator knows the passes, never the
reverse — and a test asserts it. Pass ordering, the planar-reflection pre-pass,
and the uniforms several passes share (camera, ambient scale, clustered
firework lights) stay in `Renderer`, because those are genuinely cross-pass
concerns rather than anything one pass owns.

Texture unit numbers are now constants in the modules that bind them, and a
test asserts they do not collide. Before the split they were bare integers
repeated across a single long file, where a collision would silently bind the
wrong texture rather than fail to compile.

**The decomposition was verified pixel-exact.**
`python -m tools.capture_reference` renders a fixed scene and writes the linear
RGBA16F buffer; comparing captures from before and after the refactor gives
zero differing components out of 3,686,400 across both stages. The capture is
also bit-identical between separate processes, which is what makes it usable as
a gate rather than an indication.

## Materials

`scene.frag` carried a 140-line if/else chain in which every surface's colour,
pattern scale, and blend factor was a literal buried in GLSL. Those values now
live in `simulator/materials.py` as a table indexed by surface code and uploaded
as uniform arrays, so adding a material is a row rather than a shader edit.

Each material carries the full channel set: base colour (two colours plus a
named procedural pattern), normal strength, roughness, metallic, height,
ambient occlusion, emissive, and transmission. **Every channel is consumed.**
The extension channels the architecture reserves — spectral reflectance, index
of refraction, subsurface scattering, anisotropy, clear coat, wetness,
weathering — are deliberately absent, because a field the renderer ignores
misrepresents the material model; they arrive with the transport that uses them.

The shading path is no longer purely Lambertian. Diffuse and a Trowbridge-Reitz
GGX specular lobe share the same formulation the water pass uses, so the river
and the city respond to a burst the same way. Metallic selects between an
achromatic 4% dielectric reflectance and a tinted conductor response; ambient
occlusion attenuates the sky term only, since that is the term arriving from the
whole hemisphere; transmission adds the backlit glow that thin blades and leaves
show. Height is read as a screen-space bump from the pattern gradient — it has
no parallax, and a true displacement needs the per-pixel surface footprint the
vertex stage does not yet carry.

Building elevations keep their dedicated path. A facade is an assembly — slabs,
mullions, glazing, balconies, expressed structure — not a material, and forcing
it into one base-colour row would lose that detail. Its PBR channels still come
from the table, which is what gives curtain wall its glint under a burst.

**Nothing in the table is measured.** No reflectance for any Yeouido surface has
been obtained, so every material is an appearance calibration at confidence
grade D, and a test asserts none claims otherwise.

The move was verified in two steps against `tools/capture_reference.py`. Lifting
the colours and patterns out of GLSL left **5 components of 3,686,400** changed,
each by one or two ulp of the float16 target at scattered isolated pixels —
compiler float contraction, not a changed value. That check caught a real
transcription error first: the roof colour had been routed through the table but
left at the dataclass default. Consuming the reflectance channels then changed
248,251 components on purpose, which is the specular, occlusion, and
transmission response arriving.

## Vegetation level of detail

Grass blades were generated once at build time inside a fixed 1,200 m radius of
the scene origin and then drawn at every distance with no further gating. That
radius is measured from the **origin**, not the observer, so it was a detail
budget rather than a level of detail: it neither removed geometry the camera
could not resolve nor added any where the camera actually went.

`simulator/vegetation.py` derives the bands from the camera's own optics. At the
default 24 mm lens on a 20.25 mm sensor across 720 rows, one pixel subtends
1.109 mrad. A 0.04 m blade therefore covers 1.5 px at **24.0 m** and 0.5 px at
**72.1 m**, and those are the band edges: full height in, smoothstep collapse to
a degenerate triangle out. A blade narrower than the sample spacing has its
coverage decided by where the pixel centre falls, so drawing it is aliasing, not
detail. Changing the sensor, focal length, or resolution moves the bands with no
further edit — a test asserts that.

Tree crowns keep their geometry at all distances, because a crown is metres
across and stays resolvable. Only the sway animation is gated, at the distance
where its 0.13 m amplitude drops below two pixels (58.6 m): a displacement
smaller than a pixel cannot be seen but is still computed per vertex.

The authoring radius stays, now named `EVENT_SITE_DETAIL_RADIUS_M` and
documented as what it is — where blades may be authored, separate from whether
authored blades are drawn.

## Grass blade placement

The LOD work exposed that every authored blade sat 1,049–1,184 m from the
origin, where a blade subtends 0.03 px. The cause was the builder spending a
fixed 2,500-blade budget in source-file order inside an origin radius, so the
budget was exhausted on whichever polygons the extract happened to list first.

Two things changed. Eligibility now includes `leisure=pitch` whose `sport` is
played on turf, and the budget is spent on the candidates nearest the
**scenario's observers** rather than in file order. The result went from 133 to
299 blade clusters, with the nearest moving from 1,049 m to 415 m of the
observer and from 840 m to 208 m of the default camera.

No pitch at this site carries a `surface` tag, so the playing surface is
inferred from the sport and the inference is recorded as such. Soccer, football,
rugby, cricket, and hockey are treated as turf; basketball, tennis, and running
tracks are excluded because those are hard or synthetic. Natural grass and
artificial turf are not distinguished — both are bladed, and they read the same
at any distance the renderer resolves.

**Blades are still not visible from the default view, and that is a data gap
rather than a rendering one.** The scenario's only observer is the scene-origin
reference point, which sits on the Han River; the nearest mapped turf of any
kind is 415 m away, well beyond the 72 m cutoff. Grass appears when the camera
moves onto the sports fields. Placing a spectator on a bank to make grass
appear would be inventing an observer position, which
`DATA_PROVENANCE.md` records as an unobtained dataset.

## Two observers, and the limit they share

The renderer presents the linear HDR buffer through one of two models, switched
with `V`. They are separate shaders with no shared terms, and a test asserts
neither carries the other's concepts.

**Physical Camera Mode** (`tonemap.frag`) is the existing sensor path: aperture,
shutter, per-channel quantum efficiency, Poisson shot noise, read noise, full
well, cos⁴ falloff, ACES.

**Human Vision Mode** (`human_vision.frag`) replaces all of that with the
observer state computed in `simulator/human_vision.py`:

- **Pupil** from Stanley & Davies (1995), the function Watson & Yellott (2012)
  build their unified formula on. It gives 7.7 mm under starlight and 2.7 mm at
  1,000 cd/m², and it replaces the camera's fixed f-number — a dark-adapted eye
  gathers roughly five times the light of a photopic one. Retinal illuminance
  carries the Stiles-Crawford correction, so a wide pupil delivers less than its
  geometric area suggests.
- **Adaptation** with the asymmetric time constants of Pattanaik et al. (2000):
  0.4 s toward brighter, 120 s toward darker. This is why a burst dazzles
  instantly and the recovery takes the gap before the next shell. It is driven
  by the scene's own computed ambient illuminance rather than a readback, which
  would stall the frame.
- **Mesopic mixing** across the CIE 191:2010 range, 0.005 to 5 cd/m². There is
  one rod photopigment, so rod vision carries no hue and the image collapses
  toward luminance as the cone contribution falls. **At the show's ambient
  illuminance the observer sits at a cone fraction near 0.5 — squarely mesopic**,
  which is the regime the mode exists to represent.
- **Disability glare** from the Stiles-Holladay inverse-square term of the CIE
  glare equation. Scatter in the ocular media veils the retinal image, which is
  why a burst washes out its surroundings rather than merely looking bright.
  The bloom kernel stands in for the near-field point spread and a heavily
  reduced mip for the wide 1/θ² tail; this approximates the CIE equation rather
  than evaluating it.
- **Local adaptation and afterimage** from a quarter-resolution ping-pong
  buffer (`adaptation.frag`). Normalising by the *locally* adapted level rather
  than one global exposure is what leaves a dark patch where a burst has just
  faded. Quarter resolution because adaptation pools over about a degree of
  visual angle, so per-pixel state would be finer than the process it models.
- **Peripheral acuity** by cortical magnification, `1 / (1 + e / E₂)` with
  E₂ = 2.5°, applied as a mip bias from the fixation point. Gaze is fixed at
  screen centre: no tracking is available, and a viewer watching a burst does
  fixate it, so the default is right for the moments that matter and wrong in
  the gaps.

**Not modelled**: the Purkinje spectral shift. Rod vision peaks at 507 nm
against the photopic 555 nm, so short wavelengths should additionally gain as
the eye moves into rod vision. Applying that needs the tabulated scotopic
luminous efficiency V′(λ), which this project does not hold; only the achromatic
collapse is applied, and the colour shift is a known omission rather than an
approximated one.

### The display limit both modes hit

An SDR monitor covers roughly 0.1 to 300 cd/m². A shell burst is orders of
magnitude beyond its top, and the night sky between shells is below its bottom.
**Neither mode reproduces the absolute luminance of the scene, and neither can.**
Both end in a tone-mapping step that is a perceptual compromise.

This is why the validation harness reads the linear RGBA16F buffer *before*
either transform. The HDR buffer is the physical output; the screen is a
rendering of it, and a colour or brightness metric computed on the screen would
be measuring the tone mapper. Reproducing absolute luminance would need an HDR
display path and a measured display characterisation, neither of which exists
here.

## Atmosphere as a field

`EnvironmentTimeline.sample(t)` returned one global state with no spatial
dependence at all, and there was never a visibility term.
`simulator/environment_field.py` provides the `wind / temperature / humidity /
pressure / visibility (x, y, z, t)` interface the requirements ask for.

What one station observation can honestly support is **vertical** structure.
A single point carries no horizontal information, so `StationTimelineField` is
horizontally uniform and says so; a test asserts it. The profiles are standard
relations driven by the observed surface values — barometric pressure with an
ISA lapse rate, temperature from that lapse rate, humidity at a conserved
mixing ratio, wind from the neutral logarithmic surface layer, and visibility
from aerosol extinction through the Koschmieder relation.

Each has a real consumer rather than being interface scaffolding:

- **Density aloft drives shell drag.** Air is 1.5% thinner at a 160 m break and
  2.8% thinner at 300 m, so using the surface value overstated drag through the
  whole climb. `FireworkWorld` now samples density at the shell's own altitude.
- **The unsourced ×1.4 wind factor is retired.** At the roughness length of the
  river corridor, 0.03 m — short grass over water, which is what the Han River
  banks are — the logarithmic profile gives 1.396 at 100 m. Naming the physics
  reproduces the old literal to 0.3% rather than changing the trajectory.
- **Humidity now drives aerosol extinction.** Particles take up water and
  scatter more, so the Ångström turbidity is treated as a dry value and grown
  hygroscopically. The dry coefficient is anchored so that at the show's
  observed 56% humidity the extinction is exactly what it was before: the
  change adds a response rather than silently re-tuning a calibration.
- **Visibility comes out at 17 km** for the show's modelled aerosol. This is
  what the model implies, **not** an observation — the Meteostat record carries
  no visibility field. Supplying a measured visibility would invert the
  relation and calibrate the turbidity instead, which is the more useful
  direction and is the reason the relation is written this way.

A likely **nocturnal inversion is not modelled**. A clear October night over
water very probably carried warmer air above cooler, which would refract sound
downward and extend audible range. No sounding for the event exists, so the
standard lapse rate is used and the inversion is recorded as unmeasured rather
than guessed at.

A `PrecomputedWindField` sampling an offline solve is deliberately **absent**.
Without a solved field it would be an empty class, and the architecture's own
reason for preferring offline to runtime CFD was that an uncalibrated
city-scale solve is a grade-C model wearing the costume of a measurement.

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

## Event-site ground cover and street furniture

A second Overpass query fixed to 2024-10-05 10:20 UTC adds 49 distinct
rendered sports fields, running tracks, playgrounds, gardens, and scrub areas
that were absent from the first broad land-cover pass. Imported road widths
retain the source hierarchy and select separate compacted-trail, concrete
footway, red cycleway, and asphalt-road material families. Metric world
coordinates drive joints, aggregate, lane markings, track markings, and
rubber-playground tiles without a repeating image texture.

The historical data contains 25 `natural=wood` regions but no individual tree
nodes. The asset builder therefore distributes 201 deterministic low-poly
trunks and crowns strictly inside those dated polygons. This reconstructs the
mapped canopy volume and parallax while keeping the unsurveyed trunk positions
explicitly provisional.

The Seoul Future Hangang Headquarters facility map contributes 121 official
current coordinates. Fifty-four visible structures remain after filtering
non-geometric records and positions already covered by OSM buildings:
toilets, shops, information and rescue centres, drinking fountains, smoking
booths, rental kiosks, playground equipment, cafes, and observation
structures. Coordinates are authoritative to the current map, but dimensions,
heading, and 2024 existence remain calibration inputs.

No dated source publishes individual bench or lamp coordinates. The detail
builder consequently labels those meshes as inferred LOD and places them
deterministically beside the actual mapped path segments. The railing is
extracted from water-to-land transitions in the 1024 x 1024 Han River mask,
median-filtered to remove raster stair steps, and rendered with two rails and
regular posts. The resulting detailed batch adds 100,356 vertices while
remaining one draw call in the main pass. The half-resolution water reflection
keeps only 16,560 visually dominant lamp-head and concrete-facility vertices;
dark tree crowns, benches, posts, and ground surfaces reuse the surrounding
land reflection at that distance.

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

## Night atmosphere, vegetation, water, and site luminaires

The background pass now carries a deterministic celestial catalogue even when
no stars reach the sensor. Each star's finite-raster flux is reduced by
Kasten-Young relative air mass, atmospheric extinction, and Beer-Lambert cloud
optical depth. It is then compared against the local clear-sky, twilight, moon,
and urban-cloud background radiance. The contrast gate suppresses the source
before the existing physical camera response, so the 2024 Seoul urban sky can
contain stars physically while displaying none. The catalogue is provisional
and does not yet claim measured astrometric positions.

Cloud density is a periodic three-octave field advected by the observed
horizontal wind. Coverage shifts its density threshold, optical depth controls
direct transmission, and moon/twilight/urban radiance supplies a compact
single-scattering term. The expensive noise synthesis is performed once at
startup; each sky pixel uses one advected texture sample. This retains a
bounded full-screen cost and avoids temporal boiling.

Historical `landuse=grass` and `natural=grassland` polygons seed 133 crossed
blade clusters inside 1.2 km of the event origin. Blade bases remain fixed,
while tip displacement follows a squared cantilever height profile, bounded
wind response, and two-frequency gust. Distant grass receives the same
wind-aligned travelling-wave normal field, and tree crowns receive a smaller
height-weighted sway. These paths share the atmospheric wind used by the
JONSWAP river forcing.

Water now applies wavelength-dependent Beer-Lambert absorption over a
view-angle path estimate, blue-green subsurface scattering, mask-gradient
shore foam, wind-gated crest foam, and slope/wind-dependent GGX roughness.
Planar city reflection and clustered firework energy remain unchanged.

The imported lamp heads are recovered as physical luminaire positions. A
default 72 W street fixture uses the existing LED driver, quantum, extraction,
phosphor, and optical-efficiency chain. Its downward cosine lobe illuminates
nearby land and scene geometry with inverse-square attenuation. Four nearest
lights are retained for detail geometry, while the 5 km land pass evaluates
only the two nearest lights.

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
After adding the 100,356-vertex event-site batch, the conservative
moving-camera timestamp run measures 14.028 ms GPU p95. The coupled blocking
visual section measures 16.474 ms p95. Reflection LOD is therefore required
for the fidelity default; submitting the full furniture/tree batch to the
30 Hz reflection pass is outside the intended laptop margin.

After adding cloud/star radiative visibility, wind vegetation, spectral water,
and local luminaires, a clean 240-frame `3d --integrated-only` run measures a
12.282 ms visual mean and 15.116 ms visual p95. The graphics section remains
inside 16.67 ms. The deliberately blocking diagnostic serializes two 120 Hz
physics steps and GPU completion; it records 18.618 ms frame mean and
24.522 ms frame p95, with physics p95 at 11.108 ms. Therefore this stage meets
the isolated visual budget but does not claim a universal 60 Hz guarantee
under host load or forced CPU/GPU serialization. The next latency priority is
physics-step spike reduction and real frame-pacing telemetry rather than
removing the new radiative terms.

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
