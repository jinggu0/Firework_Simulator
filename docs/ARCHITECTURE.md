# Yeouido Fireworks Simulator

## Fidelity contract

The simulator keeps physical state in SI units and linear-light rendering
values. Artistic constants are permitted only as explicitly identified
calibration values, so measured weather, shell, terrain, camera, and spectral
data can replace them without redesigning the engine.

The ordered implementation programme and current completion gates live in
[`VISUAL_FIDELITY_ROADMAP.md`](VISUAL_FIDELITY_ROADMAP.md). Static Yeouido
appearance is completed before the historical 2024 performance is integrated.

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
reports `NO_REFERENCE` and never `PASS`; nine of twenty-four are in that state
because the firing timeline, launch coordinates, reference footage, and river
gauge records have not been obtained.

Four metrics need an OpenGL context and each runs in a separate process, so the
harness itself has no GL dependency. V-12, the frame budget
(`--include-performance`), executes `tools.profile_runtime` and records
machine, platform, backend, frame count, and date alongside the number — a
frame time without that context is not comparable across machines. V-22, V-23
and V-24 (`--include-rendering`) each render a frame and compare it against a
closed-form prediction of the same frame, so unlike V-12 their tolerances come
from numeric precision and their results *are* portable: V-22 checks the aerial
perspective composite against the linear buffer, V-23 the camera's display
transform, and V-24 the observer's in full — see *The brilinear filter, and why
the blur was driver-dependent*.

Currently passing: deterministic replay (clock, ballistics, burst events, and
acoustic arrival compare bit-exactly), geodetic round trip (2.2e-9 m worst
residual), blast propagation against the analytic Sedov-Taylor solution
(exponent 0.40000), combustion and plume conservation (radiated energy closes to
2.9e-4, pressure projection removes 88% of RMS divergence), two astronomical
transform cross-checks, aerial perspective against its CPU reference (8.3e-4
residual against a 2e-3 half-float bound, sky pixels bit-identical), and the
display and observer transforms against their CPU references (0.63 display code
values each, against a 2-code bound). Reported without a gate: the CPU
simulation-state footprint, 32.9 MiB at the shipped 250,000-particle capacity.

Colour and brightness comparison reads the linear RGBA16F buffer through
`simulator/validation/capture.py`, before exposure and tone mapping, and stores
frames as `.npy`. A gamma-encoded screenshot has already lost the quantity being
validated, and an SDR display cannot represent firework luminance in any case.
V-23 is the sole exception and only because the stage it tests is what produces
the 8-bit image; every other colour metric stays upstream of it, which is why
changing the white balance or the tone curve cannot flatter any of them.

## Renderer structure

`simulator/renderer.py` was 1,891 lines: 904 of GLSL held as Python string
constants, a 463-line constructor that allocated every program, buffer,
texture, and framebuffer in the engine, and a 190-line draw method. It is now
382 lines and does one job — coordinating passes.

GLSL lives in `simulator/shaders/`, loaded and cached by
`simulator/shaders/__init__.py`. `shaders.program()` wraps compilation so a
driver error names the files involved; a raw error reports a line number
against an anonymous string, which is not enough to locate the fault among
nineteen shaders.

The loader also resolves a `#include "name.glsl"` directive, which GLSL itself
lacks. `.glsl` files are includes only and never offered as compilable stages;
a test asserts both, and a circular include is reported with its chain rather
than recursing. This exists because five stages need the same atmospheric
extinction integral, and five copies that drift would be invisible in the
rendered output.

`simulator/passes/` holds one module per pass, each owning its own programs,
buffers, textures, and draw call:

| Module | Owns |
|---|---|
| `targets.py` | HDR colour and sampleable depth, reflection colour and depth, airlight, ambient-obscurance, bloom attachments |
| `sky.py` | Background program, cloud noise, star catalogue, celestial frame, airlight field |
| `scene.py` | Static city batches, reflection subset, luminaire positions |
| `water.py` | Near/far grids, JONSWAP spectrum, wind relaxation |
| `land.py` | Terrain-displaced ground plane |
| `particles.py` | Star trails and the reusable staging buffer |
| `haze.py` | Deferred extinction and airlight composite |
| `ambient_occlusion.py` | Half-resolution depth-plane contact obscurance and bilateral resolve |
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

Asphalt, pavers, maintained ground cover, and structural concrete additionally
read four 1K CC0 photo-scanned PBR sets through one mipmapped texture array.
Diffuse colour is normalized by the scan's linear mean before it modulates the
event-calibrated base colour; this transfers physical variation without
claiming that a generic scan was sampled at Yeouido. A derivative cotangent
frame applies OpenGL normal maps on horizontal and vertical faces without
adding tangents to the 2024 city mesh. ARM channels drive per-pixel ambient
occlusion and roughness; mipmaps and a 90-420 m detail fade bound aliasing and
fragment cost.

Mapped asphalt segments inside 1.7 km derive a 140 mm raised concrete kerb and
180 mm top strip. The road footprint remains the dated OSM width, while the
vertical profile is explicitly a generic grade-D detail. Park luminaires now
carry a base collar, twelve-sided pole, outreach arm, metal housing and a
separate emissive lens instead of a pole-and-box silhouette.

Building elevations keep their dedicated path. A facade is an assembly — slabs,
mullions, glazing, balconies, expressed structure — not a material, and forcing
it into one base-colour row would lose that detail. Its PBR channels still come
from the table, which is what gives curtain wall its glint under a burst.

**Nothing identifies a measured Yeouido reflectance.** No reflectance for any
site surface has been obtained, so every material remains an appearance
calibration at confidence grade D. The scans measure their source samples, not
this site, and a test asserts no table row claims evidence grade.

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

Two things changed in the first correction. Eligibility now includes
`leisure=pitch` whose `sport` is played on turf, and the budget is spent on the
candidates nearest the **scenario's observers** rather than in file order. The
current density pass samples those dated polygons on a 0.62 m lattice and keeps
28,790 tuft anchors from 296,197 candidates. Each anchor expands to five narrow,
double-sided blades (30 vertices), with the nearest and farthest retained
anchors 411 m and 1,187 m from the scenario observer respectively.

At upload time the grass is separated into eleven 64 m spatial chunks. The
scene pass rejects chunks outside the optical blade cutoff before issuing a
draw, so the default river camera pays no grass raster cost while a camera on a
mapped field sees only nearby cells. In the near-field regression scene this
drew four of eleven cells and held visual p95 at 7.96 ms.

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
  toward luminance as the cone contribution falls. **At the show's converged
  ambient — 0.383 cd/m², the 1.2 lux urban ambient over π — the observer sits
  at a cone fraction of 0.628: squarely mesopic**, which is the regime the mode
  exists to represent. Better than a third of the image is rod signal and
  therefore already achromatic.
- **Chromatic adaptation** by von Kries gain control in the CAT02 cone
  channels, taking the field's own average chromaticity to the display white.
  See *Discounting the illuminant* below.
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
  screen centre. This virtual-retina term is enabled by observer validation but
  disabled in the interactive monitor view: without eye tracking, baking a
  fixed gaze into the image would blur it once in the shader and a second time
  in the viewer's real retina. Pupil response, adaptation, mesopic colour,
  glare, and afterimage remain active.

Human Vision Mode is the default interactive display path. `V` retains the
physical-camera path for footage matching and sensor validation.

**Not modelled**: the Purkinje spectral shift. Rod vision peaks at 507 nm
against the photopic 555 nm, so short wavelengths should additionally gain as
the eye moves into rod vision. Applying that needs the tabulated scotopic
luminous efficiency V′(λ), which this project does not hold; only the achromatic
collapse is applied, and the colour shift is a known omission rather than an
approximated one.

### Discounting the illuminant

An observer does not see a warm-lit scene as uniformly orange. The cone
channels rescale until the dominant light reads closer to neutral, and that
gain control — von Kries adaptation — is what the camera's white balance
imitates. The observer path had none of it.

It is performed in the CAT02 cone space of CIECAM02 rather than in display
primaries, because adaptation is a change of cone gains and CAT02 is the space
the degree-of-adaptation relation is defined against. Both matrices live in
`simulator/color.py` and a test extracts the GLSL literals and compares them:
GLSL `mat3` is column-major, so a matrix written row-wise compiles, runs, and
produces a plausible image.

Three things make the effect smaller than the machinery suggests, and each is a
measurement rather than a choice:

- **Adaptation is never complete.** CIECAM02's `D = F[1 − (1/3.6)e^((−L_A−42)/92)]`
  caps at the surround factor, 0.8 for the dark surround a night show is. Over
  the luminance band this show occupies D stays near **0.660** — the observer
  discounts about two thirds of the illuminant's colour. The relation is fitted
  in photopic conditions and is being extrapolated into the mesopic; that is
  flagged, and bounded by the next point.
- **A third of the image is rod signal.** Adaptation acts on cones, so it is
  applied to the cone path and the rod path keeps the unadapted luminance —
  rods have one photopigment and no gain control that could discount a hue they
  cannot see. At cone fraction 0.628 the mesopic mix passes only that much of
  the effect through.
- **The field is only mildly warm.** The adapting white converges to
  (1.141, 0.955, 0.984) — the city lighting dominating a dark sky. The
  resulting gains are **(0.978, 1.020, 1.008)**, a 4.4% spread, which after the
  mesopic mix is a **2.7%** shift in the linear signal and about 4 code values
  in the 8-bit image.

**The time course is the substantive part.** Fairchild & Reniff (1995) measured
chromatic adaptation as roughly 90% complete after 60 s, so the constant here is
60/ln(10) = 26.06 s. That slowness is why a two-second break is **not**
discounted: the adapting white moves 7.4% of the way toward a green shell's
chromaticity while it burns, so the observer sees it as green. A fast constant
would desaturate every break, which would be a modelling error rather than a
subtle one. The fast receptoral component Fairchild & Reniff also report is not
modelled, so a step change here begins slower and finishes faster than the
measurement.

The adapting white is tracked on the GPU, in the same ping-pong buffer as the
local adapting luminance — `rgb` the white, `alpha` the luminance — because it
is a per-frame image statistic and reading it back would stall the frame for a
value the shader is about to use. It is **global**: sampled from the last mip
level at a fixed point, so every texel arrives at the same value rather than
happening to. It is stored normalised to unit luminance, which is what makes
the von Kries step luminance-preserving; brightness adaptation is the other
channel's job.

Two defects found by V-24 rather than by looking at the image, both recorded
here because neither would have been visible: the pooling level was computed
with `ceil` rather than `floor` and asked for a mip that was never allocated
(drivers clamp, so it worked by accident); and storing a normalised white as
half floats let a systematic rounding of a few 1e-4 be amplified by
1/response ≈ 18, drifting the white 3.7e-3 off unit luminance. It is now
renormalised on both write and read.

### The brilinear filter, and why the blur was driver-dependent

V-24 predicts the observer transform from the buffers the shader reads, so
every stage is checked against arithmetic rather than against appearance.
Reading the generated mip levels back — rather than rebuilding a pyramid in
NumPy — makes `textureLod` reproducible. The **glare tail** fell out
immediately: it reads a fixed level and contributes exactly nothing to the
residual.

**Peripheral acuity did not, and chasing that turned up a renderer defect.**
Its mip bias left up to 7 display code values of residual. Eliminated in turn,
each by measurement: the glare term (zero contribution), LOD quantisation
(snapping to 1/32 or 1/16 changes nothing), anisotropic filtering (forcing it
to 1 changes nothing), the level contents (read back directly), a systematic
LOD bias (zero offset already optimal), and the CPU gather (exact against a
per-pixel loop). Instrumenting the shader to report its own LOD field showed it
matched the reference to within the 8-bit readback quantisation, so the LOD was
right too.

What remained was the interpolation itself. A *constant* LOD of 2.0, 3.0 or 4.0
reproduced exactly, and so did 2.5 and 3.5 — but 3.25 and 3.75 did not. Blend 0
and blend ½ exact, other fractions wrong: solving for the weight the GPU
actually applied gave **0.875 for a requested 0.75**, and at that weight the
prediction matched exactly. The curve

```
weight = clamp((frac − 1/6) / (2/3), 0, 1)
```

fits five measured points — 0.125→0, 0.25→0.125, 0.5→0.5, 0.75→0.875, and 0
and 1 exact — with the last two predicted before they were measured. This is
**brilinear filtering**: a documented driver optimisation that runs true
trilinear only over the middle third of each transition and snaps to the nearer
level outside it.

That made the peripheral blur a property of the graphics driver rather than of
the eye — unacceptable in a reconstruction that has to be reproducible on other
hardware. `human_vision.frag` now blends two explicit integer levels, where the
hardware weight is zero and the fetch is exact. It was the project's only
fractional `textureLod`; the other three sample levels 3, 5 and 10, and a test
now keeps it that way.

**The residual fell from 7.2 to 0.63 code values, and the gate acquired teeth
it never had.** Before the fix a 10% error in the acuity constant E₂ moved p99
from 1.62 to 1.80 — invisible. After it, that same error **fails** the metric
at 2.65 code values, as does a LOD bias of 0.05. Peripheral acuity is now
inside the gate, and V-24 has no unverified stages.

**Not modelled**: local chromatic adaptation, and with it coloured afterimages.
The local *luminance* adaptation above does produce achromatic afterimages, but
a coloured one comes from photopigment bleaching rather than from von Kries
gain control, and CAT02's degree of adaptation is defined for a global adapting
field. Extending it per-pixel would be using the model outside what it is
calibrated for.

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
- **Visibility comes out at 17 km** for the show's modelled aerosol, and it is
  now what the frame is drawn with — see *Aerial perspective* below. This is
  what the model implies, **not** an observation: the Meteostat record carries
  no visibility field. `AtmosphericOptics.with_visibility_m` is the adapter for
  the day one arrives — it inverts the relation and calibrates the dry
  turbidity, which turns a grade C estimate into a grade A constraint.

A likely **nocturnal inversion is not modelled**. A clear October night over
water very probably carried warmer air above cooler, which would refract sound
downward and extend audible range. No sounding for the event exists, so the
standard lapse rate is used and the inversion is recorded as unmeasured rather
than guessed at.

A `PrecomputedWindField` sampling an offline solve is deliberately **absent**.
Without a solved field it would be an empty class, and the architecture's own
reason for preferring offline to runtime CFD was that an uncalibrated
city-scale solve is a grade-C model wearing the costume of a measurement.

## Aerial perspective

The visibility field above had no consumer in the renderer. The only air
extinction in the frame was `air_extinction_per_m = 0.00012`, an unsourced
scalar that implied 32.6 km of visibility — about twice what the event's own
weather supports — and it was applied *only* between a lamp and a surface,
never along the view path. The far bank of the Han River was rendered as if it
were in vacuum.

`SurfaceExtinction` now divides the column optical depths of
`simulator/atmosphere.py` by the scale height of the species that produced
them, so the air that dims the stars is the same air that hazes the skyline.
Aerosol and molecules keep separate scale heights, 1,200 m and 8,500 m, because
haze thickens toward the horizon about seven times faster than Rayleigh
scattering does. At show conditions the surface coefficients are
2.0 / 2.3 / 2.9 × 10⁻⁴ m⁻¹ for red, green, and blue: blue is attenuated
1.51 times as strongly as red, so distant lights redden by the amount the
Ångström exponent says they should rather than by a chosen tint.

The composite is Koschmieder's own relation,
`L = L_object · T + L_air · (1 − T)`, and it is applied in five places from one
shared GLSL include (`shaders/air_extinction.glsl`, resolved by a `#include`
directive the loader implements — five drifted copies of an integral would be
undetectable in the output):

- **The opaque scene** gets a deferred full-screen composite over the depth
  buffer. Blending on the geometry itself would depend on the order the
  buildings happened to be drawn in. The blend pipeline carries one scalar
  alpha and the transmittance is per channel, so it runs as two draws:
  `(ZERO, ONE_MINUS_SRC_COLOR)` multiplies the target by `T`, then
  `(ONE, ONE)` adds the airlight. The order is not interchangeable.
- **`L_air` is the sky pass drawn a second time**, at an eighth resolution,
  with the view ray projected onto the horizontal and stars suppressed. That is
  the radiance Koschmieder's 2% contrast threshold is defined against, so an
  object at the visibility range sits at exactly that contrast and one at
  infinity is exactly the sky. Re-evaluating the sky model rather than
  authoring a horizon colour is what stops the two from drifting apart. Cost:
  0.0026 ms.
- **Stars and the plume** are additive over a composite that already carries
  the airlight, so they take their own path transmittance and no airlight —
  folded into the star colour in the vertex stage, and carried as a running
  per-step product through the smoke march.
- **Point lights** use the same integral over the source-to-surface path,
  retiring the scalar literal.

The path integral is exact rather than a constant-density approximation. For a
straight segment through an exponential profile the optical depth has a closed
form, `σ₀·L·H/Δz·(e^(−z₀/H) − e^(−z₁/H))`, and it matters: at 1.5 km range a
300 m break sits behind 12% less air than a level path claims, and a 600 m
break behind 22% less. Heights are taken as magnitudes so the mirrored water
reflection integrates the air it actually crosses.

**V-22 checks this rather than trusting it.** The same frame is rendered twice,
once with the modelled atmosphere and once with it removed, which recovers
`L_object` per pixel; the depth buffer gives the path; and the composite is
predicted on the CPU from `SurfaceExtinction.transmittance`. The measured
residual is 8.3 × 10⁻⁴ of the frame's peak radiance, against a 2 × 10⁻³ bound
set by the half-float buffer's own 4.9 × 10⁻⁴ quantum. Sky pixels are
bit-identical between the two renders — they already carry the airlight of an
infinite path, and that requirement is exact, not approximate.

At show conditions the median lit pixel loses 4.2% of its radiance, the far
bank at 1.7 km loses 17%, and airlight supplies 20% of what remains there. The
two haze draws cost 0.0707 ms.

### The reflected path

The river is 59% of the visible geometry, and it shows a skyline that reaches
the eye by way of the water: object → surface → camera. That path is longer
than the direct one, so leaving the planar reflection pre-pass clear made the
river a window onto a haze-free city.

The mirrored camera already gets the geometry right — the straight line from
the mirrored eye to an object has exactly the length of the two real segments —
so the pre-pass now runs the same deferred composite, with two things the
direct path does not need:

- **The below-datum half is discarded.** The mirrored line's water-to-eye half
  is the water pixel's own path to the camera, and the main haze pass already
  applies that to the water fragment. Counting it in both places would
  attenuate it twice, and it is not a rounding error: for a low building
  reflected at 800 m, 45% of the mirrored optical depth belongs to that half.
  Clipping at the datum also makes the remaining integral exact rather than
  approximate, because height varies linearly along what is left.
- **The airlight field is re-rendered for the mirrored bearings**, into the
  same target the main pass uses, immediately before the reflection consumes
  it. The reflected ray's azimuth is not the direct ray's.

The two path segments then compose correctly by construction:
`f·R·T_ow·T_wc + f·L_air(1−T_ow)T_wc + L_air(1−T_wc)` — each segment's airlight
appearing once, attenuated by whatever follows it.

**The plume is drawn into the reflection as well.** Its geometry needs no
special case: the volume sits entirely above the water datum, so a ray from the
mirrored camera meets the real plume above the datum and the march is already
the reflected image of it. Only the air in front of it is treated differently,
clipped at the datum for the same reason the haze composite is. The proxy box's
facing test moved from the fluid-revision path to the draw, because the frame
now marches the plume from two cameras and only one of them can be inside the
box.

Two things are worth recording about verifying it, because the first two
attempts measured nothing and both were viewpoint artifacts rather than
failures. From the default camera the plume's mirror image falls **outside the
mirrored frustum** — at 135-183 m altitude and 235 m away, its reflection sits
below the frame. From a second viewpoint the reflected ray crossed the datum
over the **bank**, where the land writes depth and correctly occludes the
plume: you cannot see a reflection where you are looking at the riverbank. A
valid viewpoint has to be derived from the river mask — the reflection point is
`0.88 × z_camera` for this plume height, which must land inside the water
polygon.

Measured from such a viewpoint: 882 reflection texels change, adding 4,243
pixels to the linear frame at a mean of 2.8e-6 and a peak of 1.2e-4 W/(m² sr).
Through the display transform that is 453 pixels at one code value — present
and correct, but subtle, because water reflects only about 22% at the grazing
incidence a reflection is seen at, the plume is optically thin, and the
reflection buffer is half resolution. Cost, measured by interleaving the two
configurations frame by frame so drift cannot bias it: **+0.046 ms** when the
reflection refreshes every frame, **+0.023 ms** amortised at its 30 Hz cap.

The **path-integral primitive gained an exact datum crossing** to support this,
which also fixes the mirrored star draw: the water reflection of a burst is
drawn from the same mirrored position, and folding the endpoints with `abs()`
understated its optical depth by 1-2%. The split reproduces numerical
quadrature to float precision and the CPU and GLSL forms are line-for-line the
same, down to the level-path threshold.

Measured: reflected geometry loses **2.7% on average and 43% at worst**, 31,486
pixels change by up to 36 display code values, and the reflected sky is
bit-identical — it already carries an infinite airlight path, the same exact
requirement the direct sky has. Cost 0.032 ms, and only on the frames that
refresh the 30 Hz reflection.

V-22 gates this by sign rather than by residual. A water pixel's radiance is no
longer recovered by the vacuum render — the reflection inside it changed too —
so water is excluded from the residual gate and checked instead against the one
thing a mirrored-path sign error would reverse: hazing a longer path must
remove radiance.

The airlight veil remains only as good as the sky model's **horizon radiance,
which is grade D**. At night the veil is dim against lit facades, so the visible
effect is mostly extinction rather than milkiness, and that balance would move
with a measured night-sky brightness.

## Render passes

1. Sky radiance and astronomical lighting, plus the horizontal airlight field
2. Terrain, bridges, buildings, and emissive city lights
3. Spectral water displacement and reflection — the mirrored pre-pass carries
   the skyline, the aerial perspective over the reflected path, and the plume
4. Aerial perspective over the opaque scene
5. Firework shells, stars, sparks, and trails
6. Participating-media smoke and light scattering
7. HDR bloom, camera exposure, sensor response, and tone mapping

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
containing the astronomical sky, terrain, 2024 building geometry, and bridges,
with aerial perspective applied over the above-datum half of the reflected path
(see *The reflected path*). The reflection camera is mirrored around the water
datum and the projected sample is perturbed by the same spectrum-derived normal
used for shading.
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
East-Up-South metres before rendering. The shipped asset contains 103,866
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

Named historical features now use their contained OSM `building:part` geometry
instead of drawing a duplicate parent extrusion over it. This exposes the 63
City main shaft, two single-slope gold-glass side masses and white crown, and
the stepped crystalline IFC parts. `roof:shape=skillion` plus `roof:height`
produces a true sloped plane and side silhouette rather than a flat roof.

Architect-published completed heights override stale named OSM totals only
where the identity is exact: Parc.1 towers are 318 m and 246 m, its Fairmont
hotel is 101 m, and FKI Tower is 240 m. Parc.1's documented red exterior frame
is projecting geometry: four corner columns and floor-scale perimeter beams
now catch light, cast depth, and appear in silhouette instead of being painted
into the window shader. FKI's facade alternates the documented 30-degree BIPV
spandrel and 15-degree vision-panel normal response on every floor; its roof
adds the documented 10-degree photovoltaic canopy.

The National Assembly keeps its OSM mass and 43-61 m curved dome part, colours
the latter as weathered copper, and places the officially documented twenty-
four octagonal exterior columns as geometry. Eight columns line each long
elevation and four line each end; because the named OSM outline includes the
chamber wings, this colonnade is contracted onto the central block rather than
the full concave bounds. Residential buildings retain smaller floor spacing
and balcony bands, while `min_height` remains preserved for every elevated
part.

This is evidence-bounded architectural reconstruction, not a claim of
photogrammetric identity. Published heights, part footprints, roof slopes and
feature counts are retained as stated. Public descriptions do not publish every
column cross-section, connection plate, facade panel, sign or rooftop plant
position, so those dimensions remain visible grade-D appearance calibrations.
Deterministic window occupancy is not measured room-by-room lighting from
October 5, 2024. Sub-metre identity still requires timestamp-matched survey or
photogrammetry.

The Han River surface is clipped by a 1024 × 1024 geographic coverage mask
generated from OSM multipolygon relation 152336. Both outer riverbanks and
inner land/island rings are preserved. The mask covers a 5 × 4 km local area,
which gives roughly 4–5 m source-mask resolution before GPU interpolation.
This removes water from Yeouido park and other land while keeping the wave
simulation independent of shoreline complexity. Land uses the constrained
terrain surface described below rather than a flat provisional plane.

With this scene, water, and the initial firework active, the uncapped
development-machine measurement is approximately 1.7 ms per frame. This is a
pipeline headroom measurement, not a guarantee for the final volumetric and
reflection workload.

After geographic shoreline masking and the provisional land pass, the same
uncapped workload measures approximately 2.6 ms per frame (about 383 FPS).

## Terrain elevation

The current height map rasterises the official 2023 NGII-notified Seoul
1:5,000 contours and spot heights distributed as Seoul Open Data OA-22241.
EPSG:5174 vectors are transformed through WGS84 into the scene's local
East-Up-South frame. Contours are sampled no farther apart than 15 m and merged
with every spot height inside a 100 m scene margin, producing 19,672 unique
land constraints after water rejection and 0.5 m positional deduplication.

A Delaunay triangulation and piecewise-linear interpolator honour every source
height without cubic overshoot. The resulting 1024 by 1024 grid spans the
5 by 4 km scene at approximately 4.89 by 3.91 m per texel. That is only the GPU
sampling density; it is not represented as a surveyed 5 m DEM. The official
constraint hull covers 100% of this grid. The former Mapzen/AWS Terrarium
surface remains a reproducible out-of-hull fallback but supplies no cells in
the shipped asset.

The vertical datum is no longer inferred from coarse river pixels. WAMIS
station 1018683 publishes a 2.07 EL.m gauge zero and observed 0.72 m stage at
both 19:00 and 20:00 KST on the event date. Simulation `y=0` is therefore
2.79 EL.m at the 19:20 show start. Hourly data records the tidal rise after
20:00, but animating the reflection plane with it remains future work.

In a common absolute EL.m comparison at all 19,672 constraints, the retired
512-grid baseline has 9.375 m RMSE and 7.029 m mean absolute error. Bilinear
sampling of the new runtime raster measures 2.488 m RMSE and 0.782 m mean
absolute error; median error is effectively zero. The remaining p95 absolute
error of 4.033 m is concentrated where multiple steep constraints fall inside
one runtime texel, so increasing texture resolution trades memory/cache cost
against those local slopes rather than creating new observations.

The following paragraph records the retired baseline for reproducibility:

The initial terrain height map samples Mapzen/AWS Terrarium elevation tiles at
zoom 12 into a 512 × 512 local grid. The median DEM elevation under the Han
River mask is 5.01 m and becomes simulation `y=0`; stored land elevations are
relative to that datum. The current 5 × 4 km scene ranges from -1 m to about
87 m relative height.

Terrain is sampled in the GPU vertex stage. Buildings and bridge decks sample
the same texture at their footprint positions, preventing detached or buried
geometry. Finite differences over the terrain texture provide land normals for
night-sky shading.

That retired DEM is a regional bare-earth baseline, not a survey of individual
embankments. Its effective source resolution is roughly 30 m in this area and
cannot resolve narrow levees, stairs, roads, or the exact riverside viewing
surface. Those features require a higher-resolution Seoul/VWorld dataset or
manual photogrammetric calibration. The constrained surface removes its urban
spikes and broad floating-road plates, but narrow structures still require the
restricted NGII 1 m DEM, site survey, or timestamp-matched photogrammetry.

With terrain displacement and terrain-sampled buildings enabled, the uncapped
development-machine run measures approximately 1.94 ms per frame (about
515 FPS). Timing varies with driver scheduling, so the frame time—not the
uncapped FPS headline—is retained for later regression comparisons.

## Camera

The runtime camera provides both six-degree free flight and a physical
camera-operator mode. Horizontal movement follows camera heading; free-flight
elevation remains independently controllable. Input velocity uses
frame-rate-independent exponential acceleration and damping, and diagonal input
is normalized. In both modes the optical centre is resolved against the same
bilinearly sampled height field as the GPU. Operator mode holds it 1.68 m above
solid terrain, rejects water and rises beyond a 0.35 m step/38 degree slope,
while free flight retains a 0.12 m lens-body clearance.

The height texture's geographic bounds identify its first and last samples,
not its outer texel edges. Both terrain vertex shaders therefore address
`(uv * (dimensions - 1) + 0.5) / dimensions`, exactly matching the CPU
interpolator. Roads, footways and planted surfaces also use the height-field
gradient as their lighting normal. This removes the former half-cell collision
offset and the flat-plate light response at road/levee joins without adding a
per-frame CPU terrain mesh pass.

The dated OSM road centrelines are now expanded as continuous strips rather
than unrelated segment rectangles. Adjacent quads share the same bounded miter
corners, eliminating cracks and overlaps at ordinary bends while clamping very
acute joins to 2.5 half-widths. Source spans longer than 12 m are split along
the unchanged mapped centreline. This caps the distance over which the vertex
shader interpolates terrain displacement to three typical height-map cells,
without tessellating the full 5 by 4 km terrain on the CPU.

Road semantics also survive the import: `footway`, `pedestrian`, and `steps`
use the footway material; `cycleway` uses the cycleway material; and `path` or
`bridleway` uses the trail material. The 2024-10-05 historical query contains
49 mapped stair ways and two `embankment=yes` trunk-link ways in the scene
bounds. Their plan positions are therefore retained.

At static-batch upload, the 49 ways comprise 108 terrain-sampling spans. The
official terrain resolves at least 0.12 m endpoint rise on 62 of them. Those
spans are terraced between their evidence-backed endpoint heights; per-corner
terrain compensation makes each rendered tread horizontal even though the
common vertex shader still adds the height map. The generated count targets
`2 * riser + tread = 0.63 m`, then enforces tread >= 0.28 m and riser <= 0.18 m
from the Korean accessibility guidance and regulation:
[CODIL mobility-facility guidance](https://www.codil.or.kr/filebank/construction/EI/CIGCEI710019/CIGCEI710019.pdf)
and [the enforcement-rule detail standard](https://www.law.go.kr/LSW/flDownload.do?flNm=%5B%EB%B3%84%ED%91%9C+1%5D+%ED%8E%B8%EC%9D%98%EC%8B%9C%EC%84%A4%EC%9D%98+%EA%B5%AC%EC%A1%B0%C2%B7%EC%9E%AC%EC%A7%88%EB%93%B1%EC%97%90+%EA%B4%80%ED%95%9C+%EC%84%B8%EB%B6%80%EA%B8%B0%EC%A4%80%28%EC%A0%9C2%EC%A1%B0%EC%A0%9C1%ED%95%AD%EA%B4%80%EB%A0%A8%29&flSeq=46489157&gubun=).
This produces 1,068 tread/riser pairs (13,092 vertices). The remaining 46
spans stay as their original draped decks because the height field does not
resolve a defensible rise. These are design-standard-constrained
reconstructions, not a claim that every 2024 tread was measured.

The OSM embankment tag marks a road carried on earthwork, not its cross-section.
The two tagged roads therefore follow official terrain and retain their mapped
width, but no shoulder height or slope is invented. Exact embankment and
unresolved stair profiles still require site survey or timestamp-matched
photogrammetry.

### NGII 1:1,000 structure evidence gate

Seoul identifies the NGII National Land Information Platform as the download
route for its most detailed 1:1,000 digital topographic map, with DXF v1.0 and
NGI v2.0 products. A platform search around Yeouido Hangang Park on 2026-08-03
identified four event-area sheets: Seoul2447 (`376082447`), Seoul2448
(`376082448`), Seoul2457 (`376082457`), and Seoul2458 (`376082458`). Every
currently listed 1:1,000 candidate was marked as produced in 2025. The checked
Seoul2447 history view returned no earlier history, and anonymous download did
not deliver the selected files. These observations and the unresolved
projected CRS are recorded in
`assets/yeouido_ngii_1000_source_manifest.json` rather than being hidden in a
developer note.

`tools/import_ngii_structures.py` is the reproducible boundary for files
obtained through an authorised platform session. It reads ASCII DXF `LINE`,
`LWPOLYLINE`, and legacy `POLYLINE/VERTEX` entities, records the SHA-256 of
every input, preserves closed rings, tessellates bulge arcs at no more than
five degrees per chord, and selects the published standard feature codes `C0050000`
(embankment), `F0030000` (cut/fill), and `F0040000` (retaining wall). The
confirmed projected CRS must be supplied explicitly; a GRS80 datum label is
not treated as an EPSG projection. The codes are cross-checked against the
[national-base-map feature-code report](https://www.codil.or.kr/filebank/original/RK/OTKCRK220915/OTKCRK220915.pdf).

The command refuses a source year later than 2024 unless
`--allow-post-event-source` is stated. Even when explicitly permitted, its
output remains labelled `official_post_event`. Same-year data is labelled
date-unverified because a production year alone cannot prove the state on
October 5. Source elevations are converted to height relative to the 2.79 m
scene datum; absent elevations remain JSON `null`. No retaining-wall height,
embankment cross-section, or mesh is inferred from a plan line. A later scene
merge requires either source top/bottom elevations or a separately cited site
survey or timestamp-matched photogrammetric measurement.

Example after the files and their CRS have been verified:

```powershell
python -m tools.import_ngii_structures <download-directory> `
  --source-crs <confirmed-EPSG-code> --source-year 2024
```

The source discovery policy follows the
[Seoul 1:1,000 digital-map notice](https://news.seoul.go.kr/gov/archives/528208)
and the [NGII platform](https://map.ngii.go.kr/ms/map/NlipMap.do). The former
[Seoul retaining-wall API](https://data.seoul.go.kr/dataList/OA-20499/S/1/datasetView.do)
is not substituted: its catalogue states that the service ended in 2022 and
covered urban-expressway installations rather than the complete Yeouido
riverside inventory.

### Survey-qualified wall and earthwork mesh merge

Normalized plan/elevation data is not rendered directly. The optional
`tools/merge_ngii_structures.py` stage requires a separate profile document
whose SHA-256 is locked to one normalized structure asset. Every profile must
carry a grade-A or grade-B `DataRecord`; grade-C modelled dimensions and
grade-D visual completion are rejected as geometry. A later-than-event asset
also meets a second explicit time gate at merge, even if its import was
previously allowed.

Two evidence-complete cases are implemented:

- `retaining_wall_face` requires the mapped source line to be independently
  identified as the wall top and the official terrain to be identified as its
  lower edge. It interpolates the surveyed top at no more than 4 m, samples the
  official terrain at each foot, and emits both windings because a map line
  carries no outward-face direction. A segment whose top is less than 0.08 m
  above the terrain is skipped rather than inverted.
- `surveyed_slope` requires two elevated lines of the same feature class,
  independently identified as crest and toe. The two edges are oriented and
  resampled by normalized station, then joined directly. No default slope
  angle, shoulder width, or toe depth exists in the code.

Source elevations are absolute relative to the event datum, whereas the static
renderer stores Y as an offset above its official terrain texture. The merge
therefore subtracts terrain elevation at every generated vertex; the GPU adds
the same terrain once during rendering. Lighting normals are computed from the
absolute surveyed surface, not from those storage offsets. This prevents both
double elevation and a false normal on sloping ground.

Retaining concrete and mixed turf/exposed earthwork receive dedicated material
slots. They share one `structure_vertices` batch and therefore add at most one
main-scene draw and one reflection draw, independent of source feature count.
The shipped scene currently holds an empty backward-compatible batch: no
profile can be authored honestly until authenticated source files and the
top/crest/toe semantics are obtained.

The profile document has this audited shape:

```json
{
  "schema_version": 1,
  "source_asset_sha256": "<exact normalized JSON SHA-256>",
  "profiles": [{
    "feature_id": "<stable imported feature id>",
    "mesh_kind": "retaining_wall_face",
    "source_edge_role": "top",
    "lower_edge_source": "official_terrain",
    "evidence": {
      "confidence_grade": "A",
      "source_id": "<identified survey>",
      "source_url": "<source URL>",
      "coordinate_reference_system": "<confirmed CRS>",
      "units": "m",
      "notes": "<how the line role was established>"
    }
  }]
}
```

The output path is mandatory, so an experimental merge cannot silently
overwrite the checked scene:

```powershell
python -m tools.merge_ngii_structures `
  --structures <normalized.json> --profiles <audited-profiles.json> `
  --output <candidate-scene.npz>
```

### Structure-reference camera registration

A photograph can identify whether a mapped line is a top, crest, or toe only
after it is registered to the same 3D frame. `simulator/photogrammetry.py`
implements that offline registration in the runtime's exact East-Up-South
camera convention. World points are transformed into an OpenCV-compatible
camera frame (`X` right, `Y` down, `Z` forward), projected with focal length,
sensor dimensions and principal point, then passed through the same
Brown-Conrady `k1,k2,p1,p2,k3` model used by the renderer's camera optics.

The solver fits camera East/Up/South position, yaw, pitch and roll with a robust
soft-L1 pixel loss. It requires at least six unique, grade-A/B control points
and an independently sourced grade-A/B image. The physical intrinsics remain
fixed: silently fitting an unknown focal length at the same time as pose would
allow geometry and lens errors to compensate for one another. Controls outside
the image are rejected before optimisation.
The authoring contract is machine-readable in
`assets/structure_reference_registration.schema.json`.

`tools/calibrate_structure_reference.py` writes every control-point residual
and passes a registration only when all of the following hold:

- the six-parameter Jacobian has rank 6 and the optimisation converged;
- RMSE is at most 2 px, p95 at most 3 px, and the worst point at most 5 px;
- the control bounding box covers at least 2% of the image, preventing a
  numerically precise but spatially local cluster from validating the frame;
- every point remains more than 0.05 m in front of the camera.

```powershell
python -m tools.calibrate_structure_reference <registration-input.json> `
  --output <registration-report.json>
```

A grade-B reconstructed structure profile must now name the registration ID
and exact report SHA-256. `merge_ngii_structures.py` independently reloads the
report, recalculates that checksum, repeats every numeric pass condition, and
requires its target date to match the normalized asset. A grade-A profile whose
top/crest/toe role is directly encoded by an official survey does not require a
photograph. A failed or absent registration therefore cannot become geometry
by merely changing the profile's confidence label.

```json
{
  "confidence_grade": "B",
  "registration_id": "event-view-01",
  "registration_report_sha256": "<exact report SHA-256>"
}
```

The held event photographs still lack published camera poses and physical
intrinsics sufficient to populate such a report. The stage therefore adds no
new scene triangles and makes no claim that an existing photograph is now
metric evidence; it establishes the gate that a future original/EXIF-backed
image must pass.

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
passes through Brown-Conrady lens distortion, an f/2.8 aperture, 90% lens
transmission, 1/60 s shutter, 5.9 um pixel area, wavelength-dependent RGB
photon energy and quantum efficiency, 45,000-electron full well, ISO 800
analogue gain, Poisson shot-noise approximation, read noise, cos^4 lens
falloff, and a von Kries white balance before ACES display mapping.

### White balance

The sensor stage was applying its spectral response and never undoing it.
Quantum efficiency (0.42 / 0.52 / 0.36) and photon energy — a red photon
carries less energy than a blue one, so a watt of red light frees more
electrons — together turn a neutral scene into electrons in the ratio
0.896 : 1 : 0.579. Every frame therefore left the camera with a yellow-green
cast that nothing corrected: the rendered frame's mean blue-to-red ratio was
**0.906**, blue *below* red on a night scene whose sky and lighting are not
yellow.

The balance is the reciprocal of the sensor's response to a reference
illuminant, a Planckian radiator at `white_balance_temperature_k`, with green
normalised to unity because exposure is set on the luminance-carrying channel.
The reference is computed by integrating Planck's law against the CIE 1931
colour matching functions — both published standards — rather than by the
`blackbody_rgb` curve fit, which is convenient for star hue but differs from
the integration by a few percent.

At the shipped 6504 K the gains are **(1.055, 1, 1.642)**. They are not simply
the reciprocal of the quantum efficiency because a 6504 K black body is not the
D65 *illuminant*: D65 carries solar and atmospheric line structure a Planckian
does not, a 5% difference the model keeps rather than rounds away. The frame's
mean blue rises by 3.7 display code values and its worst pixel by 35; the mean
blue-to-red ratio becomes 1.165.

6504 K is the default because it is the only value derivable from the
pipeline's own colour space: the renderer's radiance carries sRGB primaries, so
balancing there makes an sRGB-neutral scene render neutral and the stage a
correction rather than a look. The temperature is an **operator setting, not a
measurement** — a videographer shooting a warm-lit night city might plausibly
have set 3200-4000 K, which the config allows and which turns the whole frame
distinctly blue.

The balance sits **after** the full-well clamp, where a raw pipeline applies
it. A channel that saturates clips before the gains, which is why a clipped
burst shifts hue instead of staying neutral — the behaviour that produces
magenta suns in real footage.

### Lens distortion

Brown-Conrady radial and tangential distortion, in OpenCV's `k1 k2 p1 p2 k3`
convention on normalised image coordinates, because that is what any
calibration this project could obtain would report; adopting a private
convention would mean silently reinterpreting someone else's measurement.

**The shipped default is identity, and exactly so.** No calibration of the lens
exists and invented coefficients would put a fabricated optical claim into
every frame. With zero coefficients the shader's first inversion step is exact,
so the default camera path is bit-unperturbed rather than merely close.

The frame is rendered through an ideal pinhole, so forming the lens's image
means inverting the polynomial — which has no closed form. Five fixed-point
steps, matching OpenCV's `undistortPoints` and asserted equal between the
shader and `camera_optics.py`. Two properties are measured rather than assumed,
because a loaded calibration can break either:

- **Inverse residual.** The iteration converges for ordinary lenses (5e-8 at
  k1 = +0.08, 7e-6 at k1 = −0.12) but **not for a strong barrel**: k1 = −0.25
  leaves 3.1e-2, which would warp the image with nothing to signal it. V-23
  gates on the residual.
- **Frame coverage.** Barrel distortion pulls the image inward, so the output
  corners ask for scene an ideal render does not contain. k1 = −0.12 covers
  only 88% of the output without overscan. V-23 gates on coverage.

### Overscan

The renderer draws the scene over a wider field than the sensor covers, so a
lens that pulls the image inward still has something real to sample at the
corners. The factor is **derived from the loaded calibration**, not configured:
`LensDistortion.required_overscan` undistorts the output frame and returns how
far past the sensor's field those directions run. It is **exactly 1.0** for an
identity or pincushion lens, which is what keeps the shipped camera path
bit-identical — V-23 and V-24 hold their 0.63 and 0.62 code-value residuals
unchanged.

The field is widened **and the pixel count with it**. Widening the field alone
would hand the sensor a coarser image than its photosites sample, which is a
fidelity loss rather than a saving. The price is area, not width: k1 = −0.12
needs 1.090× the field and therefore **1.19× the pixels**; k1 = −0.18 needs
1.182× and **1.40×**. Every scene-resolution target follows, because the
renderer builds them from a `replace`d config; the display transform keeps the
sensor's own field and divides by the overscan when it samples. Human Vision
Mode has no lens distortion to undo but still gets a centred crop, so the
observer is shown their own field rather than the wider one.

Measured end to end with a synthetic calibration: coverage **0.888 → 1.000**
for k1 = −0.12, with the renderer's applied overscan matching the lens's
requirement exactly — V-23 gates that too, since a coverage of 1.0 means
nothing if measured against a render that never happened.

**A distorted lens costs sampling precision, and the tolerance says so.** With
an identity lens the source coordinate equals the output coordinate, every
fetch lands on a texel centre, and the GPU's bilinear filter degenerates to an
exact read — hence the 0.63 floor. Any distortion turns it into a real
interpolation and the CPU reference and the texture unit diverge by up to a
measured **2.26 code values**. That is filtering precision, not a model
difference: a **pincushion** lens, which needs no overscan and no change of
resolution at all, shows the same rise. V-23 therefore carries two bounds — 2
code values when the sampling is texel-aligned, 4 when it is not, at 1.8× the
measured floor. If a real calibration ever exceeds that, the fix is for the
display transform to do its own bilinear from four explicit fetches, the way
the peripheral blur now does its own mip interpolation.

The effect is not a formality when calibrated: k1 = −0.18 moves the frame
corner by 75 px horizontally at 1280 x 720.

`load_lens_calibration` requires source, licence, capture time, image size,
principal point, and coefficients, and **refuses a calibration whose principal
point is not the image centre** — an off-centre principal point displaces the
projection as well as the distortion, and applying half of a decentred
calibration would be worse than refusing it.

### What is deliberately not modelled

- **A camera colour matrix.** One converts a sensor's native spectral basis to
  a standard colour space. This renderer's values are *already* linear sRGB:
  `wavelength_rgb` and `blackbody_rgb` both emit sRGB, so every authored and
  derived colour is expressed there. A matrix built from the three channel
  wavelengths and the CIE observer has diagonal 2.47 / 1.45 / 1.77 — far from
  identity precisely because it would be a second conversion of colour already
  converted once. Closing this properly needs a spectral renderer and measured
  sensor sensitivities; neither is held, so the gap is recorded rather than
  filled with a plausible-looking matrix.
- **Defocus.** Derivable, and negligible here: at 24 mm f/2.8 with a 5.9 um
  circle of confusion the hyperfocal distance is 34.9 m, so everything beyond
  17.4 m is acceptably sharp and the whole scene is past 200 m.
- **Diffraction.** Also derivable, also negligible: the Airy radius at f/2.8
  and 550 nm is 1.9 um on the sensor, well inside one 5.9 um pixel.
- **Lateral chromatic aberration**, which is lens-specific and needs the same
  calibration the distortion does.
- **Distortion and white balance in Human Vision Mode.** Neither describes an
  eye. The observer path has no lens, and its equivalent of a white balance is
  chromatic adaptation, which is modelled on its own terms — see *Discounting
  the illuminant*.
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

Historical `landuse=grass`, `natural=grassland`, and turf-inferred sports
polygons seed 28,790 tuft anchors inside 1.2 km of the event origin. Each tuft
contains five narrow double-sided blades. Blade bases remain fixed, while tip
displacement follows a squared cantilever height profile, bounded wind
response, and two-frequency gust. Distant grass receives the same wind-aligned
travelling-wave normal field, and tree crowns receive a smaller height-weighted
sway. These paths share the atmospheric wind used by the JONSWAP river forcing.

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

The terrain/camera contact stage adds no CPU mesh traversal: camera height is
an O(1) four-texel interpolation, and the four extra GPU gradient samples run
only for draped road/ground vertices. A clean 360-frame integrated 3D run
measures 13.694 ms frame p95 and 10.164 ms visual p95, leaving 2.98 ms at the
60 Hz p95 boundary. Its p99 is 19.544 ms, so rare host/driver spikes still miss
one refresh and remain a frame-pacing target rather than being hidden by the
mean.

After replacing the 512-grid regional baseline with the official-constraint
1024 grid, a clean 360-frame integrated 3D run measures 12.775 ms frame p95,
9.641 ms visual p95, and 15.168 ms frame p99. The larger height texture adds no
draw calls or CPU traversal and remains 3.89 ms inside the 16.67 ms p95 budget
on the development laptop.

After the continuous dated-planimetry rebuild, the shipped road batch contains
35,340 six-vertex terrain-following spans (212,040 vertices), with a measured
maximum span of 12.00 m. A clean 360-frame integrated 3D run measures 14.496 ms
frame p95, 10.951 ms visual p95, and 15.459 ms frame p99. The additional
terrain samples and join geometry therefore remain 2.17 ms inside the 16.67 ms
p95 budget on the development laptop.

The evidence-gated stair expansion adds only 13,092 runtime vertices and no
draw call: its triangles are appended once to the existing road batch. A clean
360-frame integrated 3D run after this stage measures 10.833 ms frame p95,
8.378 ms visual p95, and 11.960 ms frame p99. The lower absolute values than
the prior run reflect normal host/driver variance; the relevant result is that
the extra stair geometry introduces no observed 60 FPS regression.

The NGII 1:1,000 discovery/import stage intentionally changes no runtime
geometry until authenticated, temporally qualified source data is available.
A clean 360-frame integrated 3D regression run measures 13.157 ms frame p95,
10.185 ms visual p95, and 14.381 ms frame p99. The frame p95 retains 3.51 ms
inside the 16.67 ms target; variation from the preceding run is host/driver
noise because this stage adds no draw, upload, or per-frame work.

The survey-qualified structure merge adds two material-table rows and a
backward-compatible empty structure batch to the shipped scene path. A clean
360-frame integrated 3D run measures 12.631 ms frame p95, 9.883 ms visual p95,
and 13.786 ms frame p99, retaining 4.04 ms at the 60 Hz p95 boundary. Because
the checked scene contains no unaudited structure vertices, this result proves
the dormant path and larger shader table do not regress the laptop target; it
does not predict the cost of a future populated NGII asset. Populated geometry
remains one main draw and one reflection draw, but must receive its own p95 and
triangle-count gate before replacing the shipped scene.

The camera-registration solver and checksum gate are offline-only and are not
imported by the interactive application. Two consecutive 360-frame integrated
3D runs after the full test workload measured frame p95 values of 17.131 and
16.654 ms. Their visual p95 values stayed at 12.419 and 12.327 ms, while
physics p95 stayed elevated at 5.597 and 5.686 ms. The first run misses the
16.67 ms target and the repeat retains only 0.016 ms. Because this stage changes
no runtime path, the measurement is not attributed to camera registration, but
it does prove the current laptop margin is not robust under the observed host
load/thermal state. This result is retained rather than replaced by the faster
preceding measurement; reducing physics-step p95 remains required before a
universal 60 Hz claim.

The following physics-p95 stage adds per-frame timings for world integration,
combustion-emission construction, grid injection, compute submission, and a
separate GL timestamp around the smoke solve. It does not change the 120 Hz
ballistic clock, 30 Hz plume clock, 32 x 24 x 10 MAC grid, or 24 pressure
iterations. The independent U/V/W advection, force, and pressure-projection
programs are instead fused into one compute dispatch per stage. This reduces a
default plume step from 25 to 19 dispatches without changing the equations or
their sampling locations.

Particle source reduction now writes the two interleaved grid channels through
their strided views. The previous channel `ravel()` calls produced detached
copies, so continuous star combustion was counted in diagnostics but not added
to the GPU source texture. A mass-and-thermal-energy conservation regression
now covers this path. The common all-inside case also avoids repeated boolean
copies before the two conservative `bincount` reductions.

Before these changes, the detailed 360-frame run measured 5.981 ms physics p95,
1.444 ms active smoke-submit p95, and 0.520 ms smoke GPU p95. Two final
360-frame runs measure physics p95 values of 5.096 and 5.498 ms, active submit
p95 values of 1.249 and 1.252 ms, and GPU p95 values of 0.469 and 0.465 ms. The
physics reduction is repeatable, but total frame p95 is 16.000 and 18.516 ms
because visual p95 varies from 11.713 to 14.173 ms under host/driver load. The
second run still misses 60 Hz, so the project does not yet claim a robust laptop
60 FPS guarantee.

The next latency work is ordered as follows: reduce allocation variance in the
world/star update, add real frame-pacing telemetry, then add active sparse-brick
dispatch beyond the current launch domain and calibrate smoke multiple
scattering. Dynamic resolution, if required, applies only to volumetric
radiance and reflection, never to trajectory, terrain, building geometry, or
the fixed physics clocks.

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
