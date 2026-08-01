"""Single source of truth for every declared validation metric.

Metrics that cannot run yet are declared here as data rather than as empty
implementation classes: the runner turns a declaration with no implementation
into ``NO_REFERENCE`` or ``NOT_IMPLEMENTED`` and reports which datasets are
missing. This keeps the report complete without generating placeholder code.

Tolerances and their justifications mirror ``VALIDATION_SPEC.md``.
"""

from __future__ import annotations

from .report import MetricSpec

# Dataset identifiers match DATA_PROVENANCE.md section 5.
DATASET_FIRING_TIMELINE = "firing_timeline"
DATASET_LAUNCH_POSITIONS = "launch_positions"
DATASET_REFERENCE_VIDEO = "reference_video"
DATASET_RIVER_STAGE = "han_river_stage"
DATASET_EXTERNAL_EPHEMERIS = "external_ephemeris"
DATASET_STAR_CATALOGUE = "star_catalogue"


V01 = MetricSpec(
    metric_id="V-01",
    title="Celestial azimuth and altitude against an external ephemeris",
    tolerance="0.05 deg in azimuth and altitude",
    physical_basis=(
        "Apparent refraction near the horizon varies by roughly 0.1 deg over "
        "realistic pressure and temperature spreads, and the lunar apparent "
        "semi-diameter is about 0.26 deg. Agreement at 0.05 deg therefore "
        "demonstrates the algorithm is correct while staying above the noise "
        "floor set by unmodelled local refraction."
    ),
    required_datasets=(DATASET_EXTERNAL_EPHEMERIS,),
)

V02 = MetricSpec(
    metric_id="V-02",
    title="Deterministic replay of clock, ballistics, and acoustics",
    tolerance="exact equality on CPU-side state",
    physical_basis=(
        "A software property, not a physical one. Exactness is achievable "
        "because playback position is an integer step count and every random "
        "draw comes from a named scenario seed. GPU float ordering is excluded "
        "by comparing CPU state only."
    ),
)

V03 = MetricSpec(
    metric_id="V-03",
    title="Geodetic round trip through the local tangent plane",
    tolerance="1 mm position residual",
    physical_basis=(
        "float64 ECEF retains roughly a micrometre over a 5 km baseline, so a "
        "1 mm bound sits three orders above numerical noise and far below any "
        "source accuracy. Exceeding it indicates a formula error rather than "
        "precision loss."
    ),
)

V04 = MetricSpec(
    metric_id="V-04",
    title="Camera reprojection error against landmark pixels",
    tolerance="2 px RMS at 1080p",
    physical_basis=(
        "2 px at 1080p with a 24 mm lens on a 36 mm sensor is about 0.08 deg, "
        "comparable to hand-clicked landmark identification error. A tighter "
        "bound would measure the annotator rather than the model."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO,),
)

V05 = MetricSpec(
    metric_id="V-05",
    title="Building silhouette pixel error",
    tolerance="IoU >= 0.90 for 63 City, IFC, and Parc.1",
    physical_basis=(
        "OSM footprint vertices are typically 1-3 m accurate; at 1.5 km that "
        "subtends about 0.1 deg, roughly 3 px. 0.90 IoU accommodates that "
        "without permitting a wrong height."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO,),
)

V06 = MetricSpec(
    metric_id="V-06",
    title="Burst centre position error against triangulation",
    tolerance="10 m horizontal, 15 m vertical",
    physical_basis=(
        "At about 1.5 km with a 10 m stereo baseline from typical spectator "
        "positions, depth uncertainty from a 2 px disparity error is already "
        "10-20 m. The tolerance is set by the reconstruction method's own "
        "uncertainty, not by ambition."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO, DATASET_LAUNCH_POSITIONS),
)

V07 = MetricSpec(
    metric_id="V-07",
    title="Maximum shell radius and burn duration error",
    tolerance="12 percent radius, 0.25 s duration",
    physical_basis=(
        "Measured star burn time varies by about 10 percent between nominally "
        "identical stars, so demanding better than the manufacturing spread "
        "would be fitting noise."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO, DATASET_FIRING_TIMELINE),
)

V08 = MetricSpec(
    metric_id="V-08",
    title="Visual difference against reference frames (linear HDR)",
    tolerance="reported, not gated, until a calibrated reference exists",
    physical_basis=(
        "An SDR display cannot represent firework luminance, so a gated pixel "
        "metric on the tone-mapped image would measure the tone mapper. The "
        "comparison must read the linear HDR buffer before any display "
        "transform."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO,),
    requires_opengl=True,
)

V09 = MetricSpec(
    metric_id="V-09",
    title="Smoke advection vector error against optical flow",
    tolerance="25 percent speed, 20 deg direction",
    physical_basis=(
        "The plume grid is about 12 m per cell, so sub-grid eddies are "
        "unresolved by construction and only the large coherent motion is "
        "comparable."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO,),
)

V10 = MetricSpec(
    metric_id="V-10",
    title="Water reflection position error",
    tolerance="5 m along-shore",
    physical_basis=(
        "Dominated by the water plane height, which is currently a DEM median "
        "rather than a gauge reading. This metric is in effect a test of the "
        "river stage dataset."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO, DATASET_RIVER_STAGE),
)

V11 = MetricSpec(
    metric_id="V-11",
    title="Blast propagation model self-consistency",
    tolerance=(
        "Sedov exponent 0.4 +/- 0.005; shock/sound crossing within 1 percent; "
        "arrival delay within 1 ms of the analytic two-phase solution"
    ),
    physical_basis=(
        "The Sedov-Taylor strong-shock solution gives radius proportional to "
        "t^(2/5) exactly, and the acoustic phase is a closed-form path over an "
        "effective sound speed. Residuals here are numerical, not physical, so "
        "the bounds are tight. This checks the model against its own analytic "
        "solution and needs no external data; comparing the predicted delay "
        "against a real recording is V-17."
    ),
)

V12 = MetricSpec(
    metric_id="V-12",
    title="Frame budget at 60 Hz",
    tolerance="frame p95 < 16.67 ms",
    physical_basis=(
        "16.67 ms is the 60 Hz display period. The figure is machine specific: "
        "it must be recorded with machine, driver, resolution, backend, frame "
        "count, and date, and must not be compared across machines."
    ),
    requires_opengl=True,
)

V13 = MetricSpec(
    metric_id="V-13",
    title="CPU simulation-state memory footprint",
    tolerance="reported, not gated (no ceiling has been measured yet)",
    physical_basis=(
        "An exact sum of the simulator's state array allocations. This is not "
        "peak process RSS: NumPy allocates large buffers outside the Python "
        "allocator, and driver-side GPU allocations are not visible here."
    ),
)

V14 = MetricSpec(
    metric_id="V-14",
    title="Physical conservation in combustion and the plume solver",
    tolerance=(
        "smoke and thermal yields within 2e-5 relative; radiated energy within "
        "1e-3 relative; pressure projection must reduce divergence"
    ),
    physical_basis=(
        "Yield closure is exact by construction up to float32 accumulation "
        "over the burn, so 2e-5 is an accumulation bound. Radiated energy "
        "carries an additional per-star residual from the deterministic "
        "0.97 +/- 0.03 combustion modulation: over a 2.25 s burn the 53 rad/s "
        "term completes about 19 cycles, leaving an uncancelled fraction of "
        "roughly 0.03 / (0.97 * 2 * pi * 19) = 2.6e-4, which measurement "
        "confirms at 2.9e-4."
    ),
)

V15 = MetricSpec(
    metric_id="V-15",
    title="Horizontal coordinate transform cross-check",
    tolerance="1e-9 deg in azimuth and altitude",
    physical_basis=(
        "An independent spherical-trigonometry transform from apparent RA/Dec "
        "and Greenwich apparent sidereal time is compared against the library "
        "horizon routine with refraction disabled. Both are pure rotations of "
        "the same input, so agreement is limited only by float64 precision; "
        "measurement gives 2e-13 deg. The 1e-9 deg gate sits four orders above "
        "that and still catches any sign, hemisphere, or hour-angle convention "
        "error. This validates the project's use of the ephemeris, not the "
        "ephemeris itself."
    ),
)

V16 = MetricSpec(
    metric_id="V-16",
    title="East-Up-South direction vector consistency",
    tolerance="1e-5 deg round trip, unit length within 1e-6",
    physical_basis=(
        "The renderer consumes celestial directions as EUS unit vectors, so a "
        "convention error would move every shadow and reflection without "
        "changing the reported azimuth. Round-trip precision is limited by the "
        "float32 storage of the direction vector, which carries about 7 "
        "significant digits."
    ),
)


V17 = MetricSpec(
    metric_id="V-17",
    title="Flash-to-boom delay against a recording",
    tolerance="0.10 s",
    physical_basis=(
        "0.10 s at about 345 m/s is roughly 35 m of path length, comparable to "
        "the burst-position uncertainty in V-06 and to the +/- 0.6 m/s "
        "sound-speed spread across the observed temperature and humidity "
        "range. A tighter bound would be unsubstantiated."
    ),
    required_datasets=(DATASET_REFERENCE_VIDEO, DATASET_LAUNCH_POSITIONS),
)


V18 = MetricSpec(
    metric_id="V-18",
    title="Shell library integrity",
    tolerance=(
        "every profile launches, breaks, and emits no more than its declared "
        "composition mass (1e-3 relative); every required break pattern is "
        "present; every profile is deterministic under a fixed seed"
    ),
    physical_basis=(
        "A break pattern is a velocity distribution, not a new solver, so a "
        "profile can only fail by declaring a budget it then exceeds or by "
        "producing non-finite geometry. Mass closure is exact by construction "
        "up to float32 accumulation across the burn, which bounds the residual "
        "at roughly 1e-3 for the largest shipped star counts. A secondary "
        "break is the one place this can silently break: its carriers expire "
        "over many steps, so each batch must take a share of the declared mass "
        "rather than the whole of it."
    ),
)


V19 = MetricSpec(
    metric_id="V-19",
    title="Rayleigh optical depth against published values",
    tolerance="1 percent relative across 400-700 nm",
    physical_basis=(
        "The Bodhaine et al. (1999) rational fit reproduces tabulated Rayleigh "
        "optical depths to better than 0.1 percent in the visible, so a 1 "
        "percent bound catches a transcription or unit error while tolerating "
        "the spread between published tabulations. The scattering must also "
        "follow an inverse fourth power of wavelength to within the dispersion "
        "of air's refractive index, and must scale linearly with station "
        "pressure because the optical depth is proportional to the molecular "
        "column above the observer."
    ),
)

V20 = MetricSpec(
    metric_id="V-20",
    title="Star catalogue astrometry against the ephemeris library",
    tolerance="0.5 arcsecond",
    physical_basis=(
        "The vectorised catalogue path and the library's per-star path apply "
        "the same precession, nutation, and diurnal rotation, so they should "
        "agree to numerical precision once annual aberration is included. "
        "0.5 arcsecond is far below the 1-3 arcsecond position uncertainty of "
        "the source catalogue itself, so exceeding it indicates a frame, sign, "
        "or epoch error rather than data quality. Without aberration the same "
        "comparison disagrees by up to 18 arcseconds."
    ),
    required_datasets=(DATASET_STAR_CATALOGUE,),
)


V21 = MetricSpec(
    metric_id="V-21",
    title="Declared asset checksums match the shipped files",
    tolerance="exact SHA-256 match for every record that declares one",
    physical_basis=(
        "Not a physical bound: a checksum either matches or it does not. It is "
        "gated because a provenance record that names the wrong file is worse "
        "than one that names none — it asserts an audit trail that does not "
        "hold. Derived assets are regenerated by tools in this repository, so "
        "a stale checksum is a routine consequence of ordinary work rather "
        "than an unlikely accident."
    ),
)


V22 = MetricSpec(
    metric_id="V-22",
    title="Rendered aerial perspective against its CPU reference",
    tolerance=(
        "opaque non-water pixels within 2e-3 of the predicted composite, "
        "relative to the frame's own peak radiance; both skies unchanged "
        "exactly; the reflected skyline and bright water must lose radiance"
    ),
    physical_basis=(
        "The composite claims L = L_object * T + L_air * (1 - T) with T the "
        "per-channel transmittance through an exponential atmosphere. "
        "Rendering the same frame with the atmosphere removed recovers "
        "L_object, so the claim can be predicted on the CPU and subtracted "
        "rather than judged by eye. The bound is set by the linear buffer's "
        "half-float storage: 2^-11 is 4.9e-4 per stored value and the "
        "prediction combines three of them, so 2e-3 is quantisation and "
        "anything larger is a formula difference. Sky pixels carry the "
        "airlight of an infinite path already and must be untouched, which is "
        "an exact rather than approximate requirement. "
        "Water is excluded from that gate and checked by sign instead: it "
        "shows a reflection that carries its own atmospheric path, so the "
        "vacuum render does not recover its object radiance. The reflected "
        "skyline reaches the eye by way of the river and its path is "
        "therefore longer than the direct one, so it must lose radiance — "
        "the direction a mirrored-path sign error would reverse, and the one "
        "thing the direct-path residual cannot see."
    ),
    requires_opengl=True,
)


V23 = MetricSpec(
    metric_id="V-23",
    title="Display transform against its CPU reference",
    tolerance=(
        "rendered frame within 2 display code values of the predicted "
        "transform; lens inversion converged and the frame fully covered"
    ),
    physical_basis=(
        "The linear HDR and bloom buffers are the display shader's only image "
        "inputs, so running the same chain — lens distortion, cos^4 falloff, "
        "photon-to-electron conversion, full-well clipping, white balance, "
        "ACES, gamma — on the CPU predicts the frame the GPU should have "
        "produced. The bound is 8-bit quantisation: rounding alone gives a "
        "quarter of a code value on average and about half at worst, so 2 "
        "code values is three times that margin while sitting far below any "
        "real defect. Sensor noise is disabled because its distribution is "
        "the claim, not its per-pixel value. This is the only metric that "
        "reads the display-referred image, and only because the stage under "
        "test is what produces it."
    ),
    requires_opengl=True,
)


V24 = MetricSpec(
    metric_id="V-24",
    title="Observer transform against its CPU reference",
    tolerance=(
        "rendered frame within 2 display code values of the predicted "
        "transform; the adapting white global and at unit luminance; the "
        "degree of adaptation strictly between 0 and 1"
    ),
    physical_basis=(
        "The sibling of V-23 for Human Vision Mode. The linear HDR, bloom, and "
        "adaptation buffers are the shader's only image inputs, so running the "
        "same chain — pupil gain, local adaptation, CAT02 chromatic "
        "adaptation, the mesopic mix, ACES, gamma — on the CPU predicts the "
        "frame the GPU should have produced. The bound is 8-bit quantisation, "
        "on the same reasoning as V-23. The structural checks are what a "
        "plausible-looking error would evade: a transposed cone matrix, an "
        "adapting white that stopped being global, or one that stopped "
        "carrying unit luminance and so silently changed brightness. "
        "Peripheral acuity and the glare tail are switched off for the "
        "measurement and remain unverified: both are spatial and predicting "
        "them would measure GPU mip generation rather than the model."
    ),
    requires_opengl=True,
)


CATALOGUE: tuple[MetricSpec, ...] = (
    V01, V02, V03, V04, V05, V06, V07, V08,
    V09, V10, V11, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23,
    V24,
)

BY_ID: dict[str, MetricSpec] = {spec.metric_id: spec for spec in CATALOGUE}
