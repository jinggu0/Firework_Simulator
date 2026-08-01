"""Human Vision Mode: what an observer sees, as distinct from what a camera records.

The renderer has carried a physically dimensioned camera since early on —
aperture, shutter, quantum efficiency, full well, read noise. None of that
describes an eye. A camera integrates for a fixed exposure and clips at full
well; an eye has a pupil that tracks the field, two photoreceptor systems with
different spectral sensitivities and time constants, an optical medium that
scatters light across the retina, and acuity that falls away from fixation.

This module computes the observer state each frame as a small set of scalars,
so the display shader consumes psychophysics rather than reimplementing it.
Every quantity here is named from a published model; where a value is an
estimate rather than a measurement it says so.

References
----------
Stanley P.A., Davies A.K., "The effect of field of view size on steady-state
pupil diameter", Ophthalmic and Physiological Optics 15(6), 1995 — pupil
diameter against adapting luminance and field size.

Watson A.B., Yellott J.I., "A unified formula for light-adapted pupil size",
Journal of Vision 12(10), 2012 — the formulation the above is used through,
including the Stiles-Crawford corrected troland.

Pattanaik S.N., Tumblin J., Yee H., Greenberg D.P., "Time-dependent visual
adaptation for fast realistic image display", SIGGRAPH 2000 — asymmetric light
and dark adaptation time constants.

CIE 191:2010, "Recommended system for mesopic photometry based on visual
performance" — the 0.005 to 5 cd/m2 mesopic transition range.

Vos J.J., van den Berg T.J.T.P., CIE 135/1-1999 — disability glare; the
Stiles-Holladay inverse-square term is the dominant contribution for a young
eye at the angles that matter here.

CIE 159:2004, "A colour appearance model for colour management systems:
CIECAM02" — the CAT02 cone response and the degree-of-adaptation relation.

Fairchild M.D., Reniff L., "Time course of chromatic adaptation for
colour-appearance judgments", Journal of the Optical Society of America A
12(5), 1995 — chromatic adaptation roughly 90 percent complete after 60 s.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# -- mesopic range -----------------------------------------------------------

SCOTOPIC_CEILING_CD_M2 = 0.005
"""Below this adapting luminance vision is rod-only (CIE 191:2010)."""

PHOTOPIC_FLOOR_CD_M2 = 5.0
"""Above this adapting luminance vision is cone-only (CIE 191:2010)."""

PHOTOPIC_PEAK_NM = 555.0
"""Peak of the photopic luminous efficiency function V(lambda)."""

SCOTOPIC_PEAK_NM = 507.0
"""Peak of the scotopic luminous efficiency function V'(lambda).

The 48 nm separation is the Purkinje shift: as the eye moves into rod vision,
short wavelengths gain relative to long ones, so blues hold their brightness as
reds darken.
"""

# -- adaptation --------------------------------------------------------------

LIGHT_ADAPTATION_TIME_S = 0.4
"""Time constant for adapting to an increase in luminance.

Light adaptation is fast: the visual system recovers within a second of a step
increase. A firework burst is well inside this, which is why a burst dazzles.
"""

DARK_ADAPTATION_TIME_S = 120.0
"""Time constant for adapting to a decrease in luminance.

Cone dark adaptation completes over a few minutes; the rod branch takes tens of
minutes. One cone-scale constant is used because the show is seventy minutes
long and the observer never leaves the mesopic range, so the rod branch would
not complete either way.
"""

# -- chromatic adaptation ----------------------------------------------------

CHROMATIC_ADAPTATION_TIME_S = 26.06
"""Time constant for the adapting white to follow the field's chromaticity.

Fairchild and Reniff (1995) measured chromatic adaptation as roughly 90 percent
complete after 60 s, which for a single exponential is ``60 / ln(10)``. They
also report a fast receptoral component within the first second; that two-phase
structure is **not** modelled, so a step change here is slower to begin and
faster to finish than the measurement.

The slowness is the point rather than a limitation. It is why a two-second
burst is *not* discounted: the observer sees a green shell as green because the
adapting white barely moves in two seconds. A fast constant would desaturate
every break, which would be a modelling error rather than a subtle one.
"""

DARK_SURROUND_FACTOR = 0.8
"""CIECAM02 ``F`` for a dark surround, which is what a night show is.

The other tabulated values are 1.0 for an average surround and 0.9 for dim.
``F`` caps the degree of adaptation: an observer in the dark discounts the
illuminant less completely than one in a lit room.
"""


def degree_of_adaptation(
    adapting_luminance_cd_m2: float,
    surround_factor: float = DARK_SURROUND_FACTOR,
) -> float:
    """CIECAM02 degree of adaptation ``D``.

    ``D = F [1 - (1/3.6) exp((-L_A - 42) / 92)]``. Adaptation is never complete:
    even in bright light ``D`` reaches only ``F``, and in the dark it falls
    further. Over the luminance range this show spans it stays near 0.66, so
    the observer discounts about two thirds of the illuminant's colour.

    **Extrapolated below its calibration range.** The relation was fitted in
    photopic conditions and this observer is mesopic. The mesopic mix
    independently suppresses the whole chromatic path as the cone contribution
    falls, so the extrapolation is bounded in effect as well as flagged here.
    """

    exponent = (-max(adapting_luminance_cd_m2, 0.0) - 42.0) / 92.0
    degree = surround_factor * (1.0 - math.exp(exponent) / 3.6)
    return min(max(degree, 0.0), 1.0)

# -- optics of the eye -------------------------------------------------------

FIELD_AREA_DEG2 = 600.0
"""Adapting field solid angle used by the pupil model, in square degrees.

Roughly a 28 degree field, which is the part of the scene a viewer's adaptation
actually tracks rather than the full peripheral extent.
"""

GLARE_CONSTANT = 10.0
"""Stiles-Holladay glare constant, in the equation ``L_veil = k E / theta^2``.

``E`` is the illuminance at the eye from the glare source in lux and ``theta``
is its angular separation from fixation in degrees. The value is for a young,
lightly pigmented eye.
"""

ACUITY_E2_DEG = 2.5
"""Eccentricity at which foveal acuity has halved.

From the cortical magnification relation ``acuity = 1 / (1 + e / E2)``. Values
between 2 and 3 degrees are reported for resolution tasks.
"""

MAXIMUM_PERIPHERAL_BLUR_LOD = 4.0
"""Cap on the mip bias peripheral acuity may request.

Bounds the cost and prevents the far periphery collapsing to a single texel.
"""


def pupil_diameter_mm(
    adapting_luminance_cd_m2: float, field_area_deg2: float = FIELD_AREA_DEG2
) -> float:
    """Steady-state pupil diameter from Stanley and Davies (1995).

    Returns roughly 7.6 mm under starlight and 3.5 mm under a bright interior,
    which brackets the range a dark-adapted observer at a fireworks display
    moves through.
    """

    luminance = max(adapting_luminance_cd_m2, 1e-8)
    product = luminance * max(field_area_deg2, 1e-6) / 846.0
    response = product**0.41
    return 7.75 - 5.75 * (response / (response + 2.0))


def pupil_area_mm2(diameter_mm: float) -> float:
    return math.pi * (diameter_mm * 0.5) ** 2


def retinal_illuminance_td(
    adapting_luminance_cd_m2: float, pupil_diameter_mm_value: float
) -> float:
    """Retinal illuminance in trolands, with the Stiles-Crawford correction.

    Light entering near the pupil margin is less effective at exciting cones
    than light entering the centre, so a wide pupil delivers less than its
    geometric area suggests. The correction follows Watson and Yellott (2012).
    """

    area = pupil_area_mm2(pupil_diameter_mm_value)
    effective = (
        area
        * (
            1.0
            - 0.017_0 * (pupil_diameter_mm_value - 4.9) ** 2
            + 0.000_2 * (pupil_diameter_mm_value - 4.9) ** 4
        )
    )
    return max(adapting_luminance_cd_m2, 0.0) * max(effective, 1e-6)


def mesopic_factor(adapting_luminance_cd_m2: float) -> float:
    """Cone contribution in ``[0, 1]`` across the mesopic range.

    Zero is rod-only vision, which is achromatic because there is a single rod
    photopigment and no opponent signal to carry hue. One is cone-only.
    """

    if adapting_luminance_cd_m2 <= SCOTOPIC_CEILING_CD_M2:
        return 0.0
    if adapting_luminance_cd_m2 >= PHOTOPIC_FLOOR_CD_M2:
        return 1.0
    # Logarithmic, because the transition spans three decades of luminance.
    span = math.log10(PHOTOPIC_FLOOR_CD_M2) - math.log10(SCOTOPIC_CEILING_CD_M2)
    position = (
        math.log10(adapting_luminance_cd_m2)
        - math.log10(SCOTOPIC_CEILING_CD_M2)
    ) / span
    return min(max(position, 0.0), 1.0)


def acuity_fraction(eccentricity_deg):
    """Resolvable spatial frequency relative to the fovea.

    Accepts an array as well as a scalar, so a reference implementation can
    evaluate it over a whole frame rather than reimplementing the relation.
    """

    return 1.0 / (1.0 + np.maximum(eccentricity_deg, 0.0) / ACUITY_E2_DEG)


def veiling_luminance_cd_m2(
    source_illuminance_lux: float, separation_deg: float
) -> float:
    """Stiles-Holladay disability glare.

    Light scattered in the ocular media lays a veil over the retinal image,
    which is why a bright burst washes out the surrounding scene rather than
    simply appearing bright. Diverges at zero separation, so the angle is
    floored at a tenth of a degree — about the angular size of the burst core.
    """

    theta = max(separation_deg, 0.1)
    return GLARE_CONSTANT * max(source_illuminance_lux, 0.0) / (theta * theta)


def adapt(
    current_cd_m2: float, target_cd_m2: float, dt_s: float
) -> float:
    """Advance the adaptation state one frame.

    Asymmetric by design: brightening is fast and darkening is slow, which is
    why an observer is dazzled instantly by a burst and then spends the gap
    between shells recovering.
    """

    if dt_s <= 0.0:
        return current_cd_m2
    brightening = target_cd_m2 > current_cd_m2
    tau = LIGHT_ADAPTATION_TIME_S if brightening else DARK_ADAPTATION_TIME_S
    response = 1.0 - math.exp(-dt_s / tau)
    return current_cd_m2 + (target_cd_m2 - current_cd_m2) * response


@dataclass(slots=True)
class HumanVisionState:
    """The observer's adaptation state, advanced once per frame."""

    adapting_luminance_cd_m2: float = 0.02
    """Starts near a moonless suburban sky, the state before the show begins."""

    gaze_uv: tuple[float, float] = (0.5, 0.5)
    """Fixation point in normalised screen coordinates.

    Fixed at the screen centre because no gaze tracking is available. A viewer
    watching a burst fixates it, so this is the right default for the moments
    that matter and wrong during the gaps.
    """

    def update(self, scene_illuminance_lux: float, dt_s: float) -> None:
        """Track the scene's ambient illuminance.

        The adapting luminance of a diffusely lit field is its illuminance
        divided by pi. Driving adaptation from the scene's own computed
        illuminance avoids a GPU readback, which would stall the frame.
        """

        target = max(scene_illuminance_lux, 0.0) / math.pi
        self.adapting_luminance_cd_m2 = adapt(
            self.adapting_luminance_cd_m2, target, dt_s
        )

    # -- derived quantities ------------------------------------------------

    @property
    def pupil_diameter_mm(self) -> float:
        return pupil_diameter_mm(self.adapting_luminance_cd_m2)

    @property
    def retinal_illuminance_td(self) -> float:
        return retinal_illuminance_td(
            self.adapting_luminance_cd_m2, self.pupil_diameter_mm
        )

    @property
    def cone_fraction(self) -> float:
        return mesopic_factor(self.adapting_luminance_cd_m2)

    @property
    def chromatic_degree(self) -> float:
        """How completely the illuminant's colour is discounted.

        The adapting *white* itself is tracked on the GPU, in the same buffer
        as the local luminance: it is the field's own average chromaticity, and
        reading it back to the CPU would stall the frame for a value the shader
        is about to use anyway.
        """

        return degree_of_adaptation(self.adapting_luminance_cd_m2)

    @property
    def pupil_gain(self) -> float:
        """Retinal illuminance relative to a 3 mm photopic reference pupil.

        This is the eye's own aperture control, and it replaces the camera's
        fixed f-number: a dark-adapted pupil gathers roughly five times the
        light of a photopic one.
        """

        reference = pupil_area_mm2(3.0)
        return pupil_area_mm2(self.pupil_diameter_mm) / reference

    def uniforms(self) -> dict[str, float]:
        """Scalars the display shader consumes."""

        return {
            "adapting_luminance_cd_m2": self.adapting_luminance_cd_m2,
            "pupil_gain": self.pupil_gain,
            "cone_fraction": self.cone_fraction,
            "chromatic_degree": self.chromatic_degree,
            "glare_constant": GLARE_CONSTANT,
            "acuity_e2_deg": ACUITY_E2_DEG,
            "maximum_blur_lod": MAXIMUM_PERIPHERAL_BLUR_LOD,
            "gaze_u": self.gaze_uv[0],
            "gaze_v": self.gaze_uv[1],
        }

    def summary(self) -> dict[str, float | str]:
        return {
            "adapting_luminance_cd_m2": self.adapting_luminance_cd_m2,
            "pupil_diameter_mm": self.pupil_diameter_mm,
            "retinal_illuminance_td": self.retinal_illuminance_td,
            "cone_fraction": self.cone_fraction,
            "chromatic_degree": self.chromatic_degree,
            "regime": (
                "scotopic"
                if self.cone_fraction <= 0.0
                else "photopic"
                if self.cone_fraction >= 1.0
                else "mesopic"
            ),
        }
