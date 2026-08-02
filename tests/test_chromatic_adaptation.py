"""Chromatic adaptation: the cone transform, the degree, and the time course.

The GPU side is checked by V-24, which reproduces the observer transform on the
CPU from the linear buffers. These tests cover the model that reference is
built from, and in particular the two things a transposition or a sign error
would leave looking plausible: the cone matrix and luminance preservation.
"""

import math
import re

import numpy as np
import pytest

from simulator import shaders
from simulator.color import (
    CAT02_LMS_TO_LINEAR_SRGB,
    LINEAR_SRGB_TO_CAT02_LMS,
    LINEAR_SRGB_TO_XYZ,
    XYZ_TO_CAT02_LMS,
    _XYZ_TO_LINEAR_SRGB,
    chromatic_adaptation_gains,
    chromatically_adapt,
    planckian_linear_srgb,
)
from simulator.human_vision import (
    CHROMATIC_ADAPTATION_TIME_S,
    DARK_SURROUND_FACTOR,
    HumanVisionState,
    degree_of_adaptation,
)

PHOTOPIC_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])
NEUTRAL = np.ones(3)


# --- the colour spaces ------------------------------------------------------


def test_the_srgb_matrices_are_a_matched_pair() -> None:
    # The forward matrix is the published one, not a numerical inverse, so the
    # round trip is only as tight as the four digits both are quoted to.
    round_trip = _XYZ_TO_LINEAR_SRGB @ LINEAR_SRGB_TO_XYZ
    assert np.abs(round_trip - np.eye(3)).max() < 2e-4


def test_luminance_row_matches_the_weights_the_shaders_use() -> None:
    # If these drifted apart, "luminance preserving" would stop meaning the
    # same thing on the two sides of the pipeline.
    assert np.allclose(LINEAR_SRGB_TO_XYZ[1], PHOTOPIC_WEIGHTS)
    assert LINEAR_SRGB_TO_XYZ[1].sum() == pytest.approx(1.0)


def test_cat02_is_the_published_matrix() -> None:
    # Transcribed rather than derived, so it is worth pinning: the rows sum to
    # 1 for equal-energy XYZ, which is the normalisation CIECAM02 states.
    assert XYZ_TO_CAT02_LMS.shape == (3, 3)
    assert np.allclose(XYZ_TO_CAT02_LMS.sum(axis=1), [1.0, 1.0, 1.0])


def test_the_cone_round_trip_is_exact() -> None:
    assert np.abs(
        CAT02_LMS_TO_LINEAR_SRGB @ LINEAR_SRGB_TO_CAT02_LMS - np.eye(3)
    ).max() < 1e-12


def test_the_shader_carries_the_same_cone_matrices() -> None:
    # GLSL mat3 literals are column-major, so a matrix written row-wise
    # compiles, runs, and produces a plausible image. Extracting and comparing
    # is the only way that error surfaces.
    source = shaders.source("human_vision.frag")
    for name, expected in (
        ("LINEAR_SRGB_TO_LMS", LINEAR_SRGB_TO_CAT02_LMS),
        ("LMS_TO_LINEAR_SRGB", CAT02_LMS_TO_LINEAR_SRGB),
    ):
        match = re.search(
            rf"const mat3 {name} = mat3\(([^)]*)\);", source, re.DOTALL
        )
        assert match is not None, name
        values = [float(v) for v in match.group(1).replace("\n", "").split(",")]
        assert len(values) == 9, name
        # Column-major: reshaping by columns then transposing recovers rows.
        declared = np.array(values).reshape(3, 3).T
        assert np.abs(declared - expected).max() < 1e-7, name


# --- the degree of adaptation ----------------------------------------------


def test_adaptation_is_never_complete() -> None:
    # CIECAM02 D caps at the surround factor, and a dark surround caps at 0.8.
    # The bound is only reached asymptotically: the exponential underflows
    # somewhere above a few thousand cd/m2, far outside anything this show sees.
    for luminance in (0.0, 0.01, 1.0, 100.0, 10_000.0):
        assert 0.0 < degree_of_adaptation(luminance) <= DARK_SURROUND_FACTOR
    for luminance in (0.0, 0.01, 1.0, 100.0):
        assert degree_of_adaptation(luminance) < DARK_SURROUND_FACTOR


def test_a_brighter_field_is_discounted_more_completely() -> None:
    values = [degree_of_adaptation(L) for L in (0.001, 0.1, 10.0, 1_000.0)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_the_degree_matches_the_published_relation() -> None:
    # D = F [1 - (1/3.6) exp((-L_A - 42) / 92)], evaluated by hand.
    expected = 0.8 * (1.0 - math.exp((-20.0 - 42.0) / 92.0) / 3.6)
    assert degree_of_adaptation(20.0) == pytest.approx(expected)


def test_a_lit_surround_discounts_more_than_a_dark_one() -> None:
    assert degree_of_adaptation(1.0, 1.0) > degree_of_adaptation(1.0, 0.8)


def test_the_show_sits_near_two_thirds_adaptation() -> None:
    # The observer never leaves a narrow luminance band, so D barely moves.
    band = [degree_of_adaptation(L) for L in (0.02, 0.4, 5.0)]
    assert all(0.65 < value < 0.68 for value in band), band


# --- the von Kries step -----------------------------------------------------


def test_no_adaptation_is_exactly_the_identity() -> None:
    # The stage switches off by its own physics rather than by a flag.
    gains = chromatic_adaptation_gains(np.array([1.3, 0.9, 0.8]), 0.0)
    assert np.allclose(gains, 1.0)


def test_a_neutral_field_needs_no_correction() -> None:
    gains = chromatic_adaptation_gains(NEUTRAL, 0.66)
    assert np.allclose(gains, 1.0, atol=1e-12)


def test_a_warm_field_is_discounted_by_lifting_the_short_channel() -> None:
    # A scene lit warm does not look uniformly orange, because the cone gains
    # rescale until the dominant light reads closer to neutral.
    warm = planckian_linear_srgb(3_000.0)
    gains = chromatic_adaptation_gains(warm, 1.0)
    assert gains[2] > gains[1] > gains[0]
    adapted = chromatically_adapt(warm / (warm @ PHOTOPIC_WEIGHTS), warm, 1.0)
    assert np.allclose(adapted, 1.0, atol=1e-9)


def test_a_cool_field_is_discounted_the_other_way() -> None:
    gains = chromatic_adaptation_gains(planckian_linear_srgb(20_000.0), 1.0)
    assert gains[0] > gains[1] > gains[2]


def test_luminance_is_preserved_exactly_for_the_adapting_white() -> None:
    # Both whites are normalised to unit Y and Y is linear in LMS, so the
    # adapted white is a convex combination of two unit-Y whites. Brightness
    # adaptation is a separate stage and must not be disturbed here.
    for temperature in (2_000.0, 4_000.0, 6_504.0, 12_000.0):
        white = planckian_linear_srgb(temperature)
        white = white / (white @ PHOTOPIC_WEIGHTS)
        for degree in (0.0, 0.3, 0.66, 1.0):
            adapted = chromatically_adapt(white, white, degree)
            assert adapted @ PHOTOPIC_WEIGHTS == pytest.approx(1.0, rel=1e-9)


def test_partial_adaptation_lies_between_none_and_complete() -> None:
    white = planckian_linear_srgb(3_000.0)
    none = chromatic_adaptation_gains(white, 0.0)
    partial = chromatic_adaptation_gains(white, 0.66)
    complete = chromatic_adaptation_gains(white, 1.0)
    assert np.all(np.minimum(none, complete) <= partial + 1e-12)
    assert np.all(partial <= np.maximum(none, complete) + 1e-12)


def test_the_degree_is_clamped_rather_than_extrapolated() -> None:
    white = planckian_linear_srgb(3_000.0)
    assert np.allclose(
        chromatic_adaptation_gains(white, 5.0),
        chromatic_adaptation_gains(white, 1.0),
    )
    assert np.allclose(chromatic_adaptation_gains(white, -2.0), 1.0)


def test_a_black_field_leaves_the_gains_alone() -> None:
    assert np.allclose(chromatic_adaptation_gains(np.zeros(3), 1.0), 1.0)


def test_adaptation_is_applied_per_pixel_over_an_image() -> None:
    image = np.random.default_rng(7).uniform(0.0, 1.0, size=(4, 5, 3))
    white = planckian_linear_srgb(3_500.0)
    adapted = chromatically_adapt(image, white, 0.7)
    assert adapted.shape == image.shape
    assert np.allclose(
        adapted[2, 3], chromatically_adapt(image[2, 3], white, 0.7)
    )


# --- the time course --------------------------------------------------------


def test_chromatic_adaptation_is_ninety_percent_complete_after_a_minute() -> None:
    # Fairchild and Reniff (1995). The constant is 60 / ln(10).
    assert 1.0 - math.exp(-60.0 / CHROMATIC_ADAPTATION_TIME_S) == pytest.approx(
        0.9, abs=1e-3
    )


def test_a_burst_is_not_discounted() -> None:
    # The slowness is the point. A two-second break moves the adapting white by
    # under a tenth, which is why an observer sees a green shell as green
    # instead of watching the visual system explain it away.
    response = 1.0 - math.exp(-2.0 / CHROMATIC_ADAPTATION_TIME_S)
    assert response < 0.10


def test_chromatic_adaptation_is_slower_than_light_adaptation() -> None:
    from simulator.human_vision import LIGHT_ADAPTATION_TIME_S

    assert CHROMATIC_ADAPTATION_TIME_S > 10.0 * LIGHT_ADAPTATION_TIME_S


# --- the observer state -----------------------------------------------------


def test_the_state_exposes_the_degree_to_the_shader() -> None:
    state = HumanVisionState()
    uniforms = state.uniforms()
    assert uniforms["chromatic_degree"] == state.chromatic_degree
    assert 0.0 < uniforms["chromatic_degree"] < 1.0
    assert "chromatic_degree" in state.summary()


def test_the_shader_declares_the_uniform_the_state_sends() -> None:
    source = shaders.source("human_vision.frag")
    for name in HumanVisionState().uniforms():
        assert re.search(rf"^uniform \w+ {name};", source, re.MULTILINE), name


def test_the_adapting_white_is_tracked_on_the_gpu_not_the_cpu() -> None:
    # The white is a per-frame image statistic; reading it back would stall the
    # frame for a value the shader is about to use anyway. The state carries
    # only the degree.
    assert not hasattr(HumanVisionState(), "adapting_white")
    adaptation = shaders.source("adaptation.frag")
    assert "chromatic_response" in adaptation
    assert "global_pooling_lod" in adaptation


def test_the_adaptation_buffer_stores_a_normalised_white() -> None:
    # Unit luminance is what keeps the von Kries step from changing brightness.
    adaptation = shaders.source("adaptation.frag")
    assert "field / field_luminance" in adaptation
    assert "frag_color = vec4(white, luminance);" in adaptation


def _texture_lod_levels(source: str) -> list[str]:
    """The level argument of every ``textureLod`` call in ``source``.

    Parenthesis-balanced rather than a regex, because the coordinate argument
    is itself a constructor call — ``textureLod(tex, vec2(0.5), lod)``.
    """

    levels = []
    for match in re.finditer(r"textureLod\s*\(", source):
        depth, start, arguments = 0, match.end(), []
        for index in range(match.end(), len(source)):
            character = source[index]
            if character == "(":
                depth += 1
            elif character == ")":
                if depth == 0:
                    arguments.append(source[start:index])
                    break
                depth -= 1
            elif character == "," and depth == 0:
                arguments.append(source[start:index])
                start = index + 1
        levels.append(arguments[-1].strip())
    return levels


def test_no_shader_relies_on_hardware_trilinear_filtering() -> None:
    # This driver implements GL_LINEAR_MIPMAP_LINEAR as *brilinear*: the level
    # weight is clamp((frac - 1/6) / (2/3), 0, 1), so a requested 0.75 is
    # applied as 0.875. Measured at five points by V-24. Any sample at a
    # fractional mip level is therefore driver-dependent, which a
    # reconstruction cannot have — the interpolation belongs in the shader.
    for name in shaders.available():
        for level in _texture_lod_levels(shaders.source(name)):
            # An integer literal, or a variable the caller floors, is exact:
            # at a whole level the hardware blend weight is zero.
            assert (
                re.fullmatch(r"\d+\.0", level)
                or level in ("level", "level + 1.0")
                or level.endswith("_lod")
            ), f"{name} samples a fractional mip level: {level!r}"


def test_peripheral_blur_interpolates_two_explicit_levels() -> None:
    source = shaders.source("human_vision.frag")
    assert "float level = floor(lod);" in source
    assert "textureLod(hdr_texture, uv, level)" in source
    assert "textureLod(hdr_texture, uv, level + 1.0)" in source
    assert "lod - level" in source


def test_the_rod_channel_keeps_the_unadapted_luminance() -> None:
    # Rods have one photopigment and no gain control that could discount a hue
    # they cannot see, so chromatic adaptation must not reach the rod path.
    source = shaders.source("human_vision.frag")
    assert "float luminance = dot(normalized, PHOTOPIC_WEIGHTS);" in source
    assert "mix(vec3(luminance), adapted, cone_fraction)" in source
