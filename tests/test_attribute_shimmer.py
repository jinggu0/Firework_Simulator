from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simulator import shaders
from tools.attribute_shimmer import (
    OVERRIDE_FIELDS,
    _load_diagnostics,
    carried_share,
    display_mean_map,
    mean_display_code,
    parse_override,
)
from tools.measure_temporal_shimmer import box_downsample, render_pan
from tools.shader_candidate import apply_replacements, load_candidate


DIAGNOSTICS = Path("docs/validation/shimmer_ab_v2/diagnostics")
WINDOW_TERM_DIAGNOSTICS = [
    "diagnostic_window_blinds_at_mean",
    "diagnostic_window_curtains_at_mean",
    "diagnostic_window_pane_at_mean",
    "diagnostic_window_emission_off_both_paths",
]


def test_a_float_override_is_read() -> None:
    assert parse_override("bloom_strength=0.0") == ("bloom_strength", 0.0)


def test_a_boolean_override_is_read() -> None:
    field, value = parse_override("simulate_fixed_gaze_peripheral_acuity=true")

    assert (field, value) == ("simulate_fixed_gaze_peripheral_acuity", True)


def test_an_unlisted_field_is_refused() -> None:
    # Resizing the frame or moving the camera would make the diagnostic
    # incomparable with the shipped run it is subtracted from.
    with pytest.raises(ValueError, match="must be one of"):
        parse_override("width=640")


def test_a_field_without_a_value_is_refused() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        parse_override("bloom_strength")


def test_a_value_that_does_not_parse_is_refused() -> None:
    with pytest.raises(ValueError, match="bad value"):
        parse_override("bloom_strength=dim")


def test_every_override_field_exists_on_the_render_config() -> None:
    from simulator.config import RenderConfig

    fields = {field.name for field in RenderConfig.__dataclass_fields__.values()}

    assert set(OVERRIDE_FIELDS) <= fields


def test_an_unknown_render_override_fails_before_any_rendering() -> None:
    # `render_pan` builds the config before it opens a window, so a typo in a
    # diagnostic stops instead of quietly measuring the shipped renderer twice.
    with pytest.raises(TypeError):
        render_pan(None, "human_vision", 24, 0.5, render_overrides={"nope": 1.0})


def test_a_term_that_carries_half_the_flicker_reads_as_half() -> None:
    assert carried_share(2.0, 1.0) == pytest.approx(0.5)


def test_a_term_whose_removal_makes_things_worse_reads_negative() -> None:
    assert carried_share(1.0, 1.5) == pytest.approx(-0.5)


def test_a_view_with_no_flicker_carries_nothing() -> None:
    assert carried_share(0.0, 0.0) == 0.0


def test_the_mean_display_code_uses_only_the_masked_pixels() -> None:
    values = np.array([[10.0, 200.0], [30.0, 50.0]])
    mask = np.array([[True, False], [True, False]])

    assert mean_display_code(values, mask) == pytest.approx(20.0)


def test_an_empty_mask_has_no_mean() -> None:
    assert mean_display_code(np.ones((2, 2)), np.zeros((2, 2), dtype=bool)) == 0.0


def test_the_display_mean_leaves_out_the_first_frame() -> None:
    # Frame zero is only ever a prediction source, never a predicted frame, so
    # it is not one of the pixels the shimmer score is built from.
    frames = np.stack([np.full((2, 2), 100.0), np.full((2, 2), 4.0), np.full((2, 2), 6.0)])

    assert display_mean_map({"display_luma": frames}) == pytest.approx(np.full((2, 2), 5.0))


@pytest.mark.parametrize("name", WINDOW_TERM_DIAGNOSTICS)
def test_each_window_term_diagnostic_applies_to_the_shipped_shader(name: str) -> None:
    candidate = load_candidate(DIAGNOSTICS / f"{name}.json")
    source = shaders.source("scene.frag")

    for replacement in candidate.replacements:
        assert source.count(replacement.old) == 1, replacement.old[:40]
    assert apply_replacements(source, candidate.replacements) != source


@pytest.mark.parametrize("name", WINDOW_TERM_DIAGNOSTICS)
def test_each_window_term_diagnostic_changes_both_emission_paths(name: str) -> None:
    # Since V2-3g the facade emits window light twice: one sample inline and a
    # footprint average of `window_emission_at`. A diagnostic that changed only
    # one would leave the other carrying the term it claims to have removed.
    candidate = load_candidate(DIAGNOSTICS / f"{name}.json")
    source = shaders.source("scene.frag")
    patched = apply_replacements(source, candidate.replacements)
    function = patched[patched.index("vec3 window_emission_at("):patched.index("// Derivative cotangent frame")]
    inline = patched[patched.index("void main()"):]

    assert function != source[source.index("vec3 window_emission_at("):source.index("// Derivative cotangent frame")]
    assert inline != source[source.index("void main()"):]


def test_the_means_the_diagnostics_use_are_the_area_means() -> None:
    # A diagnostic that dims a region lowers its code-value flicker through the
    # ACES slope alone, so each flattened term keeps its own area mean.
    grid = (np.arange(200_000) + 0.5) / 200_000

    blinds = np.clip((grid - 0.16) / (0.31 - 0.16), 0.0, 1.0)
    blinds = blinds * blinds * (3.0 - 2.0 * blinds)
    curtains_at = np.abs(grid - 0.5)
    curtains = np.clip((curtains_at - 0.16) / (0.28 - 0.16), 0.0, 1.0)
    curtains = curtains * curtains * (3.0 - 2.0 * curtains)

    blinds_json = (DIAGNOSTICS / "diagnostic_window_blinds_at_mean.json").read_text(encoding="utf-8")
    curtains_json = (DIAGNOSTICS / "diagnostic_window_curtains_at_mean.json").read_text(encoding="utf-8")

    assert blinds.mean() == pytest.approx(0.765, abs=1e-4)
    assert curtains.mean() == pytest.approx(0.56, abs=1e-4)
    assert "mix(.52, 1.0, .765)" in json.loads(blinds_json)["replacements"][0]["new"]
    assert "mix(.58, 1.0, .56)" in json.loads(curtains_json)["replacements"][0]["new"]


def test_a_supersampled_frame_is_averaged_block_by_block() -> None:
    frame = np.array([[1.0, 3.0, 10.0, 10.0], [5.0, 7.0, 10.0, 10.0]])

    assert box_downsample(frame, 2) == pytest.approx(np.array([[4.0, 10.0]]))


def test_a_frame_is_left_alone_at_factor_one() -> None:
    frame = np.arange(6.0).reshape(2, 3)

    assert box_downsample(frame, 1) is frame


def test_a_frame_that_does_not_divide_is_refused() -> None:
    with pytest.raises(ValueError, match="divide"):
        box_downsample(np.zeros((3, 4)), 2)


def test_a_supersample_below_one_fails_before_any_rendering() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        render_pan(None, "human_vision", 24, 0.5, supersample=0)


def test_a_supersample_diagnostic_below_two_is_refused() -> None:
    # One is the shipped render, which is the baseline every diagnostic is
    # measured against, so asking for it as a diagnostic is a mistake.
    with pytest.raises(ValueError, match="at least 2"):
        _load_diagnostics([], [], [1])


def test_a_supersample_diagnostic_says_its_display_frame_is_not_comparable() -> None:
    # Bloom and the glare tail are measured in rendered pixels, so a finer grid
    # narrows them on the scene and the display frame is a different transform.
    diagnostics = _load_diagnostics([], [], [2])

    assert diagnostics[0]["supersample"] == 2
    assert diagnostics[0]["diagnostic_id"] == "supersample:2x"
    assert "linear HDR only" in diagnostics[0]["hypothesis"]


def test_a_render_override_diagnostic_is_not_supersampled() -> None:
    diagnostics = _load_diagnostics([], ["bloom_strength=0.0"], [])

    assert diagnostics[0]["supersample"] == 1
