from __future__ import annotations

import pytest

from tools.probe_facade_draw_threshold import (
    FULL_BUILDING_VERTICES,
    ORIGINAL_BUILDING_VERTICES,
    TARGET_END,
    TARGET_FIRST,
    VERTEX_BYTES,
    _select_specs,
    draw_specs,
)


def test_threshold_draws_are_aligned_and_cover_the_target() -> None:
    specs = draw_specs()

    assert specs[0].name == "compact_quad"
    assert specs[1].name == "full_vbo_target"
    assert "original_buildings" in {spec.name for spec in specs}
    assert specs[-1].name == "full_with_rooftops"
    original = next(spec for spec in specs if spec.name == "original_buildings")
    assert original.vertices == ORIGINAL_BUILDING_VERTICES
    assert specs[-1].vertices == FULL_BUILDING_VERTICES
    for spec in specs[1:]:
        assert spec.first % 3 == 0
        assert spec.vertices % 3 == 0
        assert spec.first <= TARGET_FIRST
        assert spec.end >= TARGET_END


def test_threshold_preceding_ranges_grow_without_moving_target_offset() -> None:
    specs = draw_specs()
    preceding = [
        spec
        for spec in specs
        if spec.name.startswith("preceding_")
    ]

    assert [spec.vertices - 6 for spec in preceding] == sorted(
        spec.vertices - 6 for spec in preceding
    )
    assert all(spec.end == TARGET_END for spec in preceding)
    assert specs[1].first == TARGET_FIRST
    assert specs[1].vertices == 6


def test_rooftop_threshold_ranges_grow_from_original_boundary() -> None:
    rooftop = [
        spec for spec in draw_specs() if spec.name.startswith("rooftop_extra_")
    ]

    extras = [spec.vertices - ORIGINAL_BUILDING_VERTICES for spec in rooftop]
    assert extras == sorted(extras)
    assert extras[0] == 3
    assert extras[-1] < FULL_BUILDING_VERTICES - ORIGINAL_BUILDING_VERTICES
    assert extras[-1] == 11_577


def test_rooftop_focus_excludes_offset_and_preceding_ranges() -> None:
    focused = _select_specs(draw_specs(), True)

    assert focused[0].name == "original_buildings"
    assert focused[-1].name == "full_with_rooftops"
    assert all(
        spec.name == "original_buildings"
        or spec.name.startswith("rooftop_extra_")
        or spec.name == "full_with_rooftops"
        for spec in focused
    )


def test_final_boundary_focus_brackets_115200_total_vertices() -> None:
    focused = _select_specs(draw_specs(), False, True)
    totals = [spec.vertices for spec in focused]

    assert ORIGINAL_BUILDING_VERTICES + 11_217 in totals
    assert 115_197 in totals
    assert ORIGINAL_BUILDING_VERTICES + 11_331 in totals
    bracket = [total for total in totals if 115_173 <= total <= 115_197]
    assert all(b - a == 3 for a, b in zip(bracket, bracket[1:]))


def test_vertex_stride_matches_scene_layout() -> None:
    assert VERTEX_BYTES == 40


@pytest.mark.parametrize(
    "target,original,full",
    [
        (1, ORIGINAL_BUILDING_VERTICES, FULL_BUILDING_VERTICES),
        (TARGET_FIRST, ORIGINAL_BUILDING_VERTICES - 1, FULL_BUILDING_VERTICES),
        (TARGET_FIRST, FULL_BUILDING_VERTICES, ORIGINAL_BUILDING_VERTICES),
    ],
)
def test_threshold_draw_boundaries_fail_closed(
    target: int, original: int, full: int
) -> None:
    with pytest.raises(ValueError):
        draw_specs(target, original, full)
