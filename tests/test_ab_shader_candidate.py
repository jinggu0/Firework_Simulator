from __future__ import annotations

import numpy as np
import pytest

from tools.ab_shader_candidate import (
    REPOSITORY_ROOT,
    candidate_verdict,
    changed_pixel_mask,
    portable_path,
    view_verdict,
)


def _run(display: float, hdr: float, guard4: float, guard16: float) -> dict:
    return {
        "display_unstable": display,
        "hdr_unstable": hdr,
        "guard": {"4": guard4, "16": guard16},
    }


def test_a_more_stable_candidate_that_keeps_the_guard_is_accepted() -> None:
    shipped = _run(0.20, 0.30, 0.10, 0.05)
    repeat = _run(0.2001, 0.3001, 0.1001, 0.0501)
    candidate = _run(0.15, 0.25, 0.08, 0.04)

    verdict = candidate_verdict({"view": view_verdict(shipped, repeat, candidate)})

    assert verdict["accepted"] is True
    assert verdict["reasons"] == []


def test_moving_away_from_the_reference_rejects_even_a_more_stable_candidate() -> None:
    shipped = _run(0.20, 0.30, 0.10, 0.05)
    repeat = _run(0.20, 0.30, 0.10, 0.05)
    candidate = _run(0.10, 0.20, 0.13, 0.05)

    verdict = candidate_verdict({"facade": view_verdict(shipped, repeat, candidate)})

    assert verdict["accepted"] is False
    assert any("facade" in reason and "4 px" in reason for reason in verdict["reasons"])


def test_an_improvement_inside_the_noise_is_not_an_improvement() -> None:
    shipped = _run(0.200, 0.30, 0.10, 0.05)
    repeat = _run(0.210, 0.30, 0.10, 0.05)
    candidate = _run(0.195, 0.30, 0.10, 0.05)

    single = view_verdict(shipped, repeat, candidate)
    verdict = candidate_verdict({"view": single})

    assert single["more_stable"] is False
    assert verdict["accepted"] is False
    assert any("no view" in reason for reason in verdict["reasons"])


def test_becoming_less_stable_in_any_view_rejects_the_candidate() -> None:
    better = view_verdict(_run(0.2, 0.3, 0.1, 0.05), _run(0.2, 0.3, 0.1, 0.05), _run(0.1, 0.2, 0.1, 0.05))
    worse = view_verdict(_run(0.2, 0.3, 0.1, 0.05), _run(0.2, 0.3, 0.1, 0.05), _run(0.25, 0.3, 0.1, 0.05))

    verdict = candidate_verdict({"better": better, "worse": worse})

    assert verdict["accepted"] is False
    assert any("worse" in reason and "less stable" in reason for reason in verdict["reasons"])


def test_linear_hdr_instability_counts_as_well_as_the_display() -> None:
    shipped = _run(0.20, 0.30, 0.10, 0.05)
    candidate = _run(0.10, 0.40, 0.10, 0.05)

    single = view_verdict(shipped, shipped, candidate)

    assert single["not_less_stable"] is False


def test_the_noise_comes_from_the_repeated_shipped_run() -> None:
    single = view_verdict(_run(0.20, 0.30, 0.10, 0.05), _run(0.23, 0.30, 0.10, 0.05), _run(0.18, 0.30, 0.10, 0.05))

    assert single["noise"]["display_unstable"] == pytest.approx(0.03)
    assert single["more_stable"] is False


def test_changed_pixels_are_exact_differences() -> None:
    shipped = np.zeros((4, 4))
    candidate = shipped.copy()
    candidate[1, 2] = 1e-12

    mask = changed_pixel_mask(shipped, candidate)

    assert mask.sum() == 1 and mask[1, 2]


def test_changed_pixels_need_frames_of_one_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        changed_pixel_mask(np.zeros((4, 4)), np.zeros((4, 5)))


def test_a_candidate_inside_the_repository_is_recorded_relative() -> None:
    inside = REPOSITORY_ROOT / "docs" / "validation" / "x.json"

    assert portable_path(inside) == "docs/validation/x.json"


def test_a_candidate_outside_the_repository_is_recorded_absolute(tmp_path) -> None:
    outside = tmp_path / "diagnostic.json"

    assert portable_path(outside) == outside.resolve().as_posix()
