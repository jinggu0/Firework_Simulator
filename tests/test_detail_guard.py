from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from simulator.validation.detail_guard import (
    GUARD_SCALES_PX,
    block_distance_to_reference,
    detail_guard,
)


HEIGHT, WIDTH = 64, 128
SUPERSAMPLE = 8


def _scene_sampled(field, samples: int) -> np.ndarray:
    """Average a continuous field over each pixel with `samples` per axis."""

    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    rows = np.arange(HEIGHT, dtype=np.float64)[:, None] + offsets[None, :]
    cols = np.arange(WIDTH, dtype=np.float64)[:, None] + offsets[None, :]
    values = field(rows[:, :, None, None], cols[None, None, :, :])
    return values.mean(axis=(1, 3))


def _windows(x_cells_per_px: float = 2.3, seed: int = 3):
    """Lit or dark cells finer than the pixel grid, over a dim wall."""

    generator = np.random.default_rng(seed)
    lit = generator.random((1024, 1024)) > 0.6

    def field(row, col):
        cell_row = np.floor(row * x_cells_per_px).astype(np.int64) % 1024
        cell_col = np.floor(col * x_cells_per_px).astype(np.int64) % 1024
        return 0.05 + 0.9 * lit[cell_row, cell_col]

    return field, 0.05 + 0.9 * 0.4


def test_the_guard_scales_start_above_the_pixel() -> None:
    # A single pixel is where a supersampled reference punishes a correct fade.
    assert min(GUARD_SCALES_PX) > 1


def test_an_unchanged_candidate_matches_the_baseline_exactly() -> None:
    field, _ = _windows()
    baseline = _scene_sampled(field, 1)
    reference = _scene_sampled(field, SUPERSAMPLE)

    guard = detail_guard(baseline, baseline.copy(), reference)

    for scale in GUARD_SCALES_PX:
        entry = guard["scales"][str(scale)]
        assert entry["candidate"] == entry["baseline"]


def test_fading_unresolvable_cells_to_their_mean_moves_toward_the_reference() -> None:
    field, expectation = _windows()
    baseline = _scene_sampled(field, 1)
    reference = _scene_sampled(field, SUPERSAMPLE)
    faded = np.full_like(baseline, expectation)

    guard = detail_guard(baseline, faded, reference)

    for scale in GUARD_SCALES_PX:
        entry = guard["scales"][str(scale)]
        assert entry["candidate"] < entry["baseline"]


def test_a_fade_to_the_wrong_mean_moves_away_from_the_reference() -> None:
    field, expectation = _windows()
    baseline = _scene_sampled(field, 1)
    reference = _scene_sampled(field, SUPERSAMPLE)
    biased = np.full_like(baseline, expectation * 0.8)

    guard = detail_guard(baseline, biased, reference)

    coarsest = guard["scales"][str(max(GUARD_SCALES_PX))]
    assert coarsest["candidate"] > coarsest["baseline"]


def test_blurring_structure_the_grid_resolves_moves_away_from_the_reference() -> None:
    generator = np.random.default_rng(11)
    coarse = np.kron(generator.random((HEIGHT // 8, WIDTH // 8)), np.ones((8, 8)))
    baseline = coarse
    reference = coarse
    blurred = gaussian_filter(coarse, sigma=3.0, mode="nearest")

    guard = detail_guard(baseline, blurred, reference)

    finest = guard["scales"][str(min(GUARD_SCALES_PX))]
    assert finest["baseline"] == 0.0
    assert finest["candidate"] > 0.05


def test_distance_is_a_share_of_the_reference_light() -> None:
    reference = np.full((HEIGHT, WIDTH), 2.0)
    frame = np.full((HEIGHT, WIDTH), 2.5)

    assert block_distance_to_reference(frame, reference, 4) == pytest.approx(0.25)


def test_frames_of_different_shape_are_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        block_distance_to_reference(
            np.zeros((HEIGHT, WIDTH)), np.zeros((HEIGHT, WIDTH + 4)), 4
        )


def test_a_scale_that_does_not_divide_the_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="divide"):
        block_distance_to_reference(
            np.zeros((HEIGHT, WIDTH)), np.ones((HEIGHT, WIDTH)), 5
        )
