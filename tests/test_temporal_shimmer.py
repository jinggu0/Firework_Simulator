from __future__ import annotations

import math

import numpy as np
import pytest

from simulator.camera import FreeCamera
from simulator.renderer import FAR_PLANE_M, NEAR_PLANE_M, _look_at, _perspective
from simulator.validation.temporal_shimmer import (
    MINIMUM_FRAME_PAIRS,
    pan_source_coordinates,
    shimmer_statistics,
)


HEIGHT, WIDTH, FRAMES = 48, 128, 24
#: The golden-ratio step keeps every pair of frames a non-integer shift apart,
#: so no aliased component can hide by landing back on the pixel grid.
SPEED_PX = (math.sqrt(5.0) - 1.0) / 2.0


def _translation_coordinates(speed_px: float, frames: int = FRAMES) -> np.ndarray:
    """Content drifting left: pixel (r, c) of frame k+1 was at (r, c + speed)."""

    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    single = np.stack([rows, cols + speed_px])
    return np.repeat(single[None], frames - 1, axis=0)


def _point_sampled(pattern, speed_px: float = SPEED_PX) -> np.ndarray:
    cols = np.arange(WIDTH, dtype=np.float64) + 0.5
    frames = [
        np.tile(pattern(cols + speed_px * k), (HEIGHT, 1)) for k in range(FRAMES)
    ]
    return np.stack(frames)


def _box_sampled(pattern, samples: int = 64, speed_px: float = SPEED_PX) -> np.ndarray:
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples
    cols = np.arange(WIDTH, dtype=np.float64)[:, None] + offsets[None, :]
    frames = [
        np.tile(pattern(cols + speed_px * k).mean(axis=1), (HEIGHT, 1))
        for k in range(FRAMES)
    ]
    return np.stack(frames)


def _stripes(frequency_cycles_per_px: float, contrast: float = 0.25):
    return lambda x: 0.5 + contrast * np.cos(2.0 * np.pi * frequency_cycles_per_px * x)


def _smooth_noise(seed: int, maximum_frequency: float = 0.2, terms: int = 12):
    generator = np.random.default_rng(seed)
    frequencies = generator.uniform(0.02, maximum_frequency, (terms, 2))
    phases = generator.uniform(0.0, 2.0 * np.pi, terms)

    def field(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        total = np.zeros(np.broadcast(x, y).shape)
        for (fx, fy), phase in zip(frequencies, phases):
            total += np.cos(2.0 * np.pi * (fx * x + fy * y) + phase)
        return total / np.sqrt(terms / 2.0)

    return field


def _relative(frames: np.ndarray, coordinates: np.ndarray, **kwargs) -> dict:
    return shimmer_statistics(
        frames, coordinates, threshold=0.05, relative=True, **kwargs
    )


def test_a_band_limited_pattern_moving_rigidly_is_stable() -> None:
    frames = _point_sampled(_stripes(0.08))

    stats = _relative(frames, _translation_coordinates(SPEED_PX))

    assert stats["unstable_pixel_fraction"] == 0.0
    assert stats["mean_abs_residual"] < 1e-3


def test_detail_finer_than_the_pixel_grid_shimmers_under_motion() -> None:
    frames = _point_sampled(_stripes(0.8))

    stats = _relative(frames, _translation_coordinates(SPEED_PX))

    assert stats["unstable_pixel_fraction"] > 0.9


def test_supersampling_the_same_detail_reduces_the_shimmer() -> None:
    coordinates = _translation_coordinates(SPEED_PX)

    point = _relative(_point_sampled(_stripes(0.8)), coordinates)
    box = _relative(_box_sampled(_stripes(0.8)), coordinates)

    assert box["mean_abs_residual"] < 0.5 * point["mean_abs_residual"]


def test_fading_unresolvable_detail_to_its_mean_beats_supersampling() -> None:
    # The property the supersampled-reference metric lacked: converging detail
    # the grid cannot carry to its expected value must read as more stable,
    # not as a departure from a more detailed reference.
    coordinates = _translation_coordinates(SPEED_PX)

    faded = _relative(np.full((FRAMES, HEIGHT, WIDTH), 0.5), coordinates)
    box = _relative(_box_sampled(_stripes(0.8)), coordinates)

    assert faded["mean_abs_residual"] < 1e-12
    assert faded["unstable_pixel_fraction"] == 0.0
    assert faded["mean_abs_residual"] < box["mean_abs_residual"]


def test_noise_fixed_to_the_screen_is_unstable_while_the_scene_moves() -> None:
    noise = _smooth_noise(seed=7)
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64) + 0.5
    screen_noise = 1.0 + 0.3 * noise(cols, rows)
    base = _point_sampled(_stripes(0.03, contrast=0.2))

    stats = _relative(base * screen_noise[None], _translation_coordinates(SPEED_PX))

    assert stats["unstable_pixel_fraction"] > 0.5


def test_the_same_noise_attached_to_the_surface_is_stable() -> None:
    noise = _smooth_noise(seed=7)
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64) + 0.5
    base = _point_sampled(_stripes(0.03, contrast=0.2))
    frames = np.stack(
        [
            base[k] * (1.0 + 0.3 * noise(cols + SPEED_PX * k, rows))
            for k in range(FRAMES)
        ]
    )

    stats = _relative(frames, _translation_coordinates(SPEED_PX))

    assert stats["unstable_pixel_fraction"] < 0.01


def test_a_locked_off_camera_with_identical_frames_scores_zero() -> None:
    frames = np.repeat(_point_sampled(_stripes(0.8))[:1], FRAMES, axis=0)

    stats = _relative(frames, _translation_coordinates(0.0))

    # Spline interpolation reproduces its samples to round-off, not bit-exactly.
    assert stats["mean_abs_residual"] < 1e-12
    assert stats["unstable_pixel_fraction"] == 0.0
    # The remainder is the border band the interpolation cannot be trusted in.
    assert stats["valid_pixel_fraction"] > 0.5


def test_pixels_whose_source_left_the_previous_frame_are_not_scored() -> None:
    frames = _point_sampled(_stripes(0.08))

    stats = _relative(frames, _translation_coordinates(SPEED_PX))

    # The rightmost columns come from beyond the previous frame's edge.
    assert 0.0 < stats["valid_pixel_fraction"] < 1.0
    assert np.isfinite(stats["p95_abs_residual"])


def test_fewer_than_the_minimum_frame_pairs_is_refused() -> None:
    frames = _point_sampled(_stripes(0.08))[: MINIMUM_FRAME_PAIRS]

    with pytest.raises(ValueError, match=str(MINIMUM_FRAME_PAIRS)):
        _relative(
            frames, _translation_coordinates(SPEED_PX, frames=MINIMUM_FRAME_PAIRS)
        )


def test_an_absolute_threshold_counts_code_values() -> None:
    steady = np.full((FRAMES, HEIGHT, WIDTH), 100.0)
    small, large = steady.copy(), steady.copy()
    small[1::2] += 1.0
    large[1::2] += 3.0
    coordinates = _translation_coordinates(0.0)

    quiet = shimmer_statistics(small, coordinates, threshold=2.0, relative=False)
    loud = shimmer_statistics(large, coordinates, threshold=2.0, relative=False)

    assert quiet["unstable_pixel_fraction"] == 0.0
    assert loud["unstable_pixel_fraction"] > 0.9


def test_excluded_pixels_do_not_count() -> None:
    frames = _point_sampled(_stripes(0.08))
    frames[:, :32, :64] = _point_sampled(_stripes(0.8))[:, :32, :64]
    coordinates = _translation_coordinates(SPEED_PX)
    # Interpolation carries the patch edge a few pixels past the patch, so the
    # exclusion covers the spline's reach as any real exclusion would have to.
    exclude = np.zeros((HEIGHT, WIDTH), dtype=bool)
    exclude[:32, :72] = True

    shimmering = _relative(frames, coordinates)
    excluded = _relative(frames, coordinates, exclude=exclude)

    assert shimmering["unstable_pixel_fraction"] > 0.2
    assert excluded["unstable_pixel_fraction"] == 0.0


def test_the_worst_block_is_where_the_shimmer_is() -> None:
    frames = _point_sampled(_stripes(0.08))
    frames[:, 16:32, 64:80] = _point_sampled(_stripes(0.8))[:, 16:32, 64:80]

    stats = _relative(frames, _translation_coordinates(SPEED_PX))

    worst = stats["worst_blocks"][0]
    assert (worst["row"], worst["col"]) == (16, 64)


def _view_projection(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    camera = FreeCamera()
    camera.yaw_deg, camera.pitch_deg = yaw_deg, pitch_deg
    projection = _perspective(45.0, 16.0 / 9.0, NEAR_PLANE_M, FAR_PLANE_M)
    view = _look_at(camera.position_m, camera.position_m + camera.forward)
    return projection.astype(np.float64) @ view.astype(np.float64)


def _focal_px(height: int) -> float:
    return (height / 2.0) / math.tan(math.radians(45.0) / 2.0)


def test_identical_cameras_map_every_pixel_to_itself() -> None:
    matrix = _view_projection(30.0, 5.0)

    coordinates = pan_source_coordinates(matrix, matrix, 160, 90)

    rows, cols = np.mgrid[0:90, 0:160]
    assert np.allclose(coordinates[0], rows, atol=1e-6)
    assert np.allclose(coordinates[1], cols, atol=1e-6)


def test_turning_right_moves_content_left() -> None:
    width, height, step_deg = 160, 90, 0.2
    before = _view_projection(30.0, 0.0)
    after = _view_projection(30.0 + step_deg, 0.0)

    coordinates = pan_source_coordinates(before, after, width, height)

    centre_row, centre_col = height // 2, width // 2
    expected_shift = _focal_px(height) * math.tan(math.radians(step_deg))
    # The pixel centre sits half a pixel off the optical axis, which is why the
    # comparison is loose at the sub-millipixel level rather than exact.
    assert coordinates[1, centre_row, centre_col] - centre_col == pytest.approx(
        expected_shift, abs=2e-3
    )
    assert coordinates[0, centre_row, centre_col] == pytest.approx(centre_row, abs=2e-3)


def test_tilting_up_moves_content_down_the_image_rows() -> None:
    # Row 0 is the top of the image. Tilting up brings sky in from the top, so
    # what a lower row shows now was a row nearer the top one frame earlier.
    width, height, step_deg = 160, 90, 0.2
    before = _view_projection(30.0, 0.0)
    after = _view_projection(30.0, step_deg)

    coordinates = pan_source_coordinates(before, after, width, height)

    centre_row, centre_col = height // 2, width // 2
    expected_shift = _focal_px(height) * math.tan(math.radians(step_deg))
    assert centre_row - coordinates[0, centre_row, centre_col] == pytest.approx(
        expected_shift, abs=2e-3
    )
