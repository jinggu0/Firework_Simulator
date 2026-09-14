from __future__ import annotations

import math

import numpy as np
import pytest

from simulator.renderer import FAR_PLANE_M, NEAR_PLANE_M, _perspective
from simulator.validation.temporal_shimmer import pan_source_coordinates
from tools.measure_temporal_shimmer import (
    PIXELS_PER_FRAME,
    camera_view_projection,
    residual_exclusion_mask,
    yaw_step_for_centre_motion,
)


WIDTH, HEIGHT = 320, 180
PROJECTION = _perspective(45.0, WIDTH / HEIGHT, NEAR_PLANE_M, FAR_PLANE_M)
POSITION = np.array([10.0, 20.0, 30.0], dtype=np.float32)


def _centre_motion_px(yaw_deg: float, pitch_deg: float, step_deg: float) -> float:
    before = camera_view_projection(PROJECTION, POSITION, yaw_deg, pitch_deg)
    after = camera_view_projection(
        PROJECTION, POSITION, yaw_deg + step_deg, pitch_deg
    )
    coordinates = pan_source_coordinates(before, after, WIDTH, HEIGHT)
    row, col = HEIGHT // 2, WIDTH // 2
    return float(
        np.hypot(coordinates[0, row, col] - row, coordinates[1, row, col] - col)
    )


def test_the_default_speed_is_the_golden_ratio_step() -> None:
    assert PIXELS_PER_FRAME == pytest.approx((math.sqrt(5.0) - 1.0) / 2.0)


@pytest.mark.parametrize("pitch_deg", [0.0, 2.6, -40.0])
def test_the_yaw_step_moves_the_image_centre_by_the_requested_pixels(
    pitch_deg: float,
) -> None:
    step_deg = yaw_step_for_centre_motion(
        PROJECTION, POSITION, 23.0, pitch_deg, WIDTH, HEIGHT, PIXELS_PER_FRAME
    )

    assert _centre_motion_px(23.0, pitch_deg, step_deg) == pytest.approx(
        PIXELS_PER_FRAME, abs=1e-3
    )


def test_a_pitched_camera_needs_a_larger_yaw_step_for_the_same_motion() -> None:
    # Turning about the world vertical sweeps a tilted view across less of the
    # image, so holding the on-screen speed takes more yaw.
    level = yaw_step_for_centre_motion(
        PROJECTION, POSITION, 23.0, 0.0, WIDTH, HEIGHT, PIXELS_PER_FRAME
    )
    tilted = yaw_step_for_centre_motion(
        PROJECTION, POSITION, 23.0, -40.0, WIDTH, HEIGHT, PIXELS_PER_FRAME
    )

    assert tilted > level * 1.2


def test_residual_regions_grow_by_the_path_motion() -> None:
    residuals = [{"region_rows": [10, 12], "region_cols": [20, 25]}]

    mask = residual_exclusion_mask((40, 60), residuals, dilation_px=3)

    rows, cols = np.nonzero(mask)
    assert (rows.min(), rows.max()) == (7, 15)
    assert (cols.min(), cols.max()) == (17, 28)


def test_residual_regions_are_clipped_at_the_frame_edge() -> None:
    residuals = [{"region_rows": [0, 1], "region_cols": [57, 59]}]

    mask = residual_exclusion_mask((40, 60), residuals, dilation_px=5)

    rows, cols = np.nonzero(mask)
    assert (rows.min(), rows.max()) == (0, 6)
    assert (cols.min(), cols.max()) == (52, 59)


def test_no_residuals_exclude_nothing() -> None:
    assert not residual_exclusion_mask((40, 60), [], dilation_px=5).any()
