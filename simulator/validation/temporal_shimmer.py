"""Temporal shimmer under a known camera pan.

Shimmer is the image failing to move with the scene. When the camera turns
about its own centre, where every surface point goes on screen is fixed by the
two camera matrices alone, independent of depth, so the next frame is predicted
exactly by carrying the previous frame along that motion. Whatever the
prediction misses did not move with the scene: detail finer than the pixel grid
crawling at the wrong speed, edges stepping instead of sliding, or a noise
pattern that stays fixed to the screen while the surfaces pass beneath it.

This is why the measure can judge what a supersampled reference cannot. Fading
detail the grid cannot carry to its expected value leaves a frame that moves
cleanly, and it scores better; against a more detailed reference the same fade
reads as an error.

The prediction interpolates with a cubic spline, which is accurate for detail
the grid can carry. Detail it cannot carry has no correct prediction, and that
residual is the shimmer being measured. None of these statistics is a pass or
fail threshold on its own: a change that blurs the whole frame also moves
cleanly, so a lower score has to be read beside what the change removed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


#: Every frame pair is one sample of the motion. Below this many, a fraction
#: is too coarse to separate a real change from the scene happening to line up.
MINIMUM_FRAME_PAIRS = 20
#: Cubic B-spline interpolation.
INTERPOLATION_ORDER = 3
#: The spline prefilter reaches across the frame edge with weight |z|^n at n
#: pixels, where z = sqrt(3) - 2 is its pole. Seven pixels bring the edge's
#: influence on an interpolated value below 1e-4 of whatever lies past it.
BORDER_MARGIN_PX = math.ceil(math.log(1e-4) / math.log(2.0 - math.sqrt(3.0)))
#: Blocks are reported so the worst offenders can be found on screen.
BLOCK = 16


def pan_source_coordinates(
    view_projection_from: np.ndarray,
    view_projection_to: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Where each pixel of the later frame was in the earlier frame.

    Returns ``(2, height, width)`` fractional (row, column) indices in image
    order, row 0 at the top, pixel centres at whole indices. Exact only when
    both cameras share a centre, which is what makes it independent of depth:
    the far-plane point behind each pixel stands in for the whole ray.
    """

    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    ndc_x = (cols + 0.5) / width * 2.0 - 1.0
    ndc_y = 1.0 - (rows + 0.5) / height * 2.0
    clip = np.stack([ndc_x, ndc_y, np.ones_like(ndc_x), np.ones_like(ndc_x)])
    inverse_to = np.linalg.inv(np.asarray(view_projection_to, dtype=np.float64))
    world = np.einsum("ij,jhw->ihw", inverse_to, clip)
    world /= world[3]
    earlier = np.einsum(
        "ij,jhw->ihw", np.asarray(view_projection_from, dtype=np.float64), world
    )
    earlier_x = earlier[0] / earlier[3]
    earlier_y = earlier[1] / earlier[3]
    return np.stack(
        [
            (1.0 - earlier_y) * 0.5 * height - 0.5,
            (earlier_x + 1.0) * 0.5 * width - 0.5,
        ]
    )


def _inside(coordinates: np.ndarray, height: int, width: int) -> np.ndarray:
    margin = BORDER_MARGIN_PX
    return (
        (coordinates[0] >= margin)
        & (coordinates[0] <= height - 1 - margin)
        & (coordinates[1] >= margin)
        & (coordinates[1] <= width - 1 - margin)
    )


def _worst_blocks(score: np.ndarray, count: int = 5) -> list[dict[str, Any]]:
    height, width = score.shape
    rows, cols = height // BLOCK, width // BLOCK
    if not rows or not cols:
        return []
    blocks = score[: rows * BLOCK, : cols * BLOCK].reshape(
        rows, BLOCK, cols, BLOCK
    ).mean(axis=(1, 3))
    located = []
    for flat in np.argsort(blocks, axis=None)[::-1][:count]:
        row, col = divmod(int(flat), cols)
        located.append(
            {
                "row": row * BLOCK,
                "col": col * BLOCK,
                "size": BLOCK,
                "mean_score": float(blocks[row, col]),
            }
        )
    return located


def _validated(frames: np.ndarray, source_coordinates: np.ndarray):
    values = np.asarray(frames, dtype=np.float64)
    coordinates = np.asarray(source_coordinates, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("frames must have shape (n, height, width)")
    pairs, height, width = values.shape[0] - 1, values.shape[1], values.shape[2]
    if pairs < MINIMUM_FRAME_PAIRS:
        raise ValueError(
            f"temporal shimmer needs at least {MINIMUM_FRAME_PAIRS} frame pairs, "
            f"got {max(pairs, 0)}"
        )
    if coordinates.shape != (pairs, 2, height, width):
        raise ValueError(
            "source_coordinates must have shape (frames - 1, 2, height, width)"
        )
    return values, coordinates


def _residuals(values: np.ndarray, coordinates: np.ndarray):
    pairs, height, width = values.shape[0] - 1, values.shape[1], values.shape[2]
    residuals = np.empty((pairs, height, width))
    scored = np.ones((height, width), dtype=bool)
    for pair in range(pairs):
        predicted = map_coordinates(
            values[pair],
            coordinates[pair],
            order=INTERPOLATION_ORDER,
            mode="nearest",
        )
        residuals[pair] = values[pair + 1] - predicted
        scored &= _inside(coordinates[pair], height, width)
    return residuals, scored


def _score(
    residuals: np.ndarray, values: np.ndarray, relative: bool, magnitude_floor: float
) -> np.ndarray:
    root_mean_square = np.sqrt(np.mean(residuals**2, axis=0))
    if not relative:
        return root_mean_square
    magnitude = np.maximum(np.abs(values[1:]).mean(axis=0), magnitude_floor)
    return root_mean_square / magnitude


def shimmer_scores(
    frames: np.ndarray,
    source_coordinates: np.ndarray,
    *,
    relative: bool,
    magnitude_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's score and whether it could be scored, as two images.

    For asking where a change acted: the statistics below summarise the same
    scores over the whole frame.
    """

    values, coordinates = _validated(frames, source_coordinates)
    residuals, scored = _residuals(values, coordinates)
    return _score(residuals, values, relative, magnitude_floor), scored


def shimmer_statistics(
    frames: np.ndarray,
    source_coordinates: np.ndarray,
    *,
    threshold: float,
    relative: bool,
    magnitude_floor: float = 1e-6,
    exclude: np.ndarray | None = None,
) -> dict[str, Any]:
    """How much of a panned sequence failed to move with the scene.

    `frames` is ``(n, height, width)`` single-channel; `source_coordinates`
    holds, for each consecutive pair, where the later frame's pixels were in
    the earlier one. A pixel's score is the RMS of its prediction residual over
    all pairs — divided by its mean value when `relative` — and it counts as
    unstable above `threshold`. Only pixels whose source lies inside every
    earlier frame, clear of the interpolation border, are scored.
    """

    values, coordinates = _validated(frames, source_coordinates)
    residuals, scored = _residuals(values, coordinates)
    if exclude is not None:
        scored &= ~np.asarray(exclude, dtype=bool)
    score = _score(residuals, values, relative, magnitude_floor)
    pairs = values.shape[0] - 1
    samples = np.abs(residuals[:, scored])
    return {
        "frame_pairs": int(pairs),
        "scored_pixels": int(scored.sum()),
        "valid_pixel_fraction": float(scored.mean()),
        "threshold": float(threshold),
        "threshold_is_relative": bool(relative),
        "unstable_pixel_fraction": (
            float((score[scored] > threshold).mean()) if scored.any() else 0.0
        ),
        "mean_abs_residual": float(samples.mean()) if samples.size else 0.0,
        "p95_abs_residual": (
            float(np.percentile(samples, 95.0)) if samples.size else 0.0
        ),
        "mean_score": float(score[scored].mean()) if scored.any() else 0.0,
        "worst_blocks": _worst_blocks(np.where(scored, score, 0.0)),
    }
