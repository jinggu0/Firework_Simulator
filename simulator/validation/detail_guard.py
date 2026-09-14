"""Guard a shimmer fix against what it takes away.

Blurring a frame also makes it move more cleanly, so a lower temporal shimmer
score cannot accept a fix on its own. What a fix must not do is carry the frame
further from a well-sampled render at the scales the pixel grid resolves.

The comparison is against a supersampled reference, but never pixel by pixel.
At a single pixel the reference is simply more detailed, and fading detail the
grid cannot carry reads as an error — the trap V2-3b fell into. Over a block of
several pixels the reference averages hundreds of sub-samples while a 1x frame
averages a handful of point samples, so a fade to the right mean comes closer
to the reference, and a wrong mean, a flattened variation or a blur of visible
structure moves away from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np


#: The finest block is the smallest over which point sampling averages out
#: enough to compare with a reference; the coarsest catches a shift in mean
#: brightness that per-block noise could hide.
GUARD_SCALES_PX = (4, 16)


def _block_average(frame: np.ndarray, scale: int) -> np.ndarray:
    height, width = frame.shape
    if height % scale or width % scale:
        raise ValueError(f"a {scale} px block does not divide a {width}x{height} frame")
    return frame.reshape(height // scale, scale, width // scale, scale).mean(axis=(1, 3))


def block_distance_to_reference(
    frame: np.ndarray, reference: np.ndarray, scale: int
) -> float:
    """L1 distance of block averages, as a share of the reference's light."""

    values = np.asarray(frame, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or values.shape != target.shape:
        raise ValueError("frame and reference must share one (height, width) shape")
    blocks = _block_average(values, scale)
    reference_blocks = _block_average(target, scale)
    total = float(np.abs(reference_blocks).sum())
    difference = float(np.abs(blocks - reference_blocks).sum())
    if total == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / total


def detail_guard(
    baseline: np.ndarray, candidate: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    """Distance to the reference of the baseline and the candidate, per scale."""

    return {
        "scales_px": list(GUARD_SCALES_PX),
        "scales": {
            str(scale): {
                "baseline": block_distance_to_reference(baseline, reference, scale),
                "candidate": block_distance_to_reference(candidate, reference, scale),
            }
            for scale in GUARD_SCALES_PX
        },
    }
