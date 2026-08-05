"""Describe how two captures of the same view differ.

The existing capture comparison answers one question — did anything change by
more than a limit — which is what a regression gate needs and all it needs. It
cannot say what changed. A skyline that moved, an edge that softened and a
grade shift all read as "changed pixels".

These four measures separate those. Silhouette IoU sees geometry appearing or
disappearing against the sky. Edge displacement sees the same shapes drawn in
a different place, in pixels. The tone shift sees the image graded differently
while its shapes stayed put. Temporal stability sees a view that will not sit
still even with the camera locked.

Every value is descriptive. None of them is a pass/fail threshold, because the
right threshold depends on what the change was meant to do, and inventing one
here would turn a description into a verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np


#: Rec. 709 luminance, matching `display_sdr_statistics` so the two agree on
#: what "brightness" means for the same frame.
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

#: Depth clears to 1.0, so anything strictly nearer carries geometry. The
#: epsilon keeps a far-plane fragment from counting as sky and back.
FAR_DEPTH = 1.0
DEPTH_EPSILON = 1e-6

#: Sobel gradient magnitude, in code values per pixel, above which a pixel is
#: called an edge. Low enough to catch a soft architectural edge, high enough
#: to ignore dither and sensor noise.
DEFAULT_EDGE_THRESHOLD = 24.0


def coverage_mask(
    depth: np.ndarray, far: float = FAR_DEPTH, epsilon: float = DEPTH_EPSILON
) -> np.ndarray:
    """Boolean mask of pixels holding geometry rather than sky."""

    values = np.asarray(depth, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("depth must have shape (height, width)")
    return values < (far - epsilon)


def silhouette_iou(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Intersection over union of two coverage masks.

    Added and removed fractions come back alongside the ratio because IoU alone
    hides direction: a skyline that grew and one that shrank by the same area
    score identically.
    """

    first = np.asarray(reference, dtype=bool)
    second = np.asarray(candidate, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("masks must have the same shape")
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    pixels = int(first.size)
    return {
        "intersection_over_union": float(intersection / union) if union else 1.0,
        "reference_coverage_fraction": float(np.count_nonzero(first) / pixels),
        "candidate_coverage_fraction": float(np.count_nonzero(second) / pixels),
        "added_fraction": float(np.count_nonzero(second & ~first) / pixels),
        "removed_fraction": float(np.count_nonzero(first & ~second) / pixels),
    }


def luminance(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3)")
    return rgb.astype(np.float64) @ LUMINANCE_WEIGHTS


def edge_mask(
    frame: np.ndarray, threshold: float = DEFAULT_EDGE_THRESHOLD
) -> np.ndarray:
    """Sobel edges of a display frame's luminance."""

    values = luminance(frame)
    # Edge replication rather than zero padding: zero padding invents a bright
    # step at every border and lines the whole frame edge with false edges.
    padded = np.pad(values, 1, mode="edge")
    horizontal = (
        padded[:-2, 2:] + 2.0 * padded[1:-1, 2:] + padded[2:, 2:]
        - padded[:-2, :-2] - 2.0 * padded[1:-1, :-2] - padded[2:, :-2]
    )
    vertical = (
        padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        - padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
    )
    return np.hypot(horizontal, vertical) >= threshold


def _distance_to_nearest(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance from every pixel to the nearest True in `mask`."""

    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float64)
    return distance_transform_edt(~mask).astype(np.float64)


def edge_displacement(
    reference: np.ndarray,
    candidate: np.ndarray,
    threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> dict[str, Any]:
    """How far the drawn edges moved, in pixels.

    Symmetric on purpose. Measuring only candidate-to-reference distance scores
    zero when the candidate loses edges entirely, since whatever survives may
    still sit exactly on a reference edge.
    """

    first = edge_mask(reference, threshold)
    second = edge_mask(candidate, threshold)
    if first.shape != second.shape:
        raise ValueError("frames must have the same shape")

    forward = _distance_to_nearest(first)[second]
    backward = _distance_to_nearest(second)[first]
    both = (
        np.concatenate([forward, backward])
        if forward.size or backward.size
        else np.zeros(0)
    )
    # An edge with no counterpart anywhere is infinitely displaced, and dropping
    # those as non-finite would score a frame that lost every edge as a perfect
    # match. Saturating at the frame diagonal keeps the statistics finite while
    # still counting them as maximally far, and the unmatched fraction says how
    # much of the result rests on that substitution.
    diagonal = float(np.hypot(*first.shape))
    unmatched = int(np.count_nonzero(~np.isfinite(both))) if both.size else 0
    distances = np.where(np.isfinite(both), both, diagonal) if both.size else both
    return {
        "edge_threshold": float(threshold),
        "reference_edge_fraction": float(np.count_nonzero(first) / first.size),
        "candidate_edge_fraction": float(np.count_nonzero(second) / second.size),
        "unmatched_edge_fraction": (
            float(unmatched / distances.size) if distances.size else 0.0
        ),
        "saturation_distance_px": diagonal,
        "mean_displacement_px": (
            float(distances.mean()) if distances.size else 0.0
        ),
        "p95_displacement_px": (
            float(np.percentile(distances, 95)) if distances.size else 0.0
        ),
        "max_displacement_px": float(distances.max()) if distances.size else 0.0,
        "edge_intersection_over_union": (
            float(np.count_nonzero(first & second) / np.count_nonzero(first | second))
            if np.count_nonzero(first | second)
            else 1.0
        ),
    }


def tone_distribution_shift(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    """How the grade moved, independent of where anything is.

    The luminance figure is the Wasserstein-1 distance between the two
    distributions, which for one dimension is the area between their CDFs. It
    reads directly as "the typical pixel moved this many code values", and it
    stays meaningful when a change brightens some pixels and darkens others,
    where a mean difference would cancel to nothing.
    """

    first = np.asarray(reference, dtype=np.uint8)
    second = np.asarray(candidate, dtype=np.uint8)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("frames must have the same (height, width, 3) shape")

    first_luminance = luminance(first)
    second_luminance = luminance(second)
    bins = np.arange(257)
    reference_cdf = np.cumsum(
        np.histogram(first_luminance, bins=bins)[0]
    ) / first_luminance.size
    candidate_cdf = np.cumsum(
        np.histogram(second_luminance, bins=bins)[0]
    ) / second_luminance.size
    return {
        "luminance_earth_mover_code_values": float(
            np.abs(reference_cdf - candidate_cdf).sum()
        ),
        "median_luminance_shift": float(
            np.median(second_luminance) - np.median(first_luminance)
        ),
        "median_rgb8_shift": (
            np.median(second.reshape(-1, 3), axis=0)
            - np.median(first.reshape(-1, 3), axis=0)
        ).tolist(),
        "clipped_black_fraction_shift": float(
            np.all(second == 0, axis=2).mean() - np.all(first == 0, axis=2).mean()
        ),
        "clipped_white_fraction_shift": float(
            np.all(second == 255, axis=2).mean()
            - np.all(first == 255, axis=2).mean()
        ),
    }


def temporal_stability(
    frames: list[np.ndarray], flicker_threshold: int = 2
) -> dict[str, Any]:
    """How still a locked-off view sits across consecutive frames.

    The camera does not move between these frames, so every difference is the
    renderer disagreeing with itself: shimmer on tiled material, an adaptation
    state still settling, sampling noise. The flicker fraction is the share of
    pixels that move by more than a code value or two, which is roughly where
    it stops being quantisation and starts being visible.
    """

    if len(frames) < 2:
        raise ValueError("temporal stability needs at least two frames")
    stack = [np.asarray(frame, dtype=np.int16) for frame in frames]
    if any(frame.shape != stack[0].shape for frame in stack):
        raise ValueError("frames must have the same shape")

    successive = [
        np.abs(later - earlier) for earlier, later in zip(stack, stack[1:])
    ]
    differences = np.stack(successive)
    per_pixel = differences.max(axis=3)
    return {
        "frame_count": len(stack),
        "mean_successive_difference_rgb8": float(differences.mean()),
        "p99_successive_difference_rgb8": float(np.percentile(differences, 99)),
        "max_successive_difference_rgb8": int(differences.max()),
        "flicker_threshold_rgb8": int(flicker_threshold),
        "flickering_pixel_fraction": float(
            (per_pixel > flicker_threshold).any(axis=0).mean()
        ),
    }


def compare_frames(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    reference_coverage: np.ndarray | None = None,
    candidate_coverage: np.ndarray | None = None,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> dict[str, Any]:
    """Bundle the descriptive measures for one view.

    Silhouette is omitted rather than guessed when no coverage mask is
    supplied: segmenting sky out of a display frame would invent a geometry
    boundary from a grade, which is the one thing these measures exist to keep
    apart.
    """

    report: dict[str, Any] = {
        "edges": edge_displacement(reference, candidate, edge_threshold),
        "tone": tone_distribution_shift(reference, candidate),
    }
    if reference_coverage is not None and candidate_coverage is not None:
        report["silhouette"] = silhouette_iou(
            reference_coverage, candidate_coverage
        )
    else:
        report["silhouette"] = None
        report["silhouette_absent_because"] = (
            "no coverage mask was captured for this view"
        )
    return report
