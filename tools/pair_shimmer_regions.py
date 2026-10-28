"""Pair-local semantic support of the unchanged cubic prediction residual.

Energy (squared residual), NOT thresholded RMS excess, is additive across
changing labels. These diagnostic shares do not change candidate acceptance.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from simulator.validation.temporal_shimmer import INTERPOLATION_ORDER, pan_source_coordinates


# Declared before rendering: 2x2 centre support, 4x4 cubic coefficient support,
# and a conservative 16x16 neighbourhood. The spline prefilter is not compact;
# even the last neighbourhood is a sensitivity check, not an exact isolation.
SUPPORT_RADII = (0, 1, 7)
REGIONS = (
    "facade_boundary_or_mixed_support",
    "pane_interior_clean_support",
    "pane_centres_mixed_support",
    "wall_interior_clean_support",
    "facade_internal_transition",
    "water_clean_support",
    "other",
)


def source_support(mask, coordinates, radius):
    """All source pixels in floor(query)+[-radius, radius+1]^2 match mask.

    Integral rectangles avoid interpolating labels or eroding in target space.
    Out-of-image support is never clean. Coordinates are (row, column).
    """
    mask = np.asarray(mask, dtype=bool)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if mask.ndim != 2 or coordinates.shape != (2, *mask.shape) or not np.isfinite(coordinates).all():
        raise ValueError("expected finite source coordinates and a matching 2D mask")
    if type(radius) is not int or radius < 0:
        raise ValueError("support radius must be a nonnegative integer")
    height, width = mask.shape
    row, col = np.floor(coordinates).astype(np.int64)
    lo_r, lo_c = row - radius, col - radius
    hi_r, hi_c = row + radius + 2, col + radius + 2
    valid = (lo_r >= 0) & (lo_c >= 0) & (hi_r <= height) & (hi_c <= width)
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    lo_r, hi_r = np.clip(lo_r, 0, height), np.clip(hi_r, 0, height)
    lo_c, hi_c = np.clip(lo_c, 0, width), np.clip(hi_c, 0, width)
    count = integral[hi_r, hi_c] - integral[lo_r, hi_c] - integral[hi_r, lo_c] + integral[lo_r, lo_c]
    return valid & (count == (2 * radius + 2) ** 2)


def pair_masks(previous, current, coordinates, radius):
    """Disjoint receiver labels, conditioned on reprojected SOURCE support."""
    previous, current = np.asarray(previous), np.asarray(current)
    if previous.shape != current.shape or current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError("expected matching height x width x 3 fractions")
    for values in (previous, current):
        if (not np.isfinite(values).all() or values.min() < -.001 or values.max() > 1.001
                or np.any(values[..., 1] > values[..., 0] + .001)
                or np.any(values[..., 0] + values[..., 2] > 1.001)):
            raise ValueError("invalid semantic fractions")
    pf, pp, pw = (previous[..., k] for k in range(3))
    cf, cp, cw = (current[..., k] for k in range(3))
    facade = (cf >= .999) & source_support(pf >= .999, coordinates, radius)
    # Any facade in the source rectangle includes an approaching silhouette,
    # even when the current target pixel itself has no facade coverage.
    no_facade = source_support(pf <= .001, coordinates, radius)
    in_source = source_support(np.ones(pf.shape, bool), coordinates, radius)
    boundary = ((cf > .001) | (~no_facade & in_source)) & ~facade
    pane = facade & (cp >= .999) & source_support(pp >= .999, coordinates, radius)
    pane_mixed = facade & (cp >= .999) & source_support(pp >= .999, coordinates, 0) & ~pane
    wall = facade & (cp <= .001) & source_support((pf >= .999) & (pp <= .001), coordinates, radius)
    water = (cw >= .999) & source_support(pw >= .999, coordinates, radius) & ~boundary
    masks = dict(zip(REGIONS[:-1], (boundary, pane, pane_mixed, wall,
                                  facade & ~(pane | pane_mixed | wall), water), strict=True))
    masks["other"] = ~np.logical_or.reduce(list(masks.values()))
    return masks


def pair_statistics(pan, run, radii=SUPPORT_RADII, progress=None):
    """Stream pairs; retain the existing scored/excluded domain and HDR scale.

    Sum(region energy)/pairs equals sum(existing RMS score squared). No per-pair
    threshold is substituted for the original 2-code / .05 RMS thresholds.
    """
    display, hdr = pan["display_luma"], pan["linear_hdr"]
    fractions, matrices = pan["semantic_fractions"], pan["matrices"]
    scored = np.asarray(run["scored"], bool)
    if (display.shape != hdr.shape or display.ndim != 3 or display.shape[0] < 2
            or fractions.shape != (*display.shape, 3) or len(matrices) != len(display)
            or scored.shape != display.shape[1:]):
        raise ValueError("incompatible captured sequence shapes")
    pairs, height, width = len(display) - 1, *scored.shape
    magnitude = np.maximum(np.abs(hdr[1:]).mean(axis=0), 1e-6)
    accum = {radius: {name: {"samples": 0, "min_samples_per_pair": height * width,
                            "max_samples_per_pair": 0, "display_energy": 0., "hdr_energy": 0.}
                      for name in REGIONS} for radius in radii}
    for pair in range(pairs):
        coordinates = pan_source_coordinates(matrices[pair], matrices[pair + 1], width, height)
        energy = {}
        for key, values in (("display", display), ("hdr", hdr)):
            residual = values[pair + 1] - map_coordinates(values[pair], coordinates,
                                                         order=INTERPOLATION_ORDER, mode="nearest")
            if key == "hdr":
                residual = residual / magnitude
            energy[key] = residual ** 2
        for radius in radii:
            masks = pair_masks(fractions[pair], fractions[pair + 1], coordinates, radius)
            for name, mask in masks.items():
                selected = mask & scored
                count = int(selected.sum())
                item = accum[radius][name]
                item["samples"] += count
                item["min_samples_per_pair"] = min(item["min_samples_per_pair"], count)
                item["max_samples_per_pair"] = max(item["max_samples_per_pair"], count)
                for key, values in energy.items():
                    item[f"{key}_energy"] += float(values[selected].sum())
        if progress is not None:
            progress(pair + 1, pairs)
    total_samples = pairs * int(scored.sum())
    results = []
    for radius, regions in accum.items():
        totals = {key: sum(item[f"{key}_energy"] for item in regions.values()) for key in ("display", "hdr")}
        conservation = {}
        for key in totals:
            expected = float((run[f"{key}_score"][scored] ** 2).sum()) * pairs
            if not np.isclose(totals[key], expected, rtol=1e-10, atol=1e-10):
                raise ValueError(f"{key} residual energy does not reconstruct the existing RMS score")
            conservation[key] = {"pair_energy_sum": totals[key], "existing_score_squared_sum_times_pairs": expected}
        for item in regions.values():
            count = item["samples"]
            item["sample_fraction"] = count / total_samples if total_samples else 0.
            for key, total in totals.items():
                item[f"{key}_energy_share"] = item[f"{key}_energy"] / total if total else 0.
                item[f"{key}_conditional_rms"] = float(np.sqrt(item[f"{key}_energy"] / count)) if count else None
        results.append({"source_support_radius": radius, "source_support_width_px": 2 * radius + 2,
                        "scored_pixel_pairs": total_samples, "conservation": conservation, "regions": regions})
    return results
