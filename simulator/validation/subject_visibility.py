"""Screen-space gates for separate engineering cameras, not historical poses."""
from __future__ import annotations

import numpy as np


NEAR_OCCLUDER_M = 2.0
GATES = {
    "bridge_structure": (0.02, 0.10, 0.75, 0.05),
    "road_ground": (0.15, 0.35, 0.65, 0.05),
}


def measure_visibility(visible, isolated):
    """Resolved MSAA fractions: subject, near non-subject, building.

    The isolated pass counts the subject only where it projects in-frame.
    Its difference from the full-scene pass measures occlusion, not framing.
    """
    full = np.asarray(visible, dtype=np.float32)
    alone = np.asarray(isolated, dtype=np.float32)
    if full.shape != alone.shape or full.ndim != 3 or full.shape[2] != 3 or min(full.shape[:2]) < 8:
        raise ValueError("expected matching H×W×3 visibility masks (at least 8×8)")
    for values in (full, alone):
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("visibility fractions must be finite and in [0, 1]")
        if np.any(values.sum(axis=2) > 1.001):
            raise ValueError("visibility channels must be disjoint")
    subject = full[..., 0]
    support = alone[..., 0]
    if np.any(alone[..., 1:]) or np.any(subject > support + .001):
        raise ValueError("visible subject must be a subset of isolated support")
    height, width = subject.shape
    centre = subject[3 * height // 8:5 * height // 8, 3 * width // 8:5 * width // 8]
    projected = float(support.mean(dtype=np.float64))
    visible_fraction = float(subject.mean(dtype=np.float64))
    return {
        "visible_screen_fraction": visible_fraction,
        "isolated_screen_fraction": projected,
        "visible_projected_fraction": min(visible_fraction / projected, 1.0) if projected else None,
        "central_subject_fraction": float(centre.mean(dtype=np.float64)),
        "near_non_subject_fraction": float(full[..., 1].mean(dtype=np.float64)),
        "other_building_fraction": float(full[..., 2].mean(dtype=np.float64)),
    }


def visibility_gates(subject, metrics):
    screen, centre, unoccluded, near = GATES[subject]
    return {
        "screen_coverage": metrics["visible_screen_fraction"] >= screen,
        "central_coverage": metrics["central_subject_fraction"] >= centre,
        "projected_visibility": (
            metrics["visible_projected_fraction"] is not None
            and metrics["visible_projected_fraction"] >= unoccluded
        ),
        "near_occluders": metrics["near_non_subject_fraction"] <= near,
    }
