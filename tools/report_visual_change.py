"""Describe what changed between two visual baseline captures.

`capture_visual_baselines --compare` already answers whether two captures
differ by more than a regression limit. This answers what the difference was:
whether shapes moved, whether the silhouette gained or lost geometry, and
whether the grade shifted underneath unchanged shapes.

It reads the manifests that the capture tool writes, so it works on any two
capture directories without re-rendering. Views present in only one of them are
reported as such rather than skipped silently.

Example::

    python -m tools.capture_visual_baselines --output-dir before
    # ... make a change ...
    python -m tools.capture_visual_baselines --output-dir after
    python -m tools.report_visual_change before after
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from simulator.validation.capture import load_coverage_mask
from simulator.validation.frame_comparison import (
    DEFAULT_EDGE_THRESHOLD,
    compare_frames,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load_sdr(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_coverage(directory: Path, view_id: str) -> np.ndarray | None:
    """The captured geometry mask, or nothing if this capture predates them."""

    path = directory / f"{view_id}.coverage.png"
    if not path.is_file():
        return None
    return load_coverage_mask(path)


def _view_ids(directory: Path) -> dict[str, Path]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{directory} has no manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views: dict[str, Path] = {}
    for entry in manifest.get("views", []):
        view_id = entry.get("view_id") if isinstance(entry, dict) else entry
        if not view_id:
            continue
        candidate = directory / f"{view_id}.sdr.png"
        if candidate.is_file():
            views[str(view_id)] = candidate
    if not views:
        # Older manifests may not list views; fall back to what is on disk so a
        # report is still possible rather than silently empty.
        for candidate in sorted(directory.glob("*.sdr.png")):
            views[candidate.name.removesuffix(".sdr.png")] = candidate
    return views


def build_report(
    before: Path,
    after: Path,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> dict[str, Any]:
    reference_views = _view_ids(before)
    candidate_views = _view_ids(after)
    shared = sorted(set(reference_views) & set(candidate_views))

    views: dict[str, Any] = {}
    for view_id in shared:
        reference = _load_sdr(reference_views[view_id])
        candidate = _load_sdr(candidate_views[view_id])
        if reference.shape != candidate.shape:
            views[view_id] = {
                "comparable": False,
                "reason": (
                    f"resolution changed from {reference.shape[1]}x"
                    f"{reference.shape[0]} to {candidate.shape[1]}x"
                    f"{candidate.shape[0]}"
                ),
            }
            continue
        reference_coverage = _load_coverage(before, view_id)
        candidate_coverage = _load_coverage(after, view_id)
        result = compare_frames(
            reference,
            candidate,
            reference_coverage=reference_coverage,
            candidate_coverage=candidate_coverage,
            edge_threshold=edge_threshold,
        )
        result["comparable"] = True
        result["unchanged"] = bool(np.array_equal(reference, candidate))
        views[view_id] = result

    changed = [
        view_id
        for view_id, result in views.items()
        if result.get("comparable") and not result["unchanged"]
    ]
    return {
        "schema_version": 1,
        "before": before.as_posix(),
        "after": after.as_posix(),
        "edge_threshold": float(edge_threshold),
        "view_count": len(shared),
        "views_only_in_before": sorted(set(reference_views) - set(candidate_views)),
        "views_only_in_after": sorted(set(candidate_views) - set(reference_views)),
        "changed_view_ids": changed,
        "unchanged_view_ids": [
            view_id
            for view_id, result in views.items()
            if result.get("comparable") and result["unchanged"]
        ],
        "views": views,
        "note": (
            "Descriptive only. These numbers say how two captures differ, not "
            "whether the difference is an improvement; the regression gate in "
            "capture_visual_baselines --compare is what passes or fails."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    report = build_report(arguments.before, arguments.after, arguments.edge_threshold)
    if arguments.output is not None:
        output = arguments.output
        if not output.is_absolute():
            output = REPOSITORY_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote {output}: {len(report['changed_view_ids'])} of "
            f"{report['view_count']} views changed"
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


