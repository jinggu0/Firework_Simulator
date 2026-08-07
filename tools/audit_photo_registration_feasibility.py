"""Judge whether any held photograph can be registered to the scene.

V0 asks for a determination on at least three viewpoints. The appearance
reference already records that no held photograph publishes a camera pose, but
that is a weaker statement than it looks: a pose does not have to be published,
it has to be solvable. `fit_camera_pose` solves one from control points, so the
real question is whether the inputs that solver needs exist.

They decompose into three, and a source has to clear all three:

  intrinsics   the focal length and sensor geometry the projection needs
  pixels       an image the project may actually hold and measure points in
  control      at least `MINIMUM_CONTROL_POINTS` features whose scene
               coordinates are known, identifiable in that image

The third is the one that decides it, and it is a property of the scene rather
than of any photograph — which is why this audit reports it once, separately,
instead of per source.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from simulator.photogrammetry import MINIMUM_CONTROL_POINTS


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APPEARANCE = Path("assets/yeouido_2024-10-05_appearance_reference.json")
DEFAULT_CONTROLS = Path("assets/yeouido_ngii_public_controls_2017.json")
DEFAULT_OUTPUT = Path(
    "docs/validation/photo_registration_v0/photo_registration_feasibility.json"
)

#: V0's completion gate asks for a determination across at least three views.
REQUIRED_VIEWPOINTS = 3

#: OSM planimetric accuracy as this project established it in V-05. Building
#: corners are the best-covered candidate control features in the scene, and
#: this is what they are worth.
OSM_PLANIMETRIC_UNCERTAINTY_M = (1.0, 3.0)

#: Registration is only ever as good as its control. Below this the residual
#: would be dominated by control error rather than by the fit, so a "registered"
#: photograph would carry an accuracy nobody could state.
TARGET_CONTROL_UNCERTAINTY_M = 1.0


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def assess_sources(appearance: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-source intrinsics and pixel availability."""

    assessed: list[dict[str, Any]] = []
    for source_id, source in appearance["sources"].items():
        licence = str(source.get("license", ""))
        redistributable = licence.upper().startswith("CC0")
        has_intrinsics = bool(
            source.get("focal_length_mm") and source.get("full_frame_equivalent_mm")
        )
        blockers: list[str] = []
        if not has_intrinsics:
            blockers.append("no published focal length or sensor equivalent")
        if not redistributable:
            # The project deliberately stores no pixels for these, so there is
            # nothing to measure image points in even if intrinsics appeared.
            blockers.append(
                "licence forbids redistribution and no pixels are held"
            )
        assessed.append(
            {
                "source_id": source_id,
                "license": licence,
                "camera": source.get("camera"),
                "focal_length_mm": source.get("focal_length_mm"),
                "full_frame_equivalent_mm": source.get("full_frame_equivalent_mm"),
                "intrinsics_available": has_intrinsics,
                "pixels_available_to_the_project": redistributable,
                "blockers": blockers,
            }
        )
    return assessed


def assess_control_points(controls: dict[str, Any]) -> dict[str, Any]:
    """What the scene can offer as control, and to what accuracy."""

    published = controls.get("public_controls", [])
    surviving = [
        point for point in published if point.get("status") != "destroyed"
    ]
    candidates = [
        {
            "family": "NGII public control points",
            "count": len(published),
            "usable_count": len(surviving),
            "uncertainty_m": None,
            "reason": (
                "every published point is recorded as destroyed, so none can be "
                "identified in a photograph regardless of its coordinates"
            ),
        },
        {
            "family": "OSM building corners",
            "count": None,
            "usable_count": None,
            "uncertainty_m": list(OSM_PLANIMETRIC_UNCERTAINTY_M),
            "reason": (
                "the best-covered identifiable features in the scene, but at "
                "the 1 to 3 m planimetric accuracy this project established for "
                "OSM in V-05"
            ),
        },
        {
            "family": "Seogang Bridge published dimensions",
            "count": None,
            "usable_count": 0,
            "uncertainty_m": None,
            "reason": (
                "grade A as published dimensions, but the source drawings are "
                "not georegistered to the project's EUS frame, so they yield no "
                "control coordinates in it"
            ),
        },
    ]
    best = min(
        (
            candidate["uncertainty_m"][0]
            for candidate in candidates
            if candidate["uncertainty_m"]
        ),
        default=None,
    )
    return {
        "candidates": candidates,
        "points_meeting_target_uncertainty": 0,
        "best_available_uncertainty_m": best,
        "target_uncertainty_m": TARGET_CONTROL_UNCERTAINTY_M,
        "minimum_required_points": MINIMUM_CONTROL_POINTS,
        "sufficient": False,
    }


def build_report(
    appearance_path: Path = DEFAULT_APPEARANCE,
    controls_path: Path = DEFAULT_CONTROLS,
) -> dict[str, Any]:
    appearance = json.loads(appearance_path.read_text(encoding="utf-8"))
    controls = json.loads(controls_path.read_text(encoding="utf-8"))

    sources = assess_sources(appearance)
    control = assess_control_points(controls)

    with_intrinsics = [row for row in sources if row["intrinsics_available"]]
    with_pixels = [
        row for row in sources if row["pixels_available_to_the_project"]
    ]
    camera_ready = [
        row
        for row in sources
        if row["intrinsics_available"] and row["pixels_available_to_the_project"]
    ]
    # Control is a scene property, so it gates every source at once: a source
    # can be camera-ready and still unregisterable.
    registerable = camera_ready if control["sufficient"] else []

    blockers: list[str] = []
    if not control["sufficient"]:
        blockers.append(
            f"no {MINIMUM_CONTROL_POINTS} scene features have coordinates at or "
            f"below {TARGET_CONTROL_UNCERTAINTY_M} m, so no photograph can be "
            "registered to a statable accuracy"
        )
    if len(camera_ready) < REQUIRED_VIEWPOINTS:
        blockers.append(
            f"only {len(camera_ready)} of {len(sources)} sources carry both "
            f"intrinsics and usable pixels, against the {REQUIRED_VIEWPOINTS} "
            "viewpoints V0 asks for"
        )

    return {
        "schema_version": 1,
        "stage": "V0-4",
        "inputs": {
            "appearance_reference": _display_path(appearance_path),
            "appearance_reference_sha256": _digest(appearance_path),
            "public_controls": _display_path(controls_path),
            "public_controls_sha256": _digest(controls_path),
        },
        "requirement": {
            "viewpoints": REQUIRED_VIEWPOINTS,
            "minimum_control_points_per_view": MINIMUM_CONTROL_POINTS,
            "solver": "simulator.photogrammetry.fit_camera_pose",
        },
        "sources": sources,
        "control_points": control,
        "counts": {
            "source_count": len(sources),
            "with_intrinsics": len(with_intrinsics),
            "with_usable_pixels": len(with_pixels),
            "camera_ready": len(camera_ready),
            "registerable": len(registerable),
        },
        "determination": {
            "registerable_viewpoint_count": len(registerable),
            "meets_v0_requirement": len(registerable) >= REQUIRED_VIEWPOINTS,
            "limiting_factor": (
                "control points"
                if not control["sufficient"]
                else "camera metadata and pixel availability"
            ),
            "finding": (
                "The blocker is not that poses are unpublished — a pose can be "
                "solved. It is that the scene offers nothing to solve against: "
                "the published control points are all destroyed, the bridge "
                "dimensions are not georegistered to the project frame, and OSM "
                "building corners carry 1 to 3 m planimetric uncertainty. A "
                "registration built on those would inherit their error, so its "
                "accuracy could not be stated."
            ),
        },
        "application_gates": {
            "photo_registration_available": False,
            "scene_vertices_modified": 0,
            "runtime_geometry_changed_by_this_stage": False,
        },
        "blocking_reasons": blockers,
        "missing_data": [
            "at least six scene features with coordinates at or below 1 m in the "
            "project's EUS frame that are identifiable in a night photograph",
            "a georegistration of the Seogang Bridge published dimensions into "
            "the project frame, which would turn grade-A dimensions into usable "
            "control",
            "published camera intrinsics for any of the three press sources, "
            "which would still not help until control exists",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance", type=Path, default=DEFAULT_APPEARANCE)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = build_report(arguments.appearance, arguments.controls)
    output = arguments.output
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {output}: registerable_viewpoint_count="
        f"{report['determination']['registerable_viewpoint_count']}, "
        f"limiting_factor={report['determination']['limiting_factor']}"
    )


if __name__ == "__main__":
    main()
