"""Run the V1-3 road-tessellation and north-shoreline cause analysis."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from simulator.scene import load_scene
from simulator.validation.terrain_causes import (
    DEFAULT_PRIORITY_AREAS_PATH,
    analyze_terrain_causes,
    load_priority_area,
    render_terrain_cause_map,
)
from simulator.validation.terrain_contacts import audit_terrain_contacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--areas", type=Path, default=DEFAULT_PRIORITY_AREAS_PATH)
    parser.add_argument("--area-id", default="event_park_north_bank")
    parser.add_argument("--subdivision-levels", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        area = load_priority_area(arguments.areas, arguments.area_id)
        area.verify_scene_path(arguments.scene)
        scene = load_scene(arguments.scene)
        audit = audit_terrain_contacts(scene)
        analysis = analyze_terrain_causes(
            scene,
            audit,
            area,
            maximum_subdivision_level=arguments.subdivision_levels,
        )
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        map_path = render_terrain_cause_map(
            scene,
            audit,
            area,
            analysis,
            arguments.output_dir / "terrain_cause_map.png",
        )
        report = {
            **analysis.report,
            "scene_asset": _portable_path(arguments.scene),
            "scene_asset_sha256": sha256(arguments.scene.read_bytes()).hexdigest(),
            "priority_area_asset": _portable_path(arguments.areas),
            "priority_area_asset_sha256": sha256(
                arguments.areas.read_bytes()
            ).hexdigest(),
            "map": map_path.name,
        }
        report_path = arguments.output_dir / "terrain_cause_report.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "report": str(report_path),
                "map": str(map_path),
                "priority_area": report["priority_area"],
                "counterfactual": report["road_tessellation_counterfactual"],
                "north_shoreline": report["north_shoreline"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
