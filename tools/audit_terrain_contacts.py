"""Generate the V1 terrain, shoreline and draped-road visual defect map.

Example::

    python -m tools.audit_terrain_contacts \
        --output-dir docs/validation/terrain_contact_v1
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from simulator.scene import load_scene
from simulator.validation.terrain_contacts import (
    TerrainContactThresholds,
    audit_terrain_contacts,
    render_terrain_contact_map,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPOSITORY_ROOT / "assets" / "yeouido_scene.npz"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zone-size-m", type=float, default=100.0)
    arguments = parser.parse_args()
    try:
        scene = load_scene(arguments.scene)
        thresholds = TerrainContactThresholds(zone_size_m=arguments.zone_size_m)
        audit = audit_terrain_contacts(scene, thresholds)
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        map_path = render_terrain_contact_map(
            scene, audit, arguments.output_dir / "terrain_contact_audit.png"
        )
        report = {
            **audit.report,
            "scene_asset": _portable_path(arguments.scene),
            "scene_asset_sha256": _sha256(arguments.scene),
            "map": map_path.name,
        }
        report_path = arguments.output_dir / "terrain_contact_audit.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "report": str(report_path),
        "map": str(map_path),
        "terrain": report["terrain"],
        "shoreline": report["shoreline"],
        "roads": report["roads"],
        "priority_counts": report["zones"]["priority_counts"],
        "top_zones": report["zones"]["ordered"][:10],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
