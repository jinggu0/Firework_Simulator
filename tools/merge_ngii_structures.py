"""Merge audited NGII retaining-wall and earthwork profiles into a scene."""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from simulator.scene import load_scene, save_scene
from simulator.structures import StructureEvidenceError, build_structure_mesh


def _read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256(raw).hexdigest()


def _file_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("assets/yeouido_scene.npz"))
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-post-event-source", action="store_true")
    arguments = parser.parse_args()

    try:
        asset, asset_checksum = _read_json(arguments.structures)
        profiles, profile_checksum = _read_json(arguments.profiles)
        declared_checksum = str(profiles.get("source_asset_sha256", ""))
        if declared_checksum != asset_checksum:
            raise StructureEvidenceError(
                "profile source_asset_sha256 does not match the structure asset"
            )
        scene_checksum = _file_checksum(arguments.scene)
        scene = load_scene(arguments.scene)
        result = build_structure_mesh(
            asset,
            profiles,
            scene.terrain_height_m,
            scene.terrain_bounds,
            allow_post_event_source=arguments.allow_post_event_source,
        )
    except (OSError, json.JSONDecodeError, StructureEvidenceError) as error:
        parser.error(str(error))

    output_scene = replace(scene, structure_vertices=result.vertices)
    save_scene(output_scene, arguments.output)
    output_checksum = _file_checksum(arguments.output)
    report_path = arguments.report or arguments.output.with_suffix(
        ".structure_provenance.json"
    )
    report = {
        "schema_version": 1,
        "scene_input": str(arguments.scene),
        "scene_input_sha256": scene_checksum,
        "scene_output": str(arguments.output),
        "scene_output_sha256": output_checksum,
        "structure_asset": str(arguments.structures),
        "structure_asset_sha256": asset_checksum,
        "profile_document": str(arguments.profiles),
        "profile_document_sha256": profile_checksum,
        "target_event_date": asset.get("target_event_date"),
        "temporal_relation": asset.get("temporal_relation"),
        "profiles_built": result.profiles_built,
        "source_segments": result.source_segments,
        "rendered_segments": result.rendered_segments,
        "skipped_segments": result.skipped_segments,
        "vertices": len(result.vertices),
        "triangles": len(result.vertices) // 3,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"saved {arguments.output}: {len(result.vertices):,} structure vertices, "
        f"{result.profiles_built} audited profiles; provenance {report_path}"
    )


if __name__ == "__main__":
    main()
