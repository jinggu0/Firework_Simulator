from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
import json
from pathlib import Path
import urllib.parse
import urllib.request

from simulator.scene import build_scene, build_water_mask, load_scene, save_scene
from simulator.terrain import build_terrain_heightmap

BBOX = (37.515, 126.910, 37.545, 126.960)
ORIGIN = (37.529, 126.935)
SNAPSHOT_UTC = "2024-10-05T10:20:00Z"


def _overpass(query: str) -> dict:
    last_error: Exception | None = None
    for endpoint in (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ):
        request = urllib.request.Request(
            endpoint,
            data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": "FireworkSimulator/0.1 (local research project)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.load(response)
                payload["_retrieval_endpoint"] = endpoint
                return payload
        except Exception as error:
            last_error = error
    raise RuntimeError("all configured Overpass endpoints failed") from last_error


def download(snapshot_utc: str = SNAPSHOT_UTC) -> dict:
    south, west, north, east = BBOX
    query = f"""
[out:json][timeout:240][date:"{snapshot_utc}"];
(
  way["building"]({south},{west},{north},{east});
  way["building:part"]({south},{west},{north},{east});
  way["bridge"]({south},{west},{north},{east});
  way["highway"]({south},{west},{north},{east});
  way["man_made"="embankment"]({south},{west},{north},{east});
  way["embankment"]({south},{west},{north},{east});
  way["leisure"~"^(park|playground|pitch|track|garden)$"]({south},{west},{north},{east});
  way["landuse"~"^(grass|meadow|flowerbed|forest|recreation_ground)$"]({south},{west},{north},{east});
  way["natural"~"^(wood|grassland|scrub)$"]({south},{west},{north},{east});
);
out geom;
"""
    return _overpass(query)


def download_han_river(snapshot_utc: str = SNAPSHOT_UTC) -> dict:
    query = f"""
[out:json][timeout:240][date:"{snapshot_utc}"];
relation(152336);
(._;>>;);
out body;
"""
    return _overpass(query)


def road_query(snapshot_utc: str = SNAPSHOT_UTC) -> str:
    south, west, north, east = BBOX
    return f"""
[out:json][timeout:240][date:"{snapshot_utc}"];
way["highway"]({south},{west},{north},{east});
out geom tags;
"""


def download_roads(snapshot_utc: str = SNAPSHOT_UTC) -> dict:
    return _overpass(road_query(snapshot_utc))


def write_road_source_snapshot(
    payload: dict,
    output: Path,
    snapshot_utc: str = SNAPSHOT_UTC,
) -> None:
    elements = [
        {
            "type": element.get("type", "way"),
            "id": int(element["id"]),
            "geometry": element.get("geometry", []),
            "tags": element.get("tags", {}),
        }
        for element in payload.get("elements", [])
        if element.get("type") == "way"
        and element.get("tags", {}).get("highway")
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "OpenStreetMap contributors via Overpass API",
                "licence": "ODbL 1.0",
                "endpoint": payload.get("_retrieval_endpoint", ""),
                "query": road_query(snapshot_utc).strip(),
                "bbox_wgs84": [BBOX[1], BBOX[0], BBOX[3], BBOX[2]],
                "snapshot_utc": snapshot_utc,
                "retrieved_date": date.today().isoformat(),
                "elements": elements,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )


def building_tag_query(snapshot_utc: str = SNAPSHOT_UTC) -> str:
    """Dated building tags, without geometry.

    The scene build already downloads building geometry and consumes it. What
    it does not keep is the tags, so nothing afterwards can tell whether a
    given building's height came from a `height` tag, from `building:levels`
    times an assumed storey height, or from the untagged fallback. This asks
    for exactly that evidence and nothing else: `out tags` keeps the archive
    small enough to version alongside the roads snapshot.
    """

    south, west, north, east = BBOX
    return f"""
[out:json][timeout:240][date:"{snapshot_utc}"];
(
  way["building"]({south},{west},{north},{east});
  way["building:part"]({south},{west},{north},{east});
);
out tags;
"""


def download_building_tags(snapshot_utc: str = SNAPSHOT_UTC) -> dict:
    return _overpass(building_tag_query(snapshot_utc))


def write_building_tag_snapshot(
    payload: dict,
    output: Path,
    snapshot_utc: str = SNAPSHOT_UTC,
) -> None:
    elements = [
        {
            "type": element.get("type", "way"),
            "id": int(element["id"]),
            "tags": element.get("tags", {}),
        }
        for element in payload.get("elements", [])
        if element.get("type") == "way"
        and (
            element.get("tags", {}).get("building")
            or element.get("tags", {}).get("building:part")
        )
    ]
    elements.sort(key=lambda element: element["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": "OpenStreetMap contributors via Overpass API",
                "licence": "ODbL 1.0",
                "endpoint": payload.get("_retrieval_endpoint", ""),
                "query": building_tag_query(snapshot_utc).strip(),
                "geometry_omitted": True,
                "geometry_omitted_reason": (
                    "This archive exists to evidence height provenance, not to "
                    "rebuild geometry; the scene build downloads its own."
                ),
                "bbox_wgs84": [BBOX[1], BBOX[0], BBOX[3], BBOX[2]],
                "snapshot_utc": snapshot_utc,
                "retrieved_date": date.today().isoformat(),
                "elements": elements,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/yeouido_scene.npz"),
    )
    parser.add_argument("--snapshot", default=SNAPSHOT_UTC)
    parser.add_argument(
        "--buildings-only",
        action="store_true",
        help=(
            "Rebuild dated building geometry while preserving the shipped "
            "terrain, roads, water mask and authored site details."
        ),
    )
    parser.add_argument(
        "--planimetry-only",
        action="store_true",
        help=(
            "Rebuild dated buildings, bridges, roads and green-space geometry "
            "while preserving the official terrain, event water level/mask "
            "and authored site details."
        ),
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=Path("assets/yeouido_detail_osm_2024-10-05.json"),
    )
    parser.add_argument(
        "--roads-output",
        type=Path,
        default=Path("assets/yeouido_road_osm_2024-10-05.json"),
    )
    parser.add_argument(
        "--roads-only",
        action="store_true",
        help="Download and preserve dated highway ways without rebuilding the scene.",
    )
    parser.add_argument(
        "--building-tags-output",
        type=Path,
        default=Path("assets/yeouido_building_osm_2024-10-05.json"),
    )
    parser.add_argument(
        "--building-tags-only",
        action="store_true",
        help=(
            "Download and preserve dated building tags without geometry and "
            "without rebuilding the scene."
        ),
    )
    args = parser.parse_args()
    if args.building_tags_only:
        buildings = download_building_tags(args.snapshot)
        write_building_tag_snapshot(
            buildings, args.building_tags_output, args.snapshot
        )
        print(
            f"saved {args.building_tags_output}: "
            f"{len(buildings.get('elements', [])):,} dated building ways"
        )
        return
    if args.roads_only:
        roads = download_roads(args.snapshot)
        write_road_source_snapshot(roads, args.roads_output, args.snapshot)
        print(
            f"saved {args.roads_output}: "
            f"{len(roads.get('elements', [])):,} dated highway ways"
        )
        return
    osm = download(args.snapshot)
    scene = build_scene(osm, *ORIGIN, snapshot_utc=args.snapshot)
    if args.buildings_only or args.planimetry_only:
        if not args.output.exists():
            raise FileNotFoundError(
                "partial rebuild requires an existing scene asset"
            )
        previous = load_scene(args.output)
        replacements = {
            "building_vertices": scene.building_vertices,
            "snapshot_utc": args.snapshot,
        }
        if args.planimetry_only:
            replacements.update(
                bridge_vertices=scene.bridge_vertices,
                road_vertices=scene.road_vertices,
                vegetation_vertices=scene.vegetation_vertices,
            )
        rebuilt = replace(previous, **replacements)
        save_scene(rebuilt, args.output)
        scope = "planimetry" if args.planimetry_only else "building"
        print(
            f"saved {args.output}: "
            f"rebuilt dated {scope} arrays; preserved official terrain, "
            "event water and authored details"
        )
        return
    detail_elements = [
        element
        for element in osm.get("elements", [])
        if element.get("tags", {}).get("leisure")
        in {"playground", "pitch", "track", "garden"}
        or element.get("tags", {}).get("landuse")
        in {"grass", "meadow", "flowerbed", "forest", "recreation_ground"}
        or element.get("tags", {}).get("natural")
        in {"wood", "grassland", "scrub"}
    ]
    args.details_output.write_text(
        json.dumps(
            {
                "version": osm.get("version", 0.6),
                "generator": osm.get("generator", "Overpass API"),
                "snapshot_utc": args.snapshot,
                "elements": detail_elements,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    mask, bounds = build_water_mask(
        download_han_river(args.snapshot), *ORIGIN
    )
    terrain, datum_m = build_terrain_heightmap(*ORIGIN, tuple(bounds), mask)
    scene = replace(
        scene,
        water_mask=mask,
        water_mask_bounds=bounds,
        terrain_height_m=terrain,
        terrain_bounds=bounds,
        elevation_datum_m=datum_m,
    )
    save_scene(scene, args.output)
    print(
        f"saved {args.output}: "
        f"{len(scene.building_vertices):,} building vertices, "
        f"{len(scene.bridge_vertices):,} bridge vertices, "
        f"{len(scene.road_vertices):,} road vertices, "
        f"{len(scene.vegetation_vertices):,} green-space vertices, "
        f"elevation datum {scene.elevation_datum_m:.2f} m"
    )



if __name__ == "__main__":
    main()
