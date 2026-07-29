from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import urllib.parse
import urllib.request

from simulator.scene import build_scene, build_water_mask, save_scene
from simulator.terrain import build_terrain_heightmap

BBOX = (37.515, 126.910, 37.545, 126.960)
ORIGIN = (37.529, 126.935)
SNAPSHOT_UTC = "2024-10-05T10:20:00Z"


def _overpass(query: str) -> dict:
    last_error: Exception | None = None
    for endpoint in (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ):
        request = urllib.request.Request(
            endpoint,
            data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": "FireworkSimulator/0.1 (local research project)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except Exception as error:
            last_error = error
    raise RuntimeError("all configured Overpass endpoints failed") from last_error


def download(snapshot_utc: str = SNAPSHOT_UTC) -> dict:
    south, west, north, east = BBOX
    query = f"""
[out:json][timeout:240][date:"{snapshot_utc}"];
(
  way["building"]({south},{west},{north},{east});
  way["bridge"]({south},{west},{north},{east});
  way["highway"]({south},{west},{north},{east});
  way["leisure"="park"]({south},{west},{north},{east});
  way["landuse"~"^(grass|forest|recreation_ground)$"]({south},{west},{north},{east});
  way["natural"~"^(wood|grassland)$"]({south},{west},{north},{east});
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/yeouido_scene.npz"),
    )
    parser.add_argument("--snapshot", default=SNAPSHOT_UTC)
    args = parser.parse_args()
    osm = download(args.snapshot)
    scene = build_scene(osm, *ORIGIN, snapshot_utc=args.snapshot)
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
