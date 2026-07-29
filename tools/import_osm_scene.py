from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import urllib.parse
import urllib.request

from simulator.scene import build_scene, build_water_mask, save_scene

BBOX = (37.515, 126.910, 37.545, 126.960)
ORIGIN = (37.529, 126.935)


def download() -> dict:
    south, west, north, east = BBOX
    query = f"""
[out:json][timeout:90];
(
  way["building"]({south},{west},{north},{east});
  way["bridge"]({south},{west},{north},{east});
);
out geom;
"""
    request = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": "FireworkSimulator/0.1 (local research project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def download_han_river() -> dict:
    request = urllib.request.Request(
        "https://api.openstreetmap.org/api/0.6/relation/152336/full.json",
        headers={"User-Agent": "FireworkSimulator/0.1 (local research project)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/yeouido_scene.npz"),
    )
    args = parser.parse_args()
    osm = download()
    scene = build_scene(osm, *ORIGIN)
    mask, bounds = build_water_mask(download_han_river(), *ORIGIN)
    scene = replace(scene, water_mask=mask, water_mask_bounds=bounds)
    save_scene(scene, args.output)
    print(
        f"saved {args.output}: "
        f"{len(scene.building_vertices):,} building vertices, "
        f"{len(scene.bridge_vertices):,} bridge vertices"
    )


if __name__ == "__main__":
    main()
