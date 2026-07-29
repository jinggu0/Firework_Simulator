from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

from simulator.scene import build_scene, save_scene

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
    save_scene(scene, args.output)
    print(
        f"saved {args.output}: "
        f"{len(scene.building_vertices):,} building vertices, "
        f"{len(scene.bridge_vertices):,} bridge vertices"
    )


if __name__ == "__main__":
    main()

