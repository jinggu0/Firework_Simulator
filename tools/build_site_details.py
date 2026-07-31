from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from simulator.scenario import DEFAULT_SCENARIO_PATH, Scenario
from simulator.scene import load_scene, save_scene
from simulator.site_details import (
    build_site_detail_mesh,
    classify_path_surfaces,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("assets/yeouido_scene.npz"),
    )
    parser.add_argument(
        "--historical-details",
        type=Path,
        default=Path("assets/yeouido_detail_osm_2024-10-05.json"),
    )
    parser.add_argument(
        "--official-facilities",
        type=Path,
        default=Path("assets/yeouido_official_facilities.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help=(
            "Scenario whose observers the grass-blade budget is spent around. "
            "Without one the budget falls back to the scene origin."
        ),
    )
    args = parser.parse_args()
    scene = load_scene(args.scene)
    observers = None
    if args.scenario is not None and args.scenario.exists():
        scenario = Scenario.load(args.scenario)
        observers = np.array(
            [
                scenario.observer_position_eus_m(observer.observer_id)[[0, 2]]
                for observer in scenario.observers
            ],
            dtype=np.float64,
        )
    historical = json.loads(
        args.historical_details.read_text(encoding="utf-8")
    )
    facilities = json.loads(
        args.official_facilities.read_text(encoding="utf-8")
    )
    detail_vertices, counts = build_site_detail_mesh(
        scene, historical, facilities, observers
    )
    detailed_scene = replace(
        scene,
        road_vertices=classify_path_surfaces(scene.road_vertices),
        detail_vertices=detail_vertices,
    )
    output = args.output or args.scene
    save_scene(detailed_scene, output)
    print(
        f"saved {output}: {len(detail_vertices):,} detail vertices; "
        + ", ".join(f"{key}={value:,}" for key, value in counts.items())
    )


if __name__ == "__main__":
    main()
