"""Capture the deterministic V2-2c close road-marking verification view."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from simulator.scenario import DEFAULT_SCENARIO_PATH
from simulator.validation.capture import (
    display_sdr_statistics,
    save_display_sdr,
)
from simulator.validation.views import VisualRegressionView
from tools.capture_visual_baselines import capture_view


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "validation"
    / "road_markings_v2"
    / "runtime_current"
)
VIEW = VisualRegressionView(
    view_id="metric_lane_close",
    subject="road_ground",
    position_eus_m=(-880.5568261, 12.0525367, 349.1555049),
    target_eus_m=(-845.5282393, 10.5672874, 320.9066446),
    expected_surface="land",
    minimum_ground_clearance_m=1.6,
    notes=(
        "Engineering close view along an explicit one-way two-lane OSM way; "
        "not a historical camera."
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    _, sdr, context = capture_view(
        VIEW,
        DEFAULT_SCENARIO_PATH,
        args.frames,
        "human_vision",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = save_display_sdr(
        sdr, args.output_dir / "metric_lane_close.sdr.png"
    )
    report = {
        **context,
        "notes": VIEW.notes,
        "scene_asset": "assets/yeouido_scene.npz",
        "scene_asset_sha256": sha256(
            (REPOSITORY_ROOT / "assets" / "yeouido_scene.npz").read_bytes()
        ).hexdigest(),
        "sdr": {
            "asset": image_path.name,
            "sha256": sha256(image_path.read_bytes()).hexdigest(),
            "statistics": display_sdr_statistics(sdr),
        },
    }
    report_path = args.output_dir / "metric_lane_close.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {image_path} and {report_path}")


if __name__ == "__main__":
    main()
