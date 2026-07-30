import argparse
from pathlib import Path

from simulator.app import run
from simulator.scenario import DEFAULT_SCENARIO_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yeouido fireworks simulator")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="render ten frames and exit (used for installation verification)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO_PATH,
        help=(
            "Scenario file to reconstruct. The default carries no firing "
            "timeline; assets/scenario_demo_synthetic.json is a synthetic "
            "demonstration sequence, not the 2024 performance."
        ),
    )
    args = parser.parse_args()
    run(
        max_frames=10 if args.smoke_test else None,
        scenario_path=args.scenario,
    )
