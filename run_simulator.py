import argparse

from simulator.app import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yeouido fireworks simulator")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="render ten frames and exit (used for installation verification)",
    )
    args = parser.parse_args()
    run(max_frames=10 if args.smoke_test else None)
