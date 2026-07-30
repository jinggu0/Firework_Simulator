"""Run the validation harness and report per-metric residuals.

    python -m tools.validate
    python -m tools.validate --summary
    python -m tools.validate --include-performance --frames 240
    python -m tools.validate --ephemeris assets/reference_ephemeris_2024-10-05.json

Exit code is non-zero only for FAIL or ERROR. NO_REFERENCE and NOT_IMPLEMENTED
are honest states of the project, not regressions, and must not turn a build red.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from simulator.scenario import DEFAULT_SCENARIO_PATH, Scenario
from simulator.validation import MetricStatus, run_validation
from simulator.validation.runner import DEFAULT_EPHEMERIS_PATH

_STATUS_ORDER = (
    MetricStatus.FAIL,
    MetricStatus.ERROR,
    MetricStatus.PASS,
    MetricStatus.REPORTED,
    MetricStatus.NO_REFERENCE,
    MetricStatus.NOT_IMPLEMENTED,
)


def print_summary(report) -> None:
    counts = report.counts()
    print(f"scenario   {report.scenario_id}")
    print(f"generated  {report.generated_at.isoformat()}")
    print(
        "counts     "
        + "  ".join(
            f"{status.value}={counts[status.value]}"
            for status in _STATUS_ORDER
            if counts[status.value]
        )
    )
    print()
    width = max(len(result.spec.title) for result in report)
    for result in report:
        print(
            f"{result.metric_id:<5} {result.status.value:<16} "
            f"{result.spec.title:<{width}}  {result.message}"
        )
        for name, value in result.residuals.items():
            print(f"        {name} = {value:.6g}")
    print()
    if report.failures:
        print("FAILED: " + ", ".join(result.metric_id for result in report.failures))
    else:
        print("No metric failed. Unrunnable metrics are listed above as-is.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the reconstruction validation harness."
    )
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO_PATH)
    parser.add_argument(
        "--ephemeris",
        type=Path,
        default=DEFAULT_EPHEMERIS_PATH,
        help="External ephemeris extract enabling V-01.",
    )
    parser.add_argument(
        "--include-performance",
        action="store_true",
        help="Run V-12, which needs an OpenGL context and a separate process.",
    )
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument(
        "--fluid-backend", choices=("3d", "2d", "cpu"), default="3d"
    )
    parser.add_argument("--replay-steps", type=int, default=400)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a human-readable table instead of JSON.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write the JSON report to a file."
    )
    args = parser.parse_args()

    report = run_validation(
        Scenario.load(args.scenario),
        ephemeris_path=args.ephemeris,
        include_performance=args.include_performance,
        performance_frames=args.frames,
        fluid_backend=args.fluid_backend,
        replay_steps=args.replay_steps,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.summary:
        print_summary(report)
    else:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
