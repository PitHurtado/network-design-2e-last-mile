"""Run all flexibility configurations over expected, optimization and annual expected.

poetry run python -m src.optimization.cli.run_flexibility_experiment --version v3
"""

import argparse

from src.core.constants import DEFAULT_SCENARIO_VERSION, REGIMES, TypeOfFlexibility
from src.optimization.experiments.flexibility import CASES, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION, help="scenario and results version")
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument(
        "--flexibilities",
        nargs="+",
        choices=[item.value for item in TypeOfFlexibility],
        default=[item.value for item in TypeOfFlexibility],
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--time-limit", type=float, default=600.0, help="seconds per solve")
    parser.add_argument("--mip-gap", type=float, default=0.0, help="relative MIP gap; keep 0 for valid policy comparisons")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = run_experiment(
        args.version, args.regimes, args.flexibilities, args.cases, args.time_limit, args.mip_gap, args.overwrite
    )
    errors = [result for result in results if result.get("status") == "ERROR"]
    print(f"Persisted {len(results)} runs; errors: {len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
