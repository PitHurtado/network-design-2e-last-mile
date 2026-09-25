"""Evaluate annual_expected, expected and optimization solutions on validation scenarios."""

import argparse

from src.core.constants import DEFAULT_SCENARIO_VERSION, REGIMES, TypeOfFlexibility
from src.optimization.experiments.flexibility_evaluation import SOLUTION_CASES, evaluate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument(
        "--flexibilities", nargs="+", choices=[x.value for x in TypeOfFlexibility], default=[x.value for x in TypeOfFlexibility]
    )
    parser.add_argument("--solution-cases", nargs="+", choices=SOLUTION_CASES, default=list(SOLUTION_CASES))
    parser.add_argument("--time-limit", type=float, default=600.0, help="seconds per fixed-Y recourse evaluation")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = evaluate_experiment(
        args.version, args.regimes, args.flexibilities, args.solution_cases, args.time_limit, args.overwrite
    )
    print(f"Persisted {len(results)} out-of-sample evaluations.")


if __name__ == "__main__":
    main()
