"""Run the 100-scenario validation benchmark required for theoretical VSS."""

import argparse

from src.core.constants import DEFAULT_SCENARIO_VERSION, REGIMES, TypeOfFlexibility
from src.optimization.experiments.validation_benchmark import run_all_validation_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--flexibilities", nargs="+", choices=[x.value for x in TypeOfFlexibility], default=[x.value for x in TypeOfFlexibility])
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = run_all_validation_benchmarks(args.version, args.regimes, args.flexibilities, args.time_limit, args.overwrite)
    print(f"Persisted {len(results)} validation RP benchmarks.")


if __name__ == "__main__":
    main()
