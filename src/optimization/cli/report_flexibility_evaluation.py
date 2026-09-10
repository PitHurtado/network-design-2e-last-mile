"""Build the interactive VSS report from fixed-Y out-of-sample evaluations."""

import argparse

from src.core.constants import DEFAULT_SCENARIO_VERSION
from src.optimization.reports.flexibility_evaluation import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION)
    args = parser.parse_args()
    print(f"Reporte: {build_report(args.version)}")


if __name__ == "__main__":
    main()
