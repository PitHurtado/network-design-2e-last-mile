"""Build the interactive VSS report from fixed-Y out-of-sample evaluations."""

import argparse

from src.core.constants import DEFAULT_SCENARIO_VERSION
from src.optimization.reports import build_evaluation_preview, build_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION)
    parser.add_argument("--partial", action="store_true", help="build a progress preview without requiring all 27 evaluations")
    args = parser.parse_args()
    builder = build_evaluation_preview if args.partial else build_evaluation_report
    print(f"Reporte: {builder(args.version)}")


if __name__ == "__main__":
    main()
