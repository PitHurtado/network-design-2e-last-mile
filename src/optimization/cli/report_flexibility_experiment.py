"""Build the comparative flexibility report from persisted experiment results."""

import argparse

from src.core.constants import DEFAULT_SCENARIO_VERSION
from src.optimization.reports import build_flexibility_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION)
    args = parser.parse_args()
    path = build_flexibility_report(args.version)
    print(f"Reporte: {path}")


if __name__ == "__main__":
    main()
