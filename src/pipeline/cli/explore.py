"""Build the comparative explorer for low, normal, and high demand regimes.

poetry run python -m src.pipeline.cli.explore --version v3
"""

import argparse
from pathlib import Path

from src.core.constants import DEFAULT_SCENARIO_VERSION, REGIMES
from src.pipeline.reports.explore import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=REGIMES,
        default=list(REGIMES),
        help="must include low, normal, and high for the comparative report",
    )
    parser.add_argument("--output", default=None, help="output HTML path (default: versioned results directory)")
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION, help="scenario version whose validation set is explored")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    path = build_report(args.regimes, output_path, version=args.version)
    print(f"\nReporte: {path}")
    print(f"Abrir con: open {path}")


if __name__ == "__main__":
    main()
