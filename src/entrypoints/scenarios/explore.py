"""Build the interactive scenario explorer (grid by regime/layer + variability stats).

poetry run python -m src.entrypoints.scenarios.explore
poetry run python -m src.entrypoints.scenarios.explore --regimes normal high
"""

import argparse
from pathlib import Path

from src.analysis.explore_scenarios import build_report
from src.constants import REGIMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--output", default=None, help="Output HTML path (default: results/explore_scenarios.html).")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    path = build_report(args.regimes, output_path)
    print(f"\nReporte: {path}")
    print(f"Abrir con: open {path}")


if __name__ == "__main__":
    main()
