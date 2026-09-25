"""Build the validation report for the generated demand scenarios.

    poetry run python -m src.scenarios.cli.analyze --version v3
    poetry run python -m src.scenarios.cli.analyze --version v3 --regimes normal

Exits non-zero if any contract invariant fails, so it can gate a run.
"""

import argparse
import sys

from src.core.constants import DEFAULT_SCENARIO_VERSION, REGIMES
from src.scenarios.reports import build_validation_report
from src.tools.logging import get_logger

logger = get_logger("AnalyzeScenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION, help="scenario version whose validation set is analyzed")
    args = parser.parse_args()

    path, n_failed = build_validation_report(args.regimes, version=args.version)
    print(f"\nReporte: {path}")
    print(f"Abrir con: open {path}")

    if n_failed:
        logger.error(f"{n_failed} chequeos del contrato fallaron.")
        sys.exit(1)
    print("Todos los invariantes del contrato pasan.")


if __name__ == "__main__":
    main()
