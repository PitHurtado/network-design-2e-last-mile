"""Build the validation report for the generated demand scenarios.

    poetry run python -m src.pipeline.cli.analyze
    poetry run python -m src.pipeline.cli.analyze --regimes normal

Exits non-zero if any contract invariant fails, so it can gate a run.
"""

import argparse
import sys

from src.core.constants import REGIMES
from src.core.logging import get_logger
from src.pipeline.reports.scenarios import build_report

logger = get_logger("AnalyzeScenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    args = parser.parse_args()

    path, n_failed = build_report(args.regimes)
    print(f"\nReporte: {path}")
    print(f"Abrir con: open {path}")

    if n_failed:
        logger.error(f"{n_failed} chequeos del contrato fallaron.")
        sys.exit(1)
    print("Todos los invariantes del contrato pasan.")


if __name__ == "__main__":
    main()
