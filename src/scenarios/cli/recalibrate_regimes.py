"""Recalibrate regime multipliers from persisted shape parameters.

Use this when the regime policy changes but the fitted historical marginals and
spatial dependence do not need to be refit:

    python -m src.scenarios.cli.recalibrate_regimes --validation-n 100
"""

import argparse

from src.core.constants import PATH_SHAPE_PARAMS, SEED_BASE
from src.scenarios.fitting.fitter import ParamsFitter
from src.scenarios.params import ShapeParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-n", type=int, default=100, help="validation scenarios used for calibration")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = parser.parse_args()

    params, regimes = ParamsFitter(seed_base=args.seed_base).recalibrate(ShapeParams.load(PATH_SHAPE_PARAMS), args.validation_n)
    params.save(PATH_SHAPE_PARAMS)

    print("Regímenes recalibrados:")
    for regime, result in regimes.items():
        print(f"  {regime:8s} multiplicador={result['multiplier']:.4f} realizado={result['realized']:,.0f}")


if __name__ == "__main__":
    main()
