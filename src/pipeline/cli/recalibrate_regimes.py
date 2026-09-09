"""Recalibrate regime multipliers from persisted shape parameters.

Use this when the regime policy changes but the fitted historical marginals and
spatial dependence do not need to be refit:

    python -m src.pipeline.cli.recalibrate_regimes --n 50
"""

import argparse
import json
from datetime import date

import numpy as np

from src.core.constants import PATH_SHAPE_PARAMS, REGIME_DROP_EXPONENT, REGIME_STOP_EXPONENT, REGIME_TARGETS, SEED_BASE
from src.pipeline.generate import ScenarioGenerator
from src.pipeline.regimes import calibrate_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-n", type=int, default=100, help="validation scenarios used for calibration")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = parser.parse_args()

    with open(PATH_SHAPE_PARAMS) as file:
        params = json.load(file)

    params["regime_scaling"] = {
        "stop_exponent": REGIME_STOP_EXPONENT,
        "drop_exponent": REGIME_DROP_EXPONENT,
    }
    generator = ScenarioGenerator.from_params(params)
    regimes = calibrate_all(
        total_for=lambda multiplier, rng: generator.mean_period_total(rng, multiplier),
        base_total=generator.base_period_total(),
        seed_base=args.seed_base,
        n_scenarios=args.validation_n,
        seeds=np.random.SeedSequence([args.seed_base, 100]).spawn(args.validation_n),
    )
    params["version"] = max(int(params.get("version", 0)), 3)
    params["generated_on"] = date.today().isoformat()
    params["seed_base"] = args.seed_base
    params["calibrated_n_scenarios"] = args.validation_n
    params["regimes"] = {
        regime: {
            "target_period_demand": REGIME_TARGETS[regime],
            "multiplier": result["multiplier"],
            "realized_period_demand": result["realized"],
        }
        for regime, result in regimes.items()
    }
    with open(PATH_SHAPE_PARAMS, "w") as file:
        json.dump(params, file, indent=2)

    print("Regímenes recalibrados:")
    for regime, result in regimes.items():
        print(f"  {regime:8s} multiplicador={result['multiplier']:.4f} realizado={result['realized']:,.0f}")


if __name__ == "__main__":
    main()
