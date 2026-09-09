"""Generate demand scenarios for a regime from the fitted shape parameters.

poetry run python -m src.pipeline.cli.generate --regime normal --n 50
poetry run python -m src.pipeline.cli.generate --all --n 50
"""

import argparse
import json

from src.core.constants import PATH_SHAPE_PARAMS, REGIMES, SEED_BASE
from src.core.logging import get_logger
from src.pipeline.generate import ScenarioGenerator, generate_regime, write_manifest

logger = get_logger("GenerateScenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, help="demand regime to generate")
    parser.add_argument("--all", action="store_true", help="generate every regime")
    parser.add_argument("--n", type=int, default=50, help="number of simulated scenarios per regime")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = parser.parse_args()

    if not args.all and not args.regime:
        parser.error("pass --regime <name> or --all")

    if not PATH_SHAPE_PARAMS.exists():
        parser.error(f"{PATH_SHAPE_PARAMS} not found — run src.pipeline.cli.fit_params first")

    with open(PATH_SHAPE_PARAMS) as file:
        params = json.load(file)

    generator = ScenarioGenerator.from_params(params)
    regimes = REGIMES if args.all else (args.regime,)

    for regime in regimes:
        multiplier = params["regimes"][regime]["multiplier"]
        summary = generate_regime(
            generator=generator,
            regime=regime,
            multiplier=multiplier,
            n_scenarios=args.n,
            seed_base=args.seed_base,
        )
        write_manifest(
            regime,
            {
                **summary,
                "shape_params_version": params["version"],
                "shape_params_generated_on": params["generated_on"],
                "target_period_demand": params["regimes"][regime]["target_period_demand"],
                "spatial": {k: params["spatial"][k] for k in ("model", "rho_km", "nugget", "plateau", "weighted_r2")},
            },
        )

    print("\nResumen:")
    for regime in regimes:
        with open(f"data/scenarios/generated/{regime}/manifest.json") as file:
            manifest = json.load(file)
        print(
            f"  {regime:8s} n={manifest['n_scenarios']:3d} "
            f"objetivo={manifest['target_period_demand']:>9,.0f} "
            f"realizado={manifest['period_total_mean']:>9,.0f} "
            f"p10={manifest['period_total_p10']:>9,.0f} p90={manifest['period_total_p90']:>9,.0f} "
            f"| piso de stop {manifest['stop_floor_share'] * 100:.3f}%"
        )


if __name__ == "__main__":
    main()
