"""Generate versioned demand scenario sets from fitted shape parameters.

poetry run python -m src.scenarios.cli.generate --all --version v3
"""

import argparse
import json

from src.core.constants import DEFAULT_SCENARIO_VERSION, PATH_SHAPE_PARAMS, REGIMES, SEED_BASE
from src.core.contract import ScenarioLayout
from src.scenarios.generation.generator import ScenarioGenerator, generate_set, write_manifest
from src.tools.io import sha256_json
from src.tools.logging import get_logger

logger = get_logger("GenerateScenarios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, help="demand regime to generate")
    parser.add_argument("--all", action="store_true", help="generate every regime")
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION, help="immutable scenario version label")
    parser.add_argument("--optimization-n", type=int, default=30, help="simulated scenarios for optimization")
    parser.add_argument("--validation-n", type=int, default=100, help="simulated scenarios for validation")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing version/set explicitly")
    args = parser.parse_args()

    if not args.all and not args.regime:
        parser.error("pass --regime <name> or --all")

    if not PATH_SHAPE_PARAMS.exists():
        parser.error(f"{PATH_SHAPE_PARAMS} not found — run src.scenarios.cli.fit_params first")

    with open(PATH_SHAPE_PARAMS) as file:
        params = json.load(file)

    generator = ScenarioGenerator.from_params(params)
    regimes = REGIMES if args.all else (args.regime,)

    digest = sha256_json(params)
    for regime in regimes:
        multiplier = params["regimes"][regime]["multiplier"]
        for scenario_set, count in (
            ("optimization", args.optimization_n),
            ("validation", args.validation_n),
            ("expected", 1),
            ("annual_expected", 1),
        ):
            directory = ScenarioLayout.generated(args.version).set_dir(regime, scenario_set)
            if directory.exists() and any(directory.iterdir()) and not args.overwrite:
                parser.error(f"{directory} already exists; choose a new --version or pass --overwrite")
            summary = generate_set(generator, regime, multiplier, scenario_set, count, args.version, args.seed_base, digest)
            write_manifest(
                regime,
                args.version,
                scenario_set,
                {
                    **summary,
                    "shape_params_version": params["version"],
                    "shape_params_generated_on": params["generated_on"],
                    "target_period_demand": params["regimes"][regime]["target_period_demand"],
                },
            )

    print("\nResumen:")
    for regime in regimes:
        print(
            f"  {regime:8s} optimization={args.optimization_n:3d} validation={args.validation_n:3d} expected=1 annual_expected=1"
        )


if __name__ == "__main__":
    main()
