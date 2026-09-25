"""Generate and report paired independent and spatial demand scenarios.

Example:
    poetry run python -m src.pipeline.cli.compare --all --version v4 --n 100
"""

import argparse
import json
from pathlib import Path

from src.core.constants import DEFAULT_SCENARIO_VERSION, PATH_SHAPE_PARAMS, REGIMES, SEED_BASE, comparison_dir
from src.pipeline.generate import (
    ScenarioGenerator,
    generate_comparison_set,
    historical_bootstrap_shocks,
    shape_params_digest,
    write_comparison_manifest,
)
from src.pipeline.reports.comparison import METHODS, build_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, help="one regime to compare")
    parser.add_argument("--all", action="store_true", help="compare every regime")
    parser.add_argument("--version", default=DEFAULT_SCENARIO_VERSION)
    parser.add_argument("--n", type=int, default=100, help="paired validation scenarios per method")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", default=None, help="HTML output path for a single regime")
    args = parser.parse_args()

    if not args.all and not args.regime:
        parser.error("pass --regime <name> or --all")
    if args.n < 2:
        parser.error("--n must be at least 2")
    if not PATH_SHAPE_PARAMS.exists():
        parser.error(f"{PATH_SHAPE_PARAMS} not found; run fit_params first")

    with open(PATH_SHAPE_PARAMS) as file:
        params = json.load(file)
    regimes = REGIMES if args.all else (args.regime,)
    digest = shape_params_digest(params)

    for regime in regimes:
        for method in METHODS:
            directory = comparison_dir(args.version, regime, method, "validation")
            if directory.exists() and any(directory.iterdir()) and not args.overwrite:
                parser.error(f"{directory} already exists; choose a new --version or pass --overwrite")
            if directory.exists() and args.overwrite:
                for path in directory.glob("scenario_*.json"):
                    path.unlink()
                manifest = directory / "manifest.json"
                if manifest.exists():
                    manifest.unlink()

            generator = ScenarioGenerator.from_params(params, dependence_mode=method)
            if method == "historical_bootstrap":
                from src.pipeline.demand_panel import load_panel

                stop_shocks, drop_shocks = historical_bootstrap_shocks(
                    load_panel(), generator.pixels, generator.expected_stop, generator.expected_drop
                )
                generator.set_bootstrap_shocks(stop_shocks, drop_shocks)
            summary = generate_comparison_set(
                generator,
                regime,
                params["regimes"][regime]["multiplier"],
                method,
                args.n,
                version=args.version,
                seed_base=args.seed_base,
                params_sha256=digest,
            )
            write_comparison_manifest(regime, args.version, method, "validation", summary)

        output = Path(args.output) if args.output and len(regimes) == 1 else None
        report = build_comparison_report(regime, version=args.version, output_path=output)
        print(f"[{regime}] Report: {report}")


if __name__ == "__main__":
    main()
