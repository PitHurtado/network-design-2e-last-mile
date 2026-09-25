"""Generate and report paired independent and spatial demand scenarios.

Example:
    poetry run python -m src.scenarios.cli.compare --all --version v4 --n 100
"""

import argparse
from pathlib import Path

from src.core.constants import DEFAULT_SCENARIO_VERSION, PATH_SHAPE_PARAMS, REGIMES, SEED_BASE
from src.core.contract import ScenarioLayout
from src.scenarios.fitting.panel import load_panel
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.generation.sets import ScenarioSetWriter, SetSpec
from src.scenarios.params import ShapeParams
from src.scenarios.reports.comparison import METHODS, build_comparison_report


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

    params = ShapeParams.load(PATH_SHAPE_PARAMS)
    regimes = REGIMES if args.all else (args.regime,)
    digest = params.sha256
    writer = ScenarioSetWriter(ScenarioLayout.for_comparison(args.version))

    for regime in regimes:
        for method in METHODS:
            directory = writer.layout.set_dir(regime, "validation", method)
            if directory.exists() and any(directory.iterdir()) and not args.overwrite:
                parser.error(f"{directory} already exists; choose a new --version or pass --overwrite")
            if directory.exists() and args.overwrite:
                for path in directory.glob("scenario_*.json"):
                    path.unlink()
                manifest = directory / "manifest.json"
                if manifest.exists():
                    manifest.unlink()

            panel = load_panel() if method == "historical_bootstrap" else None
            generator = ScenarioGenerator.from_params(params, method=method, panel=panel)
            multiplier = params["regimes"][regime]["multiplier"]
            spec = SetSpec(regime, "validation", args.n, multiplier, args.version, args.seed_base, method=method)
            summary = writer.generate(generator, spec, params_sha256=digest)
            writer.write_manifest(regime, "validation", summary, method=method)

        output = Path(args.output) if args.output and len(regimes) == 1 else None
        report = build_comparison_report(regime, version=args.version, output_path=output)
        print(f"[{regime}] Report: {report}")


if __name__ == "__main__":
    main()
