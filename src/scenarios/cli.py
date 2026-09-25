"""`scenarios`: build the panel, fit parameters, generate and compare scenario versions.

    scenarios panel build [--dry-run]
    scenarios params fit [--n 100]                        -> cp-* candidate
    scenarios params recalibrate --from p1 [--validation-n 100]
    scenarios generate --params p1                        -> cv-* candidate
    scenarios compare --params p1 [--regimes normal] [--n 100]   -> cc-* (exploratory)
    scenarios explore <v1|cv-*>
    scenarios validate <ref>        checks + reports/validation.html (scenarios)
    scenarios promote <ref>         re-executes, compares bytes, renames to p<N> / v<N>
    scenarios list | show <ref>

Every command writes a disposable candidate; only `promote` creates an official version.
"""

import argparse
import sys

from src.core.constants import REGIME_TARGETS, SEED_BASE
from src.core.contract import ScenarioLayout
from src.scenarios.stages import (
    CompareConfig,
    ComparisonStage,
    FitConfig,
    GenerateConfig,
    ParamsStage,
    ParamsValidator,
    ScenarioStage,
    ScenarioValidator,
)
from src.tools import cli
from src.tools.artifacts import ArtifactKind, ArtifactStore
from src.tools.manifest import Manifest

KINDS = [ArtifactKind.PARAMS, ArtifactKind.SCENARIOS, ArtifactKind.COMPARISONS]


def _panel_build(args) -> int:
    from src.scenarios.fitting.panel import build_panel, save_panel

    panel = build_panel()
    included = panel[~panel["excluded"]]
    print("\nDemanda por año:")
    print(included.groupby("year")["items"].sum().round(0).to_string())
    per_pixel = included.groupby("id_pixel")["n_customers"].mean()
    print("\nClientes promedio por período, por píxel:")
    print(
        f"  min={per_pixel.min():.2f}  p05={per_pixel.quantile(0.05):.2f}  "
        f"mediana={per_pixel.median():.2f}  max={per_pixel.max():.2f}"
    )
    print(f"  píxeles bajo 5 clientes promedio: {int((per_pixel < 5).sum())} de {len(per_pixel)}")
    cw = panel.groupby("id_pixel")["crosswalk_share"].mean()
    print("\nCobertura del crosswalk (share de items con layer conocido):")
    print(f"  media={cw.mean():.3f}  mediana={cw.median():.3f}  píxeles con menos de 50% del crosswalk: {int((cw < 0.5).sum())}")
    if args.dry_run:
        print("--dry-run: panel not written.")
        return 0
    save_panel(panel)
    return 0


def _print_regimes(artifact) -> None:
    details = Manifest.load(artifact.manifest_path).details
    print("\nRegímenes calibrados:")
    for regime, block in details["regimes"].items():
        print(
            f"  {regime:8s} objetivo={REGIME_TARGETS[regime]:>9,.0f} "
            f"multiplicador={block['multiplier']:6.4f}  realizado={block['realized_period_demand']:>9,.0f}"
        )
    print(f"\nCandidata: {artifact.id}  ({artifact.path})")
    print(f"Siguiente paso: scenarios validate {artifact.id}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="scenarios", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    panel = sub.add_parser("panel", help="monthly panel from the raw delivery events").add_subparsers(
        dest="action", required=True
    )
    panel.add_parser("build").add_argument("--dry-run", action="store_true")

    params = sub.add_parser("params", help="shape parameters").add_subparsers(dest="action", required=True)
    fit = params.add_parser("fit", help="fit on the panel and calibrate the regimes")
    fit.add_argument("--n", type=int, default=100, help="validation streams the regime multipliers are calibrated on")
    fit.add_argument("--bins", type=int, default=12, help="distance bins of the correlogram")
    cli.add_seed_base(fit, SEED_BASE)
    recal = params.add_parser("recalibrate", help="new regime multipliers for existing parameters")
    recal.add_argument("--from", dest="parent", required=True, help="params artifact (p1, cp-...)")
    recal.add_argument("--validation-n", type=int, default=100)
    cli.add_seed_base(recal, SEED_BASE)

    gen = sub.add_parser("generate", help="a scenario version from a params artifact")
    gen.add_argument("--params", required=True)
    cli.add_regimes(gen)
    gen.add_argument("--optimization-n", type=int, default=30)
    gen.add_argument("--validation-n", type=int, default=100)
    cli.add_seed_base(gen, SEED_BASE)

    cmp_parser = sub.add_parser("compare", help="paired independent / spatial / bootstrap sets (exploratory)")
    cmp_parser.add_argument("--params", required=True)
    cli.add_regimes(cmp_parser, default=("normal",))
    cmp_parser.add_argument("--n", type=int, default=100)
    cli.add_seed_base(cmp_parser, SEED_BASE)

    sub.add_parser("explore", help="comparative explorer of a scenario version").add_argument("ref")
    cli.add_lifecycle(sub, "params / scenarios")
    args = parser.parse_args(argv)
    store = ArtifactStore()

    def params_handler(args) -> int:
        stage = ParamsStage()
        if args.action == "fit":
            config = FitConfig(n=args.n, bins=args.bins, seed_base=args.seed_base)
            artifact = stage.fit(store, config, argv)
        else:
            parent = store.resolve(args.parent, ArtifactKind.PARAMS)
            artifact = stage.recalibrate(store, parent, args.validation_n, args.seed_base, argv)
        _print_regimes(artifact)
        return 0

    def generate_handler(args) -> int:
        if args.optimization_n < 1 or args.validation_n < 2:
            parser.error("--optimization-n must be >= 1 and --validation-n >= 2")
        config = GenerateConfig(args.params, tuple(args.regimes), args.optimization_n, args.validation_n, args.seed_base)
        artifact = ScenarioStage().generate(store, config, argv)
        print(f"Candidata: {artifact.id}  ({artifact.path})\nSiguiente paso: scenarios validate {artifact.id}")
        return 0

    def compare_handler(args) -> int:
        if args.n < 2:
            parser.error("--n must be at least 2")
        artifact = ComparisonStage().compare(store, CompareConfig(args.params, tuple(args.regimes), args.n, args.seed_base), argv)
        print(f"Comparación: {artifact.id}\nReportes: {artifact.reports_dir}")
        return 0

    def explore_handler(args) -> int:
        from src.scenarios.reports import build_explore_report

        artifact = store.resolve(args.ref, ArtifactKind.SCENARIOS)
        print(f"Reporte: {build_explore_report(ScenarioLayout(artifact.path), artifact.reports_dir / 'explore.html')}")
        return 0

    validators = {ArtifactKind.PARAMS: ParamsValidator(), ArtifactKind.SCENARIOS: ScenarioValidator(store)}
    reproducers = {ArtifactKind.PARAMS: ParamsStage().reproduce(store), ArtifactKind.SCENARIOS: ScenarioStage().reproduce(store)}
    cli.dispatch(
        args,
        {
            "panel": _panel_build,
            "params": params_handler,
            "generate": generate_handler,
            "compare": compare_handler,
            "explore": explore_handler,
            "validate": lambda a: cli.validate(store, a.ref, validators),
            "promote": lambda a: cli.promote(store, a.ref, reproducers),
            "list": lambda a: cli.list_artifacts(store, KINDS),
            "show": lambda a: cli.show(store, a.ref),
        },
    )


if __name__ == "__main__":
    main()
