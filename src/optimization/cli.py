"""`optimize`: satellite capacity tables and versioned optimization runs.

    optimize capacity analyze --scenarios v1 [--levels percentiles-a|percentiles-b|fixed-grid]  -> cf-*
    optimize flexibility --scenarios v1 --facilities f1 [--model flex|capacitated] [--regimes ...]   -> cr-*
    optimize evaluate --run r1              fixed-Y recourse on the validation scenarios   -> cr-*
    optimize benchmark --scenarios v1 --facilities f1   RP on the validation scenarios (theoretical VSS)  -> cr-*
    optimize report <run> [--benchmark rK]  HTML under <run>/reports/
    optimize verify --scenarios v1 --n 3    CA + Gurobi smoke test
    optimize validate <ref> | promote <ref> | list | show <ref>      (ref: cf-*/f<N> or cr-*/r<N>)

Capacitated models (capacitated, flex) only run with --facilities: the promoted capacity
table is where their satellite levels and costs come from.

A leaf already solved in an official run with identical inputs and solver settings is
copied instead of solved again.
"""

import argparse
import sys

from src.core.constants import TypeOfFlexibility
from src.optimization.capacity.analysis import CapacityConfig
from src.optimization.capacity.levels import LEVEL_METHODS
from src.optimization.experiments.flexibility import CASES
from src.optimization.instance import InstanceBuilder
from src.optimization.models import MODELS
from src.optimization.stages import ALL_FLEXIBILITIES, FacilityStage, FacilityValidator, RunConfig, RunStage, RunValidator
from src.optimization.verify import verify
from src.tools import cli
from src.tools.artifacts import ArtifactKind, ArtifactStore


def _add_solver(parser: argparse.ArgumentParser, mip_gap: bool = True) -> None:
    parser.add_argument("--time-limit", type=float, default=600.0, help="seconds per solve")
    if mip_gap:
        parser.add_argument("--mip-gap", type=float, default=0.0, help="relative MIP gap; keep 0 for valid policy comparisons")
    parser.add_argument("--threads", type=int, default=None, help="Gurobi Threads (pin with --seed for reproducible solves)")
    parser.add_argument("--seed", type=int, default=None, help="Gurobi Seed")


def _overrides(args) -> dict:
    return {key: value for key, value in (("Threads", args.threads), ("Seed", args.seed)) if value is not None}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="optimize", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    flexibilities = [item.value for item in TypeOfFlexibility]

    flex = sub.add_parser("flexibility", help="regime × policy × case experiment")
    flex.add_argument("--scenarios", required=True, help="scenario version (v1, cv-..., latest)")
    cli.add_regimes(flex)
    flex.add_argument("--flexibilities", nargs="+", choices=flexibilities, default=list(ALL_FLEXIBILITIES))
    flex.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    flex.add_argument("--overwrite", action="store_true", help="solve every leaf even if an official run already has it")
    flex.add_argument("--model", choices=sorted(MODELS), default="flex", help="model variant (policies apply to flex only)")
    flex.add_argument("--facilities", default=None, help="capacity table (f1, cf-...); required by capacitated models")
    _add_solver(flex)

    evaluate = sub.add_parser("evaluate", help="evaluate a flexibility run's Y out of sample")
    evaluate.add_argument("--run", required=True)
    _add_solver(evaluate, mip_gap=False)

    bench = sub.add_parser("benchmark", help="validation-sample RP benchmark")
    bench.add_argument("--scenarios", required=True)
    cli.add_regimes(bench)
    bench.add_argument("--flexibilities", nargs="+", choices=flexibilities, default=list(ALL_FLEXIBILITIES))
    bench.add_argument("--model", choices=sorted(MODELS), default="flex")
    bench.add_argument("--facilities", default=None)

    capacity = sub.add_parser("capacity", help="satellite capacity table").add_subparsers(dest="action", required=True)
    analyze = capacity.add_parser("analyze", help="peak fleet per satellite -> levels and costs")
    analyze.add_argument("--scenarios", required=True, help="scenario version with a capacity set (v1, cv-...)")
    analyze.add_argument("--levels", choices=sorted(LEVEL_METHODS), default="percentiles-a")
    analyze.add_argument("--alpha", type=float, default=None, help="fixed share of OPEX (default: the tariff table's)")
    cli.add_regimes(analyze)
    _add_solver(bench, mip_gap=False)

    report = sub.add_parser("report", help="HTML reports of a run")
    report.add_argument("ref")
    report.add_argument("--benchmark", default=None, help="benchmark run for the theoretical VSS section")

    check = sub.add_parser("verify", help="CA + Gurobi smoke test on a scenario version")
    check.add_argument("--scenarios", default="latest")
    check.add_argument("--n", type=int, default=3, help="scenarios per regime (keep small: this is a smoke test)")
    check.add_argument("--max-run-time", type=float, default=120.0)
    cli.add_regimes(check)
    cli.add_lifecycle(sub, "run")
    args = parser.parse_args(argv)
    store = ArtifactStore()
    stage = RunStage(store)
    facilities = FacilityStage(store)

    def announce(artifact) -> int:
        print(
            f"Corrida: {artifact.id}  ({artifact.path})\nSiguientes pasos: optimize report {artifact.id} · optimize validate {artifact.id}"
        )
        return 0

    def flexibility_handler(args) -> int:
        config = RunConfig(
            args.scenarios,
            tuple(args.regimes),
            tuple(args.flexibilities),
            tuple(args.cases),
            args.time_limit,
            args.mip_gap,
            _overrides(args),
            args.model,
            args.facilities,
        )
        return announce(stage.flexibility(config, argv, overwrite=args.overwrite))

    def benchmark_handler(args) -> int:
        config = RunConfig(
            args.scenarios,
            tuple(args.regimes),
            tuple(args.flexibilities),
            (),
            args.time_limit,
            0.0,
            _overrides(args),
            args.model,
            args.facilities,
        )
        return announce(stage.benchmark(config, argv))

    def capacity_handler(args) -> int:
        config = CapacityConfig(args.scenarios, args.levels, tuple(args.regimes), args.alpha)
        artifact = facilities.analyze(config, argv)
        print(f"Tabla de capacidad: {artifact.id}  ({artifact.path})\nSiguiente paso: optimize validate {artifact.id}")
        return 0

    def report_handler(args) -> int:
        if ArtifactKind.of(args.ref) is ArtifactKind.FACILITIES:
            print(f"Reporte: {facilities.report(store.resolve(args.ref, ArtifactKind.FACILITIES))}")
            return 0
        benchmark = store.resolve(args.benchmark, ArtifactKind.RUNS) if args.benchmark else None
        for path in stage.report(store.resolve(args.ref, ArtifactKind.RUNS), benchmark):
            print(f"Reporte: {path}")
        return 0

    cli.dispatch(
        args,
        {
            "flexibility": flexibility_handler,
            "evaluate": lambda a: announce(stage.evaluate(a.run, a.time_limit, argv, _overrides(a))),
            "benchmark": benchmark_handler,
            "report": report_handler,
            "verify": lambda a: verify(InstanceBuilder.for_version(a.scenarios, store), a.n, a.max_run_time, a.regimes),
            "capacity": capacity_handler,
            "validate": lambda a: cli.validate(
                store, a.ref, {ArtifactKind.RUNS: RunValidator(), ArtifactKind.FACILITIES: FacilityValidator(facilities)}
            ),
            "promote": lambda a: cli.promote(
                store, a.ref, {ArtifactKind.RUNS: None, ArtifactKind.FACILITIES: facilities.reproduce()}
            ),
            "list": lambda a: cli.list_artifacts(store, [ArtifactKind.FACILITIES, ArtifactKind.RUNS]),
            "show": lambda a: cli.show(store, a.ref),
        },
    )


if __name__ == "__main__":
    main()
