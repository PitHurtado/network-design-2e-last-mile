"""`optimize`: versioned optimization runs over official scenario versions.

    optimize flexibility --scenarios v1 [--regimes ...] [--flexibilities ...] [--cases ...]   -> cr-*
    optimize evaluate --run r1              fixed-Y recourse on the validation scenarios   -> cr-*
    optimize benchmark --scenarios v1       RP on the validation scenarios (theoretical VSS)  -> cr-*
    optimize report <run> [--benchmark rK]  HTML under <run>/reports/
    optimize verify --scenarios v1 --n 3    CA + Gurobi smoke test
    optimize validate <run> | promote <run> | list | show <run>

A leaf already solved in an official run with identical inputs and solver settings is
copied instead of solved again.
"""

import argparse
import sys

from src.core.constants import TypeOfFlexibility
from src.optimization.experiments.flexibility import CASES
from src.optimization.instance import InstanceBuilder
from src.optimization.stages import ALL_FLEXIBILITIES, RunConfig, RunStage, RunValidator
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
    _add_solver(flex)

    evaluate = sub.add_parser("evaluate", help="evaluate a flexibility run's Y out of sample")
    evaluate.add_argument("--run", required=True)
    _add_solver(evaluate, mip_gap=False)

    bench = sub.add_parser("benchmark", help="validation-sample RP benchmark")
    bench.add_argument("--scenarios", required=True)
    cli.add_regimes(bench)
    bench.add_argument("--flexibilities", nargs="+", choices=flexibilities, default=list(ALL_FLEXIBILITIES))
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
        )
        return announce(stage.flexibility(config, argv, overwrite=args.overwrite))

    def benchmark_handler(args) -> int:
        config = RunConfig(
            args.scenarios, tuple(args.regimes), tuple(args.flexibilities), (), args.time_limit, 0.0, _overrides(args)
        )
        return announce(stage.benchmark(config, argv))

    def report_handler(args) -> int:
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
            "validate": lambda a: cli.validate(store, a.ref, {ArtifactKind.RUNS: RunValidator()}),
            "promote": lambda a: cli.promote(store, a.ref, {ArtifactKind.RUNS: None}),
            "list": lambda a: cli.list_artifacts(store, [ArtifactKind.RUNS]),
            "show": lambda a: cli.show(store, a.ref),
        },
    )


if __name__ == "__main__":
    main()
