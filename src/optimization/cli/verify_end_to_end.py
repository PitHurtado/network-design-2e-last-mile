"""End-to-end check: do the generated scenarios feed the CA and Gurobi unchanged?

    poetry run python -m src.optimization.cli.verify_end_to_end --n 3

This is the acceptance test for the scenario contract. The failure modes it targets
are silent-to-read but fatal at solve time:

* the CA only writes cost keys for `demand > 0`, and `BaseSAAModel._obj_routing_facilities`
  indexes them directly, so one zero pixel-period is a `KeyError`;
* `drop` is a divisor and `density` sits under a `sqrt` in a divisor, so a zero
  there is a `ZeroDivisionError` before the model is even built;
* the objective divides by the number of scenarios, so an empty scenario list is a
  division by zero.

It also checks that the objective orders `low < normal < high`, which it must if the
regimes mean anything.
"""

import argparse
import sys

from src.core.constants import REGIMES
from src.optimization.instance import Instance
from src.optimization.models.uncapacitated import UncapacitatedSAAModel
from src.tools.logging import get_logger

logger = get_logger("VerifyEndToEnd")


def run_regime(regime: str, n_scenarios: int, max_run_time: float) -> dict:
    """Build the instance, run the CA and solve the uncapacitated model."""
    instance = Instance(
        id_instance=f"verify_{regime}",
        is_continuous_var_x=True,
        type_of_flexibility="fixed_capacity",
        N=n_scenarios,
        regime=regime,
        use_euclidean_distance=True,
    )

    model = UncapacitatedSAAModel(instance)
    model.set_params({"TimeLimit": max_run_time, "MIPGap": 0.01, "OutputFlag": 0})
    model.build()
    result = model.solve()

    # `status` now travels with the objective: BaseSAAModel.solve() records it, so a
    # time-limit incumbent can no longer be read as an optimum.
    return {
        "regime": regime,
        "objective": result["objective_value"],
        "run_time": result["actual_run_time"],
        "gap": result["optimality_gap"],
        "status": result["status"],
        "n_pixels": len(next(iter(instance.scenarios.values())).pixels),
        "n_scenarios": len(instance.scenarios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="scenarios per regime (keep small: this is a smoke test)")
    parser.add_argument("--max-run-time", type=float, default=120.0)
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    args = parser.parse_args()

    results = []
    for regime in args.regimes:
        try:
            results.append(run_regime(regime, args.n, args.max_run_time))
        except Exception as error:  # noqa: BLE001 - the point is to surface the failure
            logger.error(f"[{regime}] FAILED: {type(error).__name__}: {error}")
            raise

    print("\nResultados end-to-end (CA + modelo uncapacitated):")
    print(f"  {'régimen':10s} {'objetivo':>14s} {'gap%':>7s} {'seg':>7s} {'estado':>11s} {'píxeles':>8s} {'esc.':>5s}")
    for row in results:
        print(
            f"  {row['regime']:10s} {row['objective']:>14,.2f} {row['gap']:>7.3f} "
            f"{row['run_time']:>7.1f} {row['status']:>11s} {row['n_pixels']:>8d} {row['n_scenarios']:>5d}"
        )
    not_optimal = [r["regime"] for r in results if r["status"] != "OPTIMAL"]
    if not_optimal:
        print(f"\n  ATENCIÓN: régimen(es) sin resolver a optimalidad: {not_optimal}")

    objectives = [row["objective"] for row in results]
    if args.regimes == list(REGIMES):
        ordered = objectives == sorted(objectives)
        print(f"\nOrden low < normal < high: {'OK' if ordered else 'FALLA'}")
        if not ordered:
            logger.error(f"Los objetivos no están ordenados por régimen: {objectives}")
            sys.exit(1)
    print("El contrato de escenarios se respeta: la CA y Gurobi corren sin cambios.")


if __name__ == "__main__":
    main()
