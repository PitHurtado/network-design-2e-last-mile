"""Run and persist the full regime × flexibility × scenario-set experiment."""

from dataclasses import asdict, dataclass
from pathlib import Path

from src.core.constants import RESULTS_DIR, TypeOfFlexibility
from src.optimization.experiments.runner import ExperimentRunner, RunSpec
from src.optimization.experiments.store import ResultStore
from src.optimization.instance import InstanceSpec

CASES = {
    "expected": {"scenario_set": "expected", "n_scenarios": 1, "periods": 12},
    "optimization": {"scenario_set": "optimization", "n_scenarios": 30, "periods": 12},
    "annual_expected": {"scenario_set": "annual_expected", "n_scenarios": 1, "periods": 1},
}
RESULT_FILE = "result.json"


@dataclass(frozen=True)
class ExperimentRun:
    """One reproducible leaf of the flexibility experiment."""

    version: str
    regime: str
    flexibility: str
    case: str

    def run_spec(self, time_limit: float, mip_gap: float) -> RunSpec:
        case = CASES[self.case]
        return RunSpec(
            instance=InstanceSpec(
                id_instance="__".join((self.version, self.flexibility, self.regime, self.case)),
                n_scenarios=case["n_scenarios"],
                regime=self.regime,
                scenario_set=case["scenario_set"],
                periods=case["periods"],
                is_continuous_var_x=False,
                type_of_flexibility=self.flexibility,
                use_euclidean_distance=True,
            ),
            solver={"TimeLimit": time_limit, "MIPGap": mip_gap, "OutputFlag": 0},
        )


def _value(expression) -> float | None:
    return round(expression.getValue(), 3) if expression is not None else None


def run_one(
    run: ExperimentRun, store: ResultStore, runner: ExperimentRunner, time_limit: float, mip_gap: float, overwrite: bool
) -> dict:
    """Solve and persist one leaf, including installation and operational decisions."""
    output_path = store.leaf_path(run.version, run.flexibility, run.regime, run.case)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite or a new results version.")
    case = CASES[run.case]
    try:
        solved = runner.solve(run.run_spec(time_limit, mip_gap))
        instance, model, solve = solved.instance, solved.model, solved.solve
        payload = {
            "run": asdict(run),
            "scenario_set": case["scenario_set"],
            "scenario_ids": instance.scenarios_ids,
            "periods": instance.periods,
            "horizon_weight": instance.horizon_weight,
            "solver": {"time_limit": time_limit, "mip_gap": mip_gap},
            "assignment_variables": "binary",
            "solve": solve,
            "costs": {
                "installation": _value(model.obj.cost_installation),
                "operation_expected": _value(model.obj.cost_operation) / len(instance.scenarios),
                "routing_facilities_expected": _value(model.obj.cost_served_from_facilities) / len(instance.scenarios),
                "routing_dc_expected": _value(model.obj.cost_served_from_dc) / len(instance.scenarios),
            },
            "decisions": model.decisions() if solve["objective_value"] is not None else None,
        }
    except Exception as error:  # noqa: BLE001 - persist every failed experiment leaf for auditability
        payload = {"run": asdict(run), "status": "ERROR", "error_type": type(error).__name__, "error": str(error)}
    store.write(output_path, payload, overwrite=True)
    return payload


def run_experiment(
    version: str,
    regimes: list[str],
    flexibilities: list[str],
    cases: list[str],
    time_limit: float,
    mip_gap: float,
    overwrite: bool = False,
    output_root: Path | None = None,
    runner: ExperimentRunner | None = None,
) -> list[dict]:
    """Run the Cartesian product in a stable order and return persisted payloads."""
    store = ResultStore(output_root or (RESULTS_DIR / "flexibility"), RESULT_FILE)
    runner = runner or ExperimentRunner.for_version(version)
    invalid = set(flexibilities) - {item.value for item in TypeOfFlexibility}
    if invalid:
        raise ValueError(f"Unknown flexibility configuration(s): {sorted(invalid)}")
    invalid = set(cases) - set(CASES)
    if invalid:
        raise ValueError(f"Unknown experiment case(s): {sorted(invalid)}")
    return [
        run_one(ExperimentRun(version, regime, flexibility, case), store, runner, time_limit, mip_gap, overwrite)
        for flexibility in flexibilities
        for regime in regimes
        for case in cases
    ]
