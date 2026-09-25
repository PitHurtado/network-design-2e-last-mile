"""Solve the validation-sample recourse problem (RP) used for theoretical VSS."""

from pathlib import Path

from src.core.constants import RESULTS_DIR
from src.optimization.experiments.runner import ExperimentRunner, RunSpec
from src.optimization.experiments.store import ResultStore
from src.optimization.instance import InstanceSpec

VALIDATION_SCENARIOS = 100
BENCHMARK_FILE = "rp_validation.json"


def run_validation_benchmark(
    version: str,
    regime: str,
    flexibility: str,
    time_limit: float,
    overwrite: bool = False,
    output_root: Path | None = None,
    runner: ExperimentRunner | None = None,
) -> dict:
    """Persist an RP_100 benchmark; a time limit preserves incumbent and bound."""
    store = ResultStore(output_root or (RESULTS_DIR / "flexibility_validation_benchmark"), BENCHMARK_FILE)
    runner = runner or ExperimentRunner.for_version(version)
    output = store.leaf_path(version, flexibility, regime)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite.")
    solved = runner.solve(
        RunSpec(
            instance=InstanceSpec(
                id_instance="__".join((version, "rp_validation", flexibility, regime)),
                n_scenarios=VALIDATION_SCENARIOS,
                regime=regime,
                scenario_set="validation",
                periods=12,
                is_continuous_var_x=False,
                type_of_flexibility=flexibility,
                use_euclidean_distance=True,
            ),
            solver={"TimeLimit": time_limit, "MIPGap": 0.0, "OutputFlag": 0},
        )
    )
    solve = solved.solve
    payload = {
        "version": version,
        "regime": regime,
        "flexibility": flexibility,
        "scenario_set": "validation",
        "n_scenarios": VALIDATION_SCENARIOS,
        "assignment_variables": "binary",
        "solver": {"time_limit": time_limit, "mip_gap": 0.0},
        "solve": solve,
        "decisions": solved.model.decisions() if solve["objective_value"] is not None else None,
    }
    store.write(output, payload, overwrite=True)
    return payload


def run_all_validation_benchmarks(version, regimes, flexibilities, time_limit, overwrite=False):
    runner = ExperimentRunner.for_version(version)
    return [
        run_validation_benchmark(version, regime, flexibility, time_limit, overwrite, runner=runner)
        for flexibility in flexibilities
        for regime in regimes
    ]
