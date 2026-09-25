"""Out-of-sample recourse evaluation for persisted flexibility decisions.

Each saved installation `Y` is fixed and the recourse is re-optimized over the validation
scenarios. Fixed Y plus the same policy and validation set define the identical recourse
problem, so two cases that chose the same Y share one solve.
"""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from src.optimization.experiments.flexibility import RESULT_FILE
from src.optimization.experiments.runner import ExperimentRunner, RunSpec
from src.optimization.experiments.store import ResultStore
from src.optimization.instance import InstanceSpec

SOLUTION_CASES = ("annual_expected", "expected", "optimization")
VALIDATION_SCENARIOS = 100
EVALUATION_FILE = "evaluation.json"


def _fixed_installation(payload: dict) -> dict[str, float]:
    decisions = payload.get("decisions")
    if not decisions or not decisions.get("installation"):
        raise ValueError("Source result has no feasible persisted installation decision.")
    return {row["facility"]: row["capacity"] for row in decisions["installation"]}


def _matching_evaluation(store: ResultStore, output_path: Path, fixed_installation: dict[str, float]) -> dict | None:
    """Find a prior leaf with exactly the same fixed Y within policy and regime."""
    for candidate in store.siblings(output_path):
        payload = store.read(candidate)
        candidate_y = {row["facility"]: row["capacity"] for row in payload.get("fixed_installation", [])}
        if (
            candidate_y == fixed_installation
            and payload.get("assignment_variables") == "binary"
            and len(payload.get("scenario_costs", [])) == VALIDATION_SCENARIOS
        ):
            return payload
    return None


@dataclass(frozen=True)
class SourceRun:
    """The flexibility run whose installation decisions are evaluated."""

    run_id: str
    run_dir: Path

    @property
    def store(self) -> ResultStore:
        return ResultStore(self.run_dir / "flexibility", RESULT_FILE)


def _source_fields(source: SourceRun, source_path: Path) -> dict:
    """Where the fixed Y came from: the source run, the leaf within it, and its solve record."""
    return {
        "source_run": source.run_id,
        "source_result": source_path.relative_to(source.run_dir).as_posix(),
        "source_solve": source.store.read(source_path)["solve"],
    }


def evaluate_one(
    version: str,
    regime: str,
    flexibility: str,
    solution_case: str,
    time_limit: float,
    source: SourceRun,
    output_root: Path,
    overwrite: bool = False,
    runner: ExperimentRunner | None = None,
) -> dict:
    """Fix a saved Y and optimize recourse over the validation scenarios."""
    if solution_case not in SOLUTION_CASES:
        raise ValueError(f"Unknown source solution case: {solution_case}")
    store = ResultStore(output_root, EVALUATION_FILE)
    runner = runner or ExperimentRunner.for_version(version)
    output_path = store.leaf_path(version, flexibility, regime, solution_case)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite or another version.")

    source_path = source.store.leaf_path(version, flexibility, regime, solution_case)
    fixed_installation = _fixed_installation(source.store.read(source_path))
    matching = _matching_evaluation(store, output_path, fixed_installation)
    if matching is not None:
        matching["solution_case"] = solution_case
        matching.update(_source_fields(source, source_path))
        store.write(output_path, matching, overwrite=True)
        return matching

    solved = runner.solve(
        RunSpec(
            instance=InstanceSpec(
                id_instance="__".join((version, "evaluation", flexibility, regime, solution_case)),
                n_scenarios=VALIDATION_SCENARIOS,
                regime=regime,
                scenario_set="validation",
                periods=12,
                is_continuous_var_x=False,
                type_of_flexibility=flexibility,
                use_euclidean_distance=True,
            ),
            solver={"TimeLimit": time_limit, "MIPGap": 0.0, "OutputFlag": 0},
            model_kwargs={"fixed_installation": fixed_installation},
        )
    )
    instance, model, solve = solved.instance, solved.model, solved.solve
    if solve["objective_value"] is None:
        raise RuntimeError(f"No feasible recourse solution: {solve['status']}")
    scenario_costs = model.scenario_costs()
    second_stage_mean = sum(row["second_stage_cost"] for row in scenario_costs) / len(scenario_costs)
    total_mean = sum(row["total_cost"] for row in scenario_costs) / len(scenario_costs)
    # Per-scenario costs are a second reading of the objective; they must agree with it.
    tolerance = 0.01 + 1e-9 * abs(solve["objective_value"])
    if abs(total_mean - solve["objective_value"]) > tolerance:
        raise RuntimeError(f"Scenario costs average {total_mean:,.3f} but the objective is {solve['objective_value']:,.3f}.")
    output = {
        "version": version,
        "regime": regime,
        "flexibility": flexibility,
        "solution_case": solution_case,
        **_source_fields(source, source_path),
        "evaluation_scenario_set": "validation",
        "evaluation_scenario_ids": instance.scenarios_ids,
        "assignment_variables": "binary",
        "fixed_installation": [
            {"facility": facility, "capacity": capacity, "installed": capacity > 0}
            for facility, capacity in sorted(fixed_installation.items())
        ],
        "solve": solve,
        "means": {"second_stage_cost": round(second_stage_mean, 3), "total_cost": round(total_mean, 3)},
        "scenario_costs": scenario_costs,
    }
    store.write(output_path, output, overwrite=True)
    return output


def evaluate_experiment(
    version,
    regimes,
    flexibilities,
    solution_cases,
    time_limit,
    source: SourceRun,
    output_root: Path,
    overwrite=False,
    runner=None,
):
    """Evaluate every saved Y of `source`, reusing an exact evaluation when two cases chose the same Y.

    The duplicated leaves are still persisted independently, retaining their source label.
    """
    store = ResultStore(output_root, EVALUATION_FILE)
    runner = runner or ExperimentRunner.for_version(version)
    cache, results = {}, []
    for flexibility in flexibilities:
        for regime in regimes:
            for solution_case in solution_cases:
                source_path = source.store.leaf_path(version, flexibility, regime, solution_case)
                fixed = _fixed_installation(source.store.read(source_path))
                key = (flexibility, regime, tuple(sorted(fixed.items())))
                if key not in cache:
                    output = evaluate_one(
                        version,
                        regime,
                        flexibility,
                        solution_case,
                        time_limit,
                        source=source,
                        output_root=output_root,
                        overwrite=overwrite,
                        runner=runner,
                    )
                    cache[key] = output
                else:
                    output_path = store.leaf_path(version, flexibility, regime, solution_case)
                    if output_path.exists() and not overwrite:
                        raise FileExistsError(f"{output_path} exists; use --overwrite or another version.")
                    output = deepcopy(cache[key])
                    output["solution_case"] = solution_case
                    output.update(_source_fields(source, source_path))
                    store.write(output_path, output, overwrite=True)
                results.append(output)
    return results
