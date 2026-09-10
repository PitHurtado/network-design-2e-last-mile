"""Out-of-sample recourse evaluation for persisted flexibility decisions."""

import json
from copy import deepcopy
from pathlib import Path

from src.core.constants import RESULTS_DIR
from src.optimization.instance import Instance
from src.optimization.models.flex import FlexSAAModel

SOLUTION_CASES = ("annual_expected", "expected", "optimization")
VALIDATION_SCENARIOS = 100


def _source_path(version: str, flexibility: str, regime: str, solution_case: str, root: Path) -> Path:
    return root / version / flexibility / regime / solution_case / "result.json"


def _fixed_installation(payload: dict) -> dict[str, float]:
    decisions = payload.get("decisions")
    if not decisions or not decisions.get("installation"):
        raise ValueError("Source result has no feasible persisted installation decision.")
    return {row["facility"]: row["capacity"] for row in decisions["installation"]}


def _matching_evaluation(output_path: Path, fixed_installation: dict[str, float]) -> dict | None:
    """Find a prior leaf with exactly the same fixed Y within policy and regime."""
    for candidate in output_path.parent.parent.glob("*/evaluation.json"):
        if candidate == output_path:
            continue
        payload = json.loads(candidate.read_text())
        candidate_y = {row["facility"]: row["capacity"] for row in payload.get("fixed_installation", [])}
        if candidate_y == fixed_installation and len(payload.get("scenario_costs", [])) == VALIDATION_SCENARIOS:
            return payload
    return None


def evaluate_one(
    version: str,
    regime: str,
    flexibility: str,
    solution_case: str,
    time_limit: float,
    source_root: Path | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Fix a saved Y and optimize recourse over the 100 validation scenarios."""
    if solution_case not in SOLUTION_CASES:
        raise ValueError(f"Unknown source solution case: {solution_case}")
    source_root = source_root or (RESULTS_DIR / "flexibility")
    output_root = output_root or (RESULTS_DIR / "flexibility_evaluation")
    output_path = output_root / version / flexibility / regime / solution_case / "evaluation.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite or another version.")

    source_path = _source_path(version, flexibility, regime, solution_case, source_root)
    source = json.loads(source_path.read_text())
    fixed_installation = _fixed_installation(source)
    matching = _matching_evaluation(output_path, fixed_installation)
    if matching is not None:
        matching["solution_case"] = solution_case
        matching["source_result"] = str(source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(matching, indent=2))
        return matching
    instance = Instance(
        id_instance="__".join((version, "evaluation", flexibility, regime, solution_case)),
        is_continuous_var_x=False,
        type_of_flexibility=flexibility,
        N=VALIDATION_SCENARIOS,
        regime=regime,
        periods=12,
        scenario_version=version,
        scenario_set="validation",
        use_euclidean_distance=True,
    )
    model = FlexSAAModel(instance, fixed_installation=fixed_installation)
    model.set_params({"TimeLimit": time_limit, "MIPGap": 0.0, "OutputFlag": 0})
    model.build()
    solve = model.solve()
    if solve["objective_value"] is None:
        raise RuntimeError(f"No feasible recourse solution: {solve['status']}")
    scenario_costs = model.scenario_costs()
    second_stage_mean = sum(row["second_stage_cost"] for row in scenario_costs) / len(scenario_costs)
    total_mean = sum(row["total_cost"] for row in scenario_costs) / len(scenario_costs)
    output = {
        "version": version,
        "regime": regime,
        "flexibility": flexibility,
        "solution_case": solution_case,
        "source_result": str(source_path),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    return output


def evaluate_experiment(version, regimes, flexibilities, solution_cases, time_limit, overwrite=False):
    """Evaluate every saved Y, reusing an exact evaluation when two cases chose the same Y.

    The duplicated leaves are still persisted independently, retaining their source
    label. This is mathematically exact because fixed Y plus the same policy and
    validation set define the identical recourse problem.
    """
    source_root, output_root, cache, results = RESULTS_DIR / "flexibility", RESULTS_DIR / "flexibility_evaluation", {}, []
    for flexibility in flexibilities:
        for regime in regimes:
            for solution_case in solution_cases:
                source_path = _source_path(version, flexibility, regime, solution_case, source_root)
                fixed = _fixed_installation(json.loads(source_path.read_text()))
                key = (flexibility, regime, tuple(sorted(fixed.items())))
                if key not in cache:
                    output = evaluate_one(
                        version, regime, flexibility, solution_case, time_limit,
                        source_root=source_root, output_root=output_root, overwrite=overwrite,
                    )
                    cache[key] = output
                else:
                    output_path = output_root / version / flexibility / regime / solution_case / "evaluation.json"
                    if output_path.exists() and not overwrite:
                        raise FileExistsError(f"{output_path} exists; use --overwrite or another version.")
                    output = deepcopy(cache[key])
                    output["solution_case"] = solution_case
                    output["source_result"] = str(source_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(output, indent=2))
                results.append(output)
    return results
