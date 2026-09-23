"""Solve the validation-sample recourse problem (RP) used for theoretical VSS."""

import json
from pathlib import Path

from src.core.constants import RESULTS_DIR
from src.optimization.instance import Instance
from src.optimization.models.flex import FlexSAAModel


def run_validation_benchmark(version: str, regime: str, flexibility: str, time_limit: float, overwrite: bool = False) -> dict:
    """Persist an RP_100 benchmark; a time limit preserves incumbent and bound."""
    output = RESULTS_DIR / "flexibility_validation_benchmark" / version / flexibility / regime / "rp_validation.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite.")
    instance = Instance(
        id_instance="__".join((version, "rp_validation", flexibility, regime)),
        is_continuous_var_x=False,
        type_of_flexibility=flexibility,
        N=100,
        regime=regime,
        periods=12,
        scenario_version=version,
        scenario_set="validation",
        use_euclidean_distance=True,
    )
    model = FlexSAAModel(instance)
    model.set_params({"TimeLimit": time_limit, "MIPGap": 0.0, "OutputFlag": 0})
    model.build()
    solve = model.solve()
    payload = {
        "version": version,
        "regime": regime,
        "flexibility": flexibility,
        "scenario_set": "validation",
        "n_scenarios": 100,
        "assignment_variables": "binary",
        "solver": {"time_limit": time_limit, "mip_gap": 0.0},
        "solve": solve,
        "decisions": model.decisions() if solve["objective_value"] is not None else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def run_all_validation_benchmarks(version, regimes, flexibilities, time_limit, overwrite=False):
    return [
        run_validation_benchmark(version, regime, flexibility, time_limit, overwrite)
        for flexibility in flexibilities
        for regime in regimes
    ]
