"""Out-of-sample evaluation metrics: per-scenario costs, percentiles, observed and theoretical VSS."""

import json
from pathlib import Path

import pandas as pd


def load_evaluations(root: Path, version: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((root / version).rglob("evaluation.json"))]


def evaluation_frames(payloads: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Means (with the source and evaluation solves), per-scenario costs and fixed installations."""
    rows, scenarios, installations = [], [], []
    for data in payloads:
        key = {name: data[name] for name in ("version", "regime", "flexibility", "solution_case")}
        source = json.loads(Path(data["source_result"]).read_text())
        rows.append(
            {
                **key,
                **data["means"],
                **{f"evaluation_{k}": v for k, v in data["solve"].items()},
                **{f"source_{k}": v for k, v in source["solve"].items()},
            }
        )
        scenarios.extend([{**key, **row} for row in data["scenario_costs"]])
        installations.extend([{**key, **row} for row in data["fixed_installation"]])
    return pd.DataFrame(rows), pd.DataFrame(scenarios), pd.DataFrame(installations)


def percentiles(scenarios: pd.DataFrame) -> pd.DataFrame:
    grouped = scenarios.groupby(["regime", "flexibility", "solution_case"])
    rows = []
    for key, group in grouped:
        total, recourse = group["total_cost"], group["second_stage_cost"]
        rows.append(
            {
                "regime": key[0],
                "flexibility": key[1],
                "solution_case": key[2],
                "mean_total": total.mean(),
                "p05_total": total.quantile(0.05),
                "p50_total": total.quantile(0.50),
                "p95_total": total.quantile(0.95),
                "mean_second_stage": recourse.mean(),
                "p95_second_stage": recourse.quantile(0.95),
                "mean_operation": group["operation_cost"].mean(),
                "mean_routing_facilities": group["routing_facilities_cost"].mean(),
                "mean_routing_dc": group["routing_dc_cost"].mean(),
            }
        )
    return pd.DataFrame(rows)


def vss(percentiles: pd.DataFrame) -> pd.DataFrame:
    """Observed VSS: validation mean of a baseline Y minus that of the optimization Y."""
    index = percentiles.set_index(["regime", "flexibility", "solution_case"])
    rows = []
    for regime, flexibility in index.index.droplevel("solution_case").unique():
        optimized = index.loc[(regime, flexibility, "optimization")]
        for baseline in ("expected", "annual_expected"):
            compared = index.loc[(regime, flexibility, baseline)]
            rows.append(
                {
                    "regime": regime,
                    "flexibility": flexibility,
                    "baseline_solution": baseline,
                    "optimized_validation_mean": optimized["mean_total"],
                    "baseline_validation_mean": compared["mean_total"],
                    "vss_cost_reduction": compared["mean_total"] - optimized["mean_total"],
                    "vss_pct": 100 * (compared["mean_total"] - optimized["mean_total"]) / compared["mean_total"],
                }
            )
    return pd.DataFrame(rows)


def load_validation_benchmarks(root: Path, version: str) -> pd.DataFrame:
    """RP benchmarks solved on the validation scenarios (`rp_validation.json`), binary assignments only."""
    root = root / version
    rows = []
    for path in sorted(root.rglob("rp_validation.json")):
        item = json.loads(path.read_text())
        if item.get("assignment_variables") == "binary":
            rows.append({"regime": item["regime"], "flexibility": item["flexibility"], **item["solve"]})
    return pd.DataFrame(rows)


def theoretical_vss(percentiles: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    """VSS interval from an RP incumbent and lower bound; exact only if RP is optimal."""
    if benchmarks.empty:
        return pd.DataFrame()
    base = percentiles[percentiles.solution_case.isin(("annual_expected", "expected"))]
    rows = []
    for item in base.itertuples():
        rp = benchmarks[(benchmarks.regime == item.regime) & (benchmarks.flexibility == item.flexibility)]
        if rp.empty:
            continue
        rp = rp.iloc[0]
        if pd.isna(rp["objective_value"]) or pd.isna(rp["best_bound_value"]):
            continue
        lower = max(0.0, item.mean_total - rp["objective_value"])
        upper = item.mean_total - rp["best_bound_value"]
        rows.append(
            {
                "regime": item.regime,
                "flexibility": item.flexibility,
                "baseline_solution": item.solution_case,
                "baseline_validation_mean": item.mean_total,
                "rp_incumbent": rp["objective_value"],
                "rp_bound": rp["best_bound_value"],
                "vss_lower": lower,
                "vss_upper": upper,
                "is_exact": bool(rp["is_optimal"]),
                "rp_status": rp["status"],
                "rp_gap_pct": rp["optimality_gap"],
            }
        )
    return pd.DataFrame(rows)
