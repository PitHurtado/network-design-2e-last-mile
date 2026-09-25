"""Tables of the flexibility experiment: one row per solved leaf, plus its decisions."""

import json
from pathlib import Path

import pandas as pd


def load_runs(root: Path, version: str) -> list[dict]:
    runs = []
    for path in sorted((root / version).rglob("result.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "ERROR":
            payload["_path"] = str(path)
            runs.append(payload)
    return runs


def run_tables(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, decisions, operation = [], [], []
    for item in runs:
        run, costs, solve = item["run"], item["costs"], item["solve"]
        installed = item["decisions"]["installation"]
        summary.append(
            {
                **run,
                "objective": solve["objective_value"],
                "installation_cost": costs["installation"],
                "operation_cost": costs["operation_expected"],
                "routing_cost": costs["routing_facilities_expected"] + costs["routing_dc_expected"],
                "installed_satellites": sum(row["installed"] for row in installed),
                "installed_capacity": sum(row["capacity"] for row in installed),
                "status": solve["status"],
                "optimality_gap_pct": solve["optimality_gap"],
                "is_optimal": solve["is_optimal"],
            }
        )
        decisions.extend([{**run, **row} for row in installed])
        operation.extend([{**run, **row} for row in item["decisions"]["operation"]])
    return pd.DataFrame(summary), pd.DataFrame(decisions), pd.DataFrame(operation)
