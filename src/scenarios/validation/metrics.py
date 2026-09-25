"""Scenario statistics shown by the explorer: per-pixel levels and variability, per-period totals."""

import numpy as np
import pandas as pd


def pixel_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-pixel levels and scenario variability for all demand components."""
    grouped = frame.groupby(["id_pixel", "id_scenario"])
    annual_demand, annual_stop, mean_drop = grouped["demand"].sum(), grouped["stop"].sum(), grouped["drop"].mean()
    summary = pd.DataFrame(
        {
            "mean_demand_period": frame.groupby("id_pixel")["demand"].mean(),
            "mean_stop": frame.groupby("id_pixel")["stop"].mean(),
            "mean_drop": frame.groupby("id_pixel")["drop"].mean(),
            "cv_demand": annual_demand.groupby("id_pixel").std() / annual_demand.groupby("id_pixel").mean(),
            "cv_stop": annual_stop.groupby("id_pixel").std() / annual_stop.groupby("id_pixel").mean(),
            "cv_drop": mean_drop.groupby("id_pixel").std() / mean_drop.groupby("id_pixel").mean(),
        }
    )
    geo = frame.drop_duplicates("id_pixel").set_index("id_pixel")[["layer", "lon", "lat", "n_cells"]]
    return summary.join(geo)


def regime_period_metrics(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for regime, frame in data.items():
        totals = frame.groupby(["id_scenario", "period"])[["stop", "demand"]].sum()
        rows.append(
            pd.DataFrame(
                {
                    "regime": regime,
                    "period": totals.index.get_level_values("period"),
                    "stops": totals["stop"].to_numpy(),
                    "demand": totals["demand"].to_numpy(),
                    "drop": (totals["demand"] / totals["stop"]).to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def annual_scenario_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Annual demand, one row per scenario and one column per pixel."""
    return frame.groupby(["id_scenario", "id_pixel"])["demand"].sum().unstack("id_pixel")


def pairwise_scenario_correlations(frame: pd.DataFrame) -> np.ndarray:
    """Correlation between every pair of scenarios' annual pixel-demand vectors (upper triangle)."""
    corr = np.corrcoef(annual_scenario_matrix(frame).to_numpy())
    return corr[np.triu_indices_from(corr, k=1)]
