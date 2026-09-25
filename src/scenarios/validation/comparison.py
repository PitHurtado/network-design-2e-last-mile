"""Metrics of the dependence-model comparison: each generator against the historical panel.

Correlation-type metrics work on log values centred per pixel. The neighbour metrics use
grid-edge adjacency (`spatial.pixel_neighbor_pairs`); the distance correlogram uses
centroid distances.
"""

import numpy as np
import pandas as pd

from src.core.constants import GRID_N_COLS, GRID_N_ROWS, N_PERIODS
from src.core.contract import DEPENDENCE_METHODS as METHODS
from src.core.grid import GRID_X, GRID_Y, cell_center, layer_of
from src.scenarios.spatial import haversine_matrix, pixel_centroids, pixel_grid_cells, pixel_neighbor_pairs

QUANTITIES = ("stop", "drop", "demand")
HISTORICAL_COLUMNS = {"stop": "n_customers", "drop": "drop", "demand": "model_demand"}


def historical_panel(panel: pd.DataFrame) -> pd.DataFrame:
    included = panel[~panel["excluded"]].copy()
    included["observation"] = list(zip(included["year"], included["month"]))
    return included


def neighbor_correlations(frame: pd.DataFrame, pairs: pd.DataFrame, value: str, historical: bool) -> pd.DataFrame:
    """Correlation of normalized log values for the requested pixel pairs."""
    if frame.empty or pairs.empty:
        return pd.DataFrame(columns=["left", "right", "ring", "correlation"])
    work = frame.copy()
    if historical:
        work["observation"] = list(zip(work["year"], work["month"]))
    else:
        work["observation"] = list(zip(work["id_scenario"], work["period"]))
    work["value"] = np.log(work[value].clip(lower=1e-9))
    work["value"] = work["value"] - work.groupby("id_pixel")["value"].transform("mean")
    matrix = work.pivot_table(index="observation", columns="id_pixel", values="value", aggfunc="mean")
    rows = []
    for pair in pairs.itertuples(index=False):
        if pair.id_pixel not in matrix or pair.neighbor not in matrix:
            continue
        values = matrix[[pair.id_pixel, pair.neighbor]].dropna()
        if len(values) < 4 or values.iloc[:, 0].std() == 0 or values.iloc[:, 1].std() == 0:
            continue
        rows.append(
            {
                "left": pair.id_pixel,
                "right": pair.neighbor,
                "ring": int(pair.ring),
                "correlation": float(values.iloc[:, 0].corr(values.iloc[:, 1])),
            }
        )
    return pd.DataFrame(rows)


def conditional_high(frame: pd.DataFrame, pairs: pd.DataFrame, value: str, historical: bool) -> float:
    """P(neighbor is high | focal pixel is high), using each pixel's p75."""
    if frame.empty or pairs.empty:
        return float("nan")
    work = frame.copy()
    work["observation"] = list(zip(work["year"], work["month"])) if historical else list(zip(work["id_scenario"], work["period"]))
    thresholds = work.groupby("id_pixel")[value].quantile(0.75)
    work["high"] = work[value] >= work["id_pixel"].map(thresholds)
    matrix = work.pivot_table(index="observation", columns="id_pixel", values="high", aggfunc="mean")
    denominator = numerator = 0
    for pair in pairs.itertuples(index=False):
        if pair.id_pixel not in matrix or pair.neighbor not in matrix:
            continue
        values = matrix[[pair.id_pixel, pair.neighbor]].dropna()
        focal = values.iloc[:, 0].astype(bool)
        denominator += int(focal.sum())
        numerator += int((focal & values.iloc[:, 1].astype(bool)).sum())
    return numerator / denominator if denominator else float("nan")


def pixel_period_metrics(generated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-pixel/per-period mean, standard deviation and CV across scenarios."""
    rows = []
    for method, frame in generated.items():
        grouped = frame.groupby(["period", "id_pixel"])["demand"]
        summary = grouped.agg(
            mean_demand="mean",
            std_demand="std",
            p10=lambda x: x.quantile(0.1),
            p50="median",
            p90=lambda x: x.quantile(0.9),
        ).reset_index()
        averages = (
            frame.groupby(["period", "id_pixel"])[["stop", "drop"]]
            .mean()
            .rename(columns={"stop": "mean_stop", "drop": "mean_drop"})
            .reset_index()
        )
        summary = summary.merge(averages, on=["period", "id_pixel"], how="left")
        summary["cv_demand"] = summary["std_demand"] / summary["mean_demand"].replace(0, np.nan)
        summary["method"] = method
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def distance_rows(panel: pd.DataFrame, generated: dict[str, pd.DataFrame], pixels: list[str]) -> pd.DataFrame:
    """Correlogram by centroid distance for history and every generator."""
    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())
    iu = np.triu_indices(len(pixels), k=1)
    pair_distances = distances[iu]
    edges = np.quantile(pair_distances, np.linspace(0, 1, 13))
    edges = np.unique(edges)

    def values_by_pixel(frame: pd.DataFrame, historical: bool) -> np.ndarray:
        work = frame.copy()
        work["observation"] = (
            list(zip(work["year"], work["month"])) if historical else list(zip(work["id_scenario"], work["period"]))
        )
        work["value"] = np.log(work["demand"].clip(lower=1e-9))
        work["value"] -= work.groupby("id_pixel")["value"].transform("mean")
        return work.pivot_table(index="observation", columns="id_pixel", values="value").reindex(columns=pixels).to_numpy().T

    history = historical_panel(panel).rename(columns={"model_demand": "demand"})
    frames = {"historical": (history, True), **{method: (frame, False) for method, frame in generated.items()}}
    rows = []
    for method, (frame, is_history) in frames.items():
        values = values_by_pixel(frame, is_history)
        corr = np.corrcoef(np.nan_to_num(values, nan=0.0))
        pair_corr = corr[iu]
        for bin_index in range(len(edges) - 1):
            selected = (pair_distances >= edges[bin_index]) & (pair_distances <= edges[bin_index + 1])
            if selected.any():
                rows.append(
                    {
                        "method": method,
                        "distance_bin": bin_index + 1,
                        "distance_km": pair_distances[selected].mean(),
                        "correlation": pair_corr[selected].mean(),
                        "n_pairs": int(selected.sum()),
                    }
                )
    return pd.DataFrame(rows)


def metric_rows(panel: pd.DataFrame, generated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    historical = historical_panel(panel)
    rows = []
    for quantity in QUANTITIES:
        hist = historical[HISTORICAL_COLUMNS[quantity]].astype(float)
        for method, frame in generated.items():
            values = frame[quantity].astype(float)
            rows.append(
                {
                    "metric": "marginal",
                    "quantity": quantity,
                    "method": method,
                    "historical_mean": hist.mean(),
                    "generated_mean": values.mean(),
                    "relative_mean_error": values.mean() / hist.mean() - 1,
                    "historical_std": hist.std(ddof=1),
                    "generated_std": values.std(ddof=1),
                    "std_ratio": values.std(ddof=1) / hist.std(ddof=1),
                    "historical_p10": hist.quantile(0.10),
                    "generated_p10": values.quantile(0.10),
                    "historical_p50": hist.quantile(0.50),
                    "generated_p50": values.quantile(0.50),
                    "historical_p90": hist.quantile(0.90),
                    "generated_p90": values.quantile(0.90),
                }
            )

        historical_agg = historical.groupby("observation")[HISTORICAL_COLUMNS[quantity]].sum()
        for method, frame in generated.items():
            aggregate = frame.groupby(["id_scenario", "period"])[quantity].sum()
            rows.append(
                {
                    "metric": "aggregate_cv",
                    "quantity": quantity,
                    "method": method,
                    "historical_mean": historical_agg.mean(),
                    "generated_mean": aggregate.mean(),
                    "historical_std": historical_agg.std(ddof=1),
                    "generated_std": aggregate.std(ddof=1),
                    "historical_cv": historical_agg.std(ddof=1) / historical_agg.mean(),
                    "generated_cv": aggregate.std(ddof=1) / aggregate.mean(),
                }
            )
    return pd.DataFrame(rows)


def neighbor_rows(panel: pd.DataFrame, generated: dict[str, pd.DataFrame], pixels: list[str]) -> pd.DataFrame:
    historical = historical_panel(panel)
    rows = []
    for ring in (1, 2):
        pairs = pixel_neighbor_pairs(pixels=pixels, ring=ring)
        for quantity in QUANTITIES:
            hist_corr = neighbor_correlations(historical, pairs, HISTORICAL_COLUMNS[quantity], historical=True)
            if not hist_corr.empty:
                rows.append(
                    {
                        "metric": "neighbor_correlation",
                        "quantity": quantity,
                        "method": "historical",
                        "ring": ring,
                        "mean_correlation": hist_corr["correlation"].mean(),
                        "n_pairs": len(hist_corr),
                        "high_given_high": conditional_high(historical, pairs, HISTORICAL_COLUMNS[quantity], historical=True),
                    }
                )
            for method, frame in generated.items():
                corr = neighbor_correlations(frame, pairs, quantity, historical=False)
                rows.append(
                    {
                        "metric": "neighbor_correlation",
                        "quantity": quantity,
                        "method": method,
                        "ring": ring,
                        "mean_correlation": corr["correlation"].mean() if not corr.empty else np.nan,
                        "n_pairs": len(corr),
                        "high_given_high": conditional_high(frame, pairs, quantity, historical=False),
                    }
                )
    return pd.DataFrame(rows)


def grid_inspector_payload(
    generated: dict[str, pd.DataFrame],
    pixels: list[str],
    footprints: dict[str, set[int]] | None = None,
) -> dict:
    """Serialize the local neighborhood metrics used by the grid hover panel."""
    metrics = pixel_period_metrics(generated)

    def finite(value):
        value = float(value)
        return round(value, 6) if np.isfinite(value) else None

    footprints = footprints or pixel_grid_cells()
    payload = {
        "pixel_order": pixels,
        "pixel_index": {pixel: index for index, pixel in enumerate(pixels)},
        "layers": {pixel: layer_of(pixel) for pixel in pixels},
        "grid": {
            "rows": GRID_N_ROWS,
            "cols": GRID_N_COLS,
            "x": GRID_X,
            "y": GRID_Y,
            "pixel_cells": {pixel: [int(cell) for cell in sorted(footprints.get(pixel, set()))] for pixel in pixels},
            "centers": {
                pixel: np.asarray([cell_center(cell) for cell in footprints.get(pixel, set())], dtype=float).mean(axis=0).tolist()
                for pixel in pixels
                if footprints.get(pixel)
            },
        },
        "methods": {},
        "neighbors": {pixel: [] for pixel in pixels},
        "relative_cv": {},
    }
    for method in METHODS:
        payload["methods"][method] = {}
        view = metrics[metrics["method"] == method]
        for period in range(N_PERIODS):
            period_view = view[view["period"] == period].set_index("id_pixel")
            payload["methods"][method][str(period)] = {
                pixel: {
                    "stop": finite(period_view.loc[pixel, "mean_stop"]),
                    "drop": finite(period_view.loc[pixel, "mean_drop"]),
                    "demand": finite(period_view.loc[pixel, "mean_demand"]),
                    "cv": finite(period_view.loc[pixel, "cv_demand"]),
                }
                for pixel in pixels
                if pixel in period_view.index
            }

    # CV of demand(target) / demand(focal), evaluated over paired scenarios.
    # This is the spatial relationship requested by the local subplot: zero for
    # the focal pixel and larger values for less stable co-movement.
    for method, frame in generated.items():
        payload["relative_cv"][method] = {}
        for period in range(N_PERIODS):
            matrix = (
                frame[frame["period"] == period]
                .pivot(index="id_scenario", columns="id_pixel", values="demand")
                .reindex(columns=pixels)
                .to_numpy(dtype=float)
            )
            relative = np.full((len(pixels), len(pixels)), np.nan)
            for focal_index in range(len(pixels)):
                focal = matrix[:, focal_index]
                ratios = matrix / focal[:, None]
                mean = np.nanmean(ratios, axis=0)
                std = np.nanstd(ratios, axis=0, ddof=1)
                relative[focal_index] = std / np.where(np.abs(mean) > 1e-12, mean, np.nan)
                relative[focal_index, focal_index] = 0.0
            payload["relative_cv"][method][str(period)] = {
                pixel: [round(float(value), 6) if np.isfinite(value) else None for value in relative[focal_index]]
                for focal_index, pixel in enumerate(pixels)
            }

    for pair in pixel_neighbor_pairs(pixels=pixels, ring=1).itertuples(index=False):
        payload["neighbors"][pair.id_pixel].append(pair.neighbor)
        payload["neighbors"][pair.neighbor].append(pair.id_pixel)
    return payload


def comparison_long(panel: pd.DataFrame, generated: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every generated and historical observation in one long table (`comparison_long.csv`)."""
    generated_long = pd.concat(generated.values(), ignore_index=True)
    historical_long = (
        historical_panel(panel)
        .rename(columns={"n_customers": "stop", "model_demand": "demand"})[
            ["year", "month", "id_pixel", "stop", "drop", "demand"]
        ]
        .copy()
    )
    historical_long["method"] = "historical"
    historical_long["regime"] = "historical"
    historical_long["id_scenario"] = historical_long.apply(
        lambda row: f"historical-{int(row['year'])}-{int(row['month']):02d}", axis=1
    )
    historical_long["period"] = historical_long["month"] - 1
    return pd.concat([generated_long, historical_long], ignore_index=True, sort=False)


def paired_summary(
    regime: str,
    version: str,
    long: pd.DataFrame,
    generated: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    neighbors: pd.DataFrame,
) -> dict:
    """Headline numbers of the paired comparison (`summary.json`)."""
    paired_differences = {}
    for method in METHODS[1:]:
        pair = generated["independent"].merge(
            generated[method], on=["id_scenario", "id_pixel", "period"], suffixes=("_independent", f"_{method}")
        )
        paired_differences[method] = float((pair["demand_independent"] - pair[f"demand_{method}"]).abs().mean())
    generated_long = pd.concat(generated.values(), ignore_index=True)
    return {
        "regime": regime,
        "version": version,
        "n_rows": int(len(long)),
        "n_scenarios": int(generated_long["id_scenario"].nunique()),
        "mean_absolute_demand_difference_vs_independent": paired_differences,
        "aggregate_cv": {
            row["method"]: float(row["generated_cv"])
            for _, row in metrics[(metrics["metric"] == "aggregate_cv") & (metrics["quantity"] == "demand")].iterrows()
        },
        "first_ring_demand_correlation": {
            row["method"]: float(row["mean_correlation"])
            for _, row in neighbors[
                (neighbors["metric"] == "neighbor_correlation") & (neighbors["quantity"] == "demand") & (neighbors["ring"] == 1)
            ].iterrows()
        },
    }


def historical_aggregate_cv(panel: pd.DataFrame) -> float:
    """CV of the per-month total model demand in the panel."""
    historical = historical_panel(panel)
    return (
        historical.groupby("observation")["model_demand"].sum().std(ddof=1)
        / historical.groupby("observation")["model_demand"].sum().mean()
    )
