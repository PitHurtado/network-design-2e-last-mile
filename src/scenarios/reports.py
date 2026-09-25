"""Builders of the scenario-side reports: load the artifacts, compute, hand them to a renderer.

The rendering lives in `src.visualization.scenarios`; everything that needs this package
(the parameters, the panel, the pixel geometry, the neighbour structure) is computed here.
Inputs are explicit (a layout, the parameters, the panel); resolving them from artifact
ids is the CLI's job.
"""

from pathlib import Path

import pandas as pd

from src.core.constants import REGIMES
from src.core.contract import DEPENDENCE_METHODS, ScenarioLayout
from src.scenarios.generation.sets import ScenarioSetWriter
from src.scenarios.params import ShapeParams
from src.scenarios.spatial import pixel_centroids, pixel_grid_cells
from src.scenarios.validation import comparison as cmp
from src.scenarios.validation.metrics import pairwise_scenario_correlations, pixel_summary, regime_period_metrics
from src.scenarios.validation.params import aggregate_cv_impact, correlogram_curve, roundtrip_validation
from src.scenarios.validation.scenarios import contract_checks
from src.tools.io import read_json, write_json
from src.tools.logging import get_logger
from src.tools.validation import n_failed
from src.visualization.scenarios import comparison_report, explore_report, validation_report

logger = get_logger("ScenarioReports")


def build_validation_report(
    layout: ScenarioLayout,
    params: ShapeParams,
    panel: pd.DataFrame,
    output_path: Path,
    regimes: tuple[str, ...] = REGIMES,
    params_label: str | None = None,
) -> tuple[Path, int]:
    """Validation report of a scenario version's validation sets; returns the path and the failed checks."""
    regimes = list(regimes)
    writer = ScenarioSetWriter(layout)
    generated = {regime: writer.load_long(regime, "validation") for regime in regimes}
    manifests = {regime: read_json(layout.set_manifest(regime, "validation")) for regime in regimes}

    included = panel[~panel["excluded"]]
    n_periods_panel = included.groupby(["year", "month"]).ngroups
    checks = contract_checks(generated, manifests)
    data = validation_report.ValidationReportData(
        regimes=regimes,
        params=params,
        panel=panel,
        generated=generated,
        manifests=manifests,
        roundtrip=roundtrip_validation(params, n_periods_panel, len(params["pixels"])),
        cv_impact=aggregate_cv_impact(params),
        correlogram_curve=correlogram_curve(params),
        checks=checks,
        params_label=params_label or f"v{params['version']} del {params['generated_on']}",
    )
    path = validation_report.render(data, output_path)
    logger.info(f"Report written to {path} ({n_failed(checks)} failed checks)")
    return path, n_failed(checks)


def build_explore_report(layout: ScenarioLayout, output_path: Path, regimes: tuple[str, ...] = REGIMES) -> Path:
    """Comparative explorer of the low / normal / high validation sets."""
    regimes = list(regimes)
    if len(regimes) != 3 or set(regimes) != {"low", "normal", "high"}:
        raise ValueError("The comparative explorer requires low, normal, and high regimes.")
    centroids = pixel_centroids()[["id_pixel", "layer", "lon", "lat", "n_cells"]]
    writer = ScenarioSetWriter(layout)
    frames = {}
    for regime in regimes:
        frame = writer.load_long(regime, "validation").merge(centroids, on="id_pixel", how="left")
        if frame["layer"].isna().any():
            raise ValueError(f"[{regime}] pixels missing from pixel_centroids()")
        frames[regime] = frame

    data = explore_report.ExploreReportData(
        regimes=regimes,
        summaries={regime: pixel_summary(frame) for regime, frame in frames.items()},
        period_metrics=regime_period_metrics(frames),
        pairwise_correlations={regime: pairwise_scenario_correlations(frames[regime]) for regime in regimes},
        n_scenarios=frames["normal"]["id_scenario"].nunique(),
    )
    path = explore_report.render(data, output_path)
    logger.info(f"Report written to {path}")
    return path


def build_comparison_report(
    layout: ScenarioLayout, params: ShapeParams, panel: pd.DataFrame, regime: str, output_dir: Path, label: str
) -> Path:
    """HTML plus tabular artifacts (CSVs and `summary.json`) for one regime's paired comparison, into `output_dir`."""
    writer = ScenarioSetWriter(layout)
    generated = {method: writer.load_long(regime, "validation", method) for method in DEPENDENCE_METHODS}
    if any(frame.empty for frame in generated.values()):
        raise ValueError(f"Comparison scenarios for {regime} are missing under {layout.root}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demand_comparison.html"

    pixels = list(params["pixels"])
    long = cmp.comparison_long(panel, generated)
    long.to_csv(output_dir / "comparison_long.csv", index=False)
    metrics = cmp.metric_rows(panel, generated)
    neighbors = cmp.neighbor_rows(panel, generated, pixels)
    pixel_period = cmp.pixel_period_metrics(generated)
    distance = cmp.distance_rows(panel, generated, pixels)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    neighbors.to_csv(output_dir / "neighbor_metrics.csv", index=False)
    pixel_period.to_csv(output_dir / "pixel_period_metrics.csv", index=False)
    distance.to_csv(output_dir / "distance_metrics.csv", index=False)
    write_json(output_dir / "summary.json", cmp.paired_summary(regime, label, long, generated, metrics, neighbors))

    footprints = pixel_grid_cells()
    data = comparison_report.ComparisonReportData(
        regime=regime,
        pixels=pixels,
        generated=generated,
        metrics=metrics,
        neighbors=neighbors,
        distance=distance,
        footprints=footprints,
        grid_payload=cmp.grid_inspector_payload(generated, pixels, footprints),
        historical_cv=cmp.historical_aggregate_cv(panel),
        n_scenarios=int(long.loc[long["method"] != "historical", "id_scenario"].nunique()),
    )
    return comparison_report.render(data, output_path)
