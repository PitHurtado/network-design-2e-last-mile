"""Checks on shape parameters: does the estimator recover what the generator was given?

`roundtrip_validation` is the acceptance test of the spatial dependence; `aggregate_cv_impact`
measures how much aggregate risk each part of the dependence structure contributes.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.scenarios.fitting.marginals import fit_marginals
from src.scenarios.generation.dependence import SpatialJoint
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.spatial import (
    cholesky_factor,
    correlation_matrix,
    empirical_correlogram,
    fit_correlogram,
    haversine_matrix,
    model_correlation,
    pixel_centroids,
)
from src.tools.logging import get_logger

logger = get_logger("ScenarioAnalysis")


def correlogram_curve(params: Mapping, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """The fitted correlation model on a distance grid spanning the empirical correlogram."""
    spatial = params["spatial"]
    empirical = pd.DataFrame(spatial["correlogram"])
    grid = np.linspace(0.1, empirical["h_km"].max() * 1.05, n_points)
    return grid, model_correlation(grid, spatial["rho_km"], spatial["nugget"], spatial["plateau"])


def roundtrip_validation(params: Mapping, n_periods: int, n_pixels_reference: int) -> dict:
    """Refit the whole spatial pipeline on synthetic data with a known truth.

    This is the acceptance test for the dependence structure. Comparing the
    generated correlogram directly against the fitted model is misleading: the
    estimator centres deviations per period, and that operation does not commute
    with the row-centring inside the correlation, so the two are only comparable
    when measured through the *same* pipeline at the *same* sample size.

    Here a panel of the real panel's shape is synthesized from a known
    `(rho, nugget, plateau)` and pushed back through the estimator. If the recovered
    values match the input, the generator and the estimator agree.
    """
    spatial = params["spatial"]
    pixels = params["pixels"]

    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())

    truth = {"rho_km": spatial["rho_km"], "nugget": spatial["nugget"], "plateau": spatial["plateau"]}
    sigma = correlation_matrix(distances, truth["rho_km"], truth["nugget"], truth["plateau"])

    generator = ScenarioGenerator.from_params(params)
    generator.dependence = SpatialJoint(cholesky_factor(sigma))
    multiplier = params["regimes"]["normal"]["multiplier"]

    rng = np.random.default_rng(7)
    year_months = [(y, m) for y in (2020, 2021, 2022) for m in range(1, 13)][:n_periods]
    rows = []
    for year, month in year_months:
        stop, drop = generator.draw(rng, multiplier)
        for i, id_pixel in enumerate(pixels):
            rows.append(
                {
                    "id_pixel": id_pixel,
                    "layer": id_pixel.split("-")[0],
                    "year": year,
                    "month": month,
                    "n_customers": int(stop[i, month - 1]),
                    "drop": float(drop[i, month - 1]),
                    "excluded": False,
                }
            )
    synthetic = pd.DataFrame(rows)

    refitted = fit_marginals(synthetic)
    balanced = refitted["stop"]["deviations_balanced"]
    sub_centroids = centroids.reindex(balanced.index)
    sub_distances = haversine_matrix(sub_centroids["lon"].to_numpy(), sub_centroids["lat"].to_numpy())
    correlogram = empirical_correlogram(balanced, sub_distances, n_bins=12)
    recovered = fit_correlogram(correlogram)

    logger.info(
        f"Round-trip: truth rho={truth['rho_km']:.2f}/nugget={truth['nugget']:.3f}/plateau={truth['plateau']:.3f} "
        f"-> recovered rho={recovered['rho_km']:.2f}/nugget={recovered['nugget']:.3f}/plateau={recovered['plateau']:.3f}"
    )
    return {
        "truth": truth,
        "recovered": {k: recovered[k] for k in ("rho_km", "nugget", "plateau", "r2", "model")},
        "correlogram": correlogram,
        "n_pixels": len(balanced),
        "n_pixels_reference": n_pixels_reference,
        "n_periods": n_periods,
    }


def aggregate_cv_impact(params: Mapping) -> dict:
    """Coefficient of variation of aggregate per-period demand under each structure.

    Answers the question the whole redesign exists for: how much aggregate risk does
    the facility-location decision actually see? Computed analytically from the
    fitted dispersions and the demand-weighted pixel shares, so it does not depend
    on the Monte Carlo sample.
    """
    pixels = params["pixels"]
    spatial = params["spatial"]

    generator = ScenarioGenerator.from_params(params)
    weights = generator.expected_stop.mean(axis=1) * generator.expected_drop.mean(axis=1)
    weights = weights / weights.sum()

    sigma = np.array([params["stop"]["sigma"][p] for p in pixels])
    sigma_common = params["stop"]["sigma_common"]

    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())

    def cv(rho_km, nugget, plateau, with_common):
        corr = model_correlation(distances, rho_km, nugget, plateau)
        np.fill_diagonal(corr, 1.0)
        spatial_part = float((np.outer(weights, weights) * np.outer(sigma, sigma) * corr).sum())
        return float(np.sqrt((sigma_common**2 if with_common else 0.0) + spatial_part))

    cv_independent = cv(1e-9, 0.0, 0.0, False)
    cv_no_plateau = cv(spatial["rho_km"], spatial["nugget"], 0.0, True)
    cv_full = cv(spatial["rho_km"], spatial["nugget"], spatial["plateau"], True)

    return {
        "cv_independent": cv_independent,
        "cv_no_plateau": cv_no_plateau,
        "cv_full": cv_full,
        "ratio": cv_full / cv_independent if cv_independent else float("nan"),
        "plateau_pct": (cv_full / cv_no_plateau - 1) * 100 if cv_no_plateau else float("nan"),
    }
