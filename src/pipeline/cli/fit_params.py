"""Fit the scenario shape parameters and write `shape_params.json`.

    poetry run python -m src.pipeline.cli.fit_params

Produces the single reproducible input of the whole study: pixel levels, layer
seasonality, dispersion, the spatial decay, and the regime multipliers.
"""

import argparse
import json
from datetime import date

from src.core.constants import (
    EXCLUDED_YEAR_MONTHS,
    LAYER_DROP_THRESHOLD,
    N_PERIODS,
    PATH_SHAPE_PARAMS,
    REGIME_DROP_EXPONENT,
    REGIME_STOP_EXPONENT,
    REGIME_TARGETS,
    SEED_BASE,
)
from src.core.inputs import get_pixels
from src.pipeline.demand_panel import load_panel
from src.pipeline.generate import ScenarioGenerator
from src.pipeline.marginals import fit_marginals
from src.pipeline.regimes import calibrate_all
from src.pipeline.spatial import (
    cholesky_factor,
    correlation_matrix,
    empirical_correlogram,
    fit_correlogram,
    haversine_matrix,
    model_correlation,
    pixel_centroids,
)
from src.tools.logging import get_logger

logger = get_logger("FitParams")

SHAPE_PARAMS_VERSION = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=12, help="distance bins for the correlogram")
    parser.add_argument("--n", type=int, default=50, help="scenarios per regime (must match the generate step)")
    args = parser.parse_args()

    panel = load_panel()

    # The pixel set must match input_pixels.xlsx exactly: a pixel present in the
    # scenario but missing from the grid is dropped silently by the input reader.
    grid_pixels = set(get_pixels())
    panel_pixels = set(panel["id_pixel"].unique())
    if grid_pixels != panel_pixels:
        raise ValueError(
            f"Pixel set mismatch: {len(grid_pixels - panel_pixels)} in the grid only, "
            f"{len(panel_pixels - grid_pixels)} in the panel only. "
            f"Examples: {sorted(grid_pixels ^ panel_pixels)[:5]}"
        )
    pixels = sorted(grid_pixels)
    logger.info(f"{len(pixels)} pixels, matching input_pixels.xlsx exactly.")

    fitted = fit_marginals(panel)

    # ── Spatial structure ────────────────────────────────────────────────────
    centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
    distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())
    # The correlogram uses the balanced, re-centred deviations; centring on all
    # pixels while correlating only the complete ones inflates the long-range term.
    balanced = fitted["stop"]["deviations_balanced"]
    balanced_centroids = centroids.reindex(balanced.index)
    balanced_distances = haversine_matrix(balanced_centroids["lon"].to_numpy(), balanced_centroids["lat"].to_numpy())
    correlogram = empirical_correlogram(balanced, balanced_distances, n_bins=args.bins)
    spatial = fit_correlogram(correlogram)

    sigma = correlation_matrix(distances, spatial["rho_km"], spatial["nugget"], spatial["plateau"])
    chol = cholesky_factor(sigma)

    print("\nCorrelograma empírico vs ajuste:")
    print(f"  {'h (km)':>8} {'emp.':>8} {'ajuste':>8} {'pares':>9}")
    for _, row in correlogram.iterrows():
        pred = model_correlation(row["h_km"], spatial["rho_km"], spatial["nugget"], spatial["plateau"])
        print(f"  {row['h_km']:8.2f} {row['corr']:8.3f} {pred:8.3f} {int(row['n_pairs']):9d}")

    # ── Regime calibration ──────────────────────────────────────────────────
    generator = ScenarioGenerator.from_fit(fitted, pixels, chol)
    base_total = generator.base_period_total()
    logger.info(f"Base per-period model demand (no regime scaling): {base_total:,.0f}")

    # Calibrated against the exact seeds the generate step will consume, so the
    # written scenarios hit their target instead of being a sampling draw away.
    regimes = calibrate_all(
        total_for=lambda m, rng: generator.mean_period_total(rng, m),
        base_total=base_total,
        seed_base=SEED_BASE,
        n_scenarios=args.n,
    )

    # ── Persist ─────────────────────────────────────────────────────────────
    season = fitted["stop"]["season"].set_index(["layer", "month"])["season"]
    season_drop = fitted["drop"]["season"].set_index(["layer", "month"])["season"]

    params = {
        "version": SHAPE_PARAMS_VERSION,
        "generated_on": date.today().isoformat(),
        "n_periods": N_PERIODS,
        "base_year": fitted["base_year"],
        "pixels": pixels,
        "excluded_year_months": [list(ym) for ym in EXCLUDED_YEAR_MONTHS],
        "layer_drop_threshold": LAYER_DROP_THRESHOLD,
        "seed_base": SEED_BASE,
        "calibrated_n_scenarios": args.n,
        "spatial": {
            "n_pixels_correlogram": int(len(balanced)),
            "model": spatial["model"],
            "rho_km": spatial["rho_km"],
            "nugget": spatial["nugget"],
            "plateau": spatial["plateau"],
            "weighted_r2": spatial["r2"],
            "candidates": [{k: v for k, v in c.items()} for c in spatial["candidates"]],
            "correlogram": correlogram.to_dict(orient="records"),
        },
        "stop": {
            "level": fitted["stop"]["level"].reindex(pixels).round(6).to_dict(),
            "trend": {str(k): round(v, 6) for k, v in fitted["stop"]["trend"].items()},
            "season": {f"{layer}|{month}": round(v, 6) for (layer, month), v in season.items()},
            "sigma_common": fitted["stop"]["sigma_common"],
            "sigma": fitted["stop"]["sigma"]["sigma"].reindex(pixels).round(6).to_dict(),
        },
        "drop": {
            "level": fitted["drop"]["level"].reindex(pixels).round(6).to_dict(),
            "trend": {str(k): round(v, 6) for k, v in fitted["drop"]["trend"].items()},
            "season": {f"{layer}|{month}": round(v, 6) for (layer, month), v in season_drop.items()},
            "sigma_common": fitted["drop"]["sigma_common"],
            "sigma": fitted["drop"]["sigma"]["sigma"].reindex(pixels).round(6).to_dict(),
        },
        "corr_common_stop_drop": fitted["corr_common_stop_drop"],
        "regime_scaling": {
            "stop_exponent": REGIME_STOP_EXPONENT,
            "drop_exponent": REGIME_DROP_EXPONENT,
        },
        "size_class": fitted["size_class"].reindex(pixels).to_dict(),
        "regimes": {
            regime: {
                "target_period_demand": REGIME_TARGETS[regime],
                "multiplier": result["multiplier"],
                "realized_period_demand": result["realized"],
            }
            for regime, result in regimes.items()
        },
        "base_period_demand": base_total,
        "crosswalk_share_mean": float(panel["crosswalk_share"].mean()),
    }

    PATH_SHAPE_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH_SHAPE_PARAMS, "w") as file:
        json.dump(params, file, indent=2)
    logger.info(f"Shape parameters written to {PATH_SHAPE_PARAMS}")

    print("\nRegímenes calibrados:")
    for regime, result in regimes.items():
        print(
            f"  {regime:8s} objetivo={REGIME_TARGETS[regime]:>9,.0f} "
            f"multiplicador={result['multiplier']:6.4f}  realizado={result['realized']:>9,.0f}"
        )


if __name__ == "__main__":
    main()
