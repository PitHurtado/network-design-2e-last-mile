"""`ParamsFitter`: panel -> `ShapeParams`.

Fits pixel levels, layer seasonality, dispersion and the spatial decay from the monthly
panel, then calibrates the regime multipliers by simulation. `recalibrate` redoes only
the last step on persisted parameters, for when the regime policy changes but the
history does not need to be refit.
"""

from dataclasses import dataclass

import pandas as pd

from src.core.constants import (
    EXCLUDED_YEAR_MONTHS,
    LAYER_DROP_THRESHOLD,
    N_PERIODS,
    REGIME_DROP_EXPONENT,
    REGIME_STOP_EXPONENT,
    REGIME_TARGETS,
    SEED_BASE,
)
from src.scenarios.fitting.marginals import fit_marginals
from src.scenarios.fitting.regimes import RegimeCalibrator
from src.scenarios.generation.dependence import SpatialJoint
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.generation.seeds import set_seeds
from src.scenarios.params import ShapeParams
from src.scenarios.spatial import (
    cholesky_factor,
    correlation_matrix,
    empirical_correlogram,
    fit_correlogram,
    haversine_matrix,
    pixel_centroids,
)
from src.tools.logging import get_logger

logger = get_logger("FitParams")

SHAPE_PARAMS_VERSION = 3


@dataclass
class FitResult:
    """The fitted parameters plus what is worth printing about how they were obtained."""

    params: ShapeParams
    correlogram: pd.DataFrame
    spatial: dict
    regimes: dict


class ParamsFitter:
    """Fits `ShapeParams` from the monthly panel.

    `n_calibration` is the number of validation streams the regime multipliers are tuned
    on (the validation set's size, so its scenarios hit the targets exactly); `bins` the
    number of distance bins of the empirical correlogram.
    """

    def __init__(self, bins: int = 12, n_calibration: int = 100, seed_base: int = SEED_BASE):
        self.bins = bins
        self.n_calibration = n_calibration
        self.seed_base = seed_base

    def fit(self, panel: pd.DataFrame, grid_pixels: set[str]) -> FitResult:
        # The pixel set must match input_pixels.xlsx exactly: a pixel present in the
        # scenario but missing from the grid is dropped by the input reader.
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

        # ── Spatial structure ────────────────────────────────────────────────
        centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
        distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())
        # The correlogram uses the balanced, re-centred deviations; centring on all
        # pixels while correlating only the complete ones inflates the long-range term.
        balanced = fitted["stop"]["deviations_balanced"]
        balanced_centroids = centroids.reindex(balanced.index)
        balanced_distances = haversine_matrix(balanced_centroids["lon"].to_numpy(), balanced_centroids["lat"].to_numpy())
        correlogram = empirical_correlogram(balanced, balanced_distances, n_bins=self.bins)
        spatial = fit_correlogram(correlogram)

        sigma = correlation_matrix(distances, spatial["rho_km"], spatial["nugget"], spatial["plateau"])
        chol = cholesky_factor(sigma)

        base_total = ScenarioGenerator.from_fit(fitted, pixels, SpatialJoint(chol)).base_period_total()
        logger.info(f"Base per-period model demand (no regime scaling): {base_total:,.0f}")

        # ── Persisted form ───────────────────────────────────────────────────
        season = fitted["stop"]["season"].set_index(["layer", "month"])["season"]
        season_drop = fitted["drop"]["season"].set_index(["layer", "month"])["season"]

        params = {
            "version": SHAPE_PARAMS_VERSION,
            "n_periods": N_PERIODS,
            "base_year": fitted["base_year"],
            "pixels": pixels,
            "excluded_year_months": [list(ym) for ym in EXCLUDED_YEAR_MONTHS],
            "layer_drop_threshold": LAYER_DROP_THRESHOLD,
            "seed_base": self.seed_base,
            "calibrated_n_scenarios": self.n_calibration,
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
            "stop": self._quantity(fitted["stop"], season, pixels),
            "drop": self._quantity(fitted["drop"], season_drop, pixels),
            "corr_common_stop_drop": fitted["corr_common_stop_drop"],
            "regime_scaling": {
                "stop_exponent": REGIME_STOP_EXPONENT,
                "drop_exponent": REGIME_DROP_EXPONENT,
            },
            "size_class": fitted["size_class"].reindex(pixels).to_dict(),
            "regimes": None,  # calibrated below, on the persisted form
            "base_period_demand": base_total,
            "crosswalk_share_mean": float(panel["crosswalk_share"].mean()),
        }
        # ── Regime calibration ───────────────────────────────────────────────
        # On the generator the scenarios will actually be drawn from (rebuilt from the
        # rounded, persisted parameters) and on the validation streams they will consume,
        # so the written validation sets hit their targets exactly.
        regimes = self._calibrate(ShapeParams(params), self.n_calibration)
        params["regimes"] = self._regimes_block(regimes)
        return FitResult(ShapeParams(params), correlogram, spatial, regimes)

    def _calibrate(self, params: ShapeParams, n_scenarios: int) -> dict:
        generator = ScenarioGenerator.from_params(params)
        return RegimeCalibrator(generator, set_seeds(self.seed_base, "validation", n_scenarios)).calibrate_all()

    @staticmethod
    def _quantity(part: dict, season: pd.Series, pixels: list[str]) -> dict:
        return {
            "level": part["level"].reindex(pixels).round(6).to_dict(),
            "trend": {str(k): round(v, 6) for k, v in part["trend"].items()},
            "season": {f"{layer}|{month}": round(v, 6) for (layer, month), v in season.items()},
            "sigma_common": part["sigma_common"],
            "sigma": part["sigma"]["sigma"].reindex(pixels).round(6).to_dict(),
        }

    @staticmethod
    def _regimes_block(regimes: dict) -> dict:
        return {
            regime: {
                "target_period_demand": REGIME_TARGETS[regime],
                "multiplier": result["multiplier"],
                "realized_period_demand": result["realized"],
            }
            for regime, result in regimes.items()
        }

    def recalibrate(self, params: ShapeParams, validation_n: int = 100) -> tuple[ShapeParams, dict]:
        """New regime multipliers for persisted parameters, calibrated on the validation seeds."""
        raw = dict(params.raw)
        raw["regime_scaling"] = {
            "stop_exponent": REGIME_STOP_EXPONENT,
            "drop_exponent": REGIME_DROP_EXPONENT,
        }
        regimes = self._calibrate(ShapeParams(raw), validation_n)
        raw["version"] = max(int(raw.get("version", 0)), 3)
        raw.pop("generated_on", None)
        raw["seed_base"] = self.seed_base
        raw["calibrated_n_scenarios"] = validation_n
        raw["regimes"] = self._regimes_block(regimes)
        return ShapeParams(raw), regimes
