"""Spatially correlated demand scenario generator.

Three levels per period, following the fitted structure in `fitting/marginals.py` and the
spatial dependence in `spatial.py`:

    level 1  F_t          common shock shared by every pixel in the period
    level 2  dev_j        pixel deviation, spatially correlated via a Gaussian copula
    level 3  nugget       the purely local component, folded into the copula's
                          correlation matrix as `corr(0+) = 1 - nugget`

    stop[j,t]   = max(1, round( E_stop[j,t] * multiplier**0.70 * F_t * dev_j ))
    drop[j,t]   = E_drop[j,t] * multiplier**0.30 * G_t * dev'_j
    demand[j,t] = stop[j,t] * drop[j,t]

How `F_t` and `dev_j` are drawn is the generator's `DependenceStrategy`
(`dependence.py`): the spatial copula above, an independent baseline, or a historical
bootstrap.

All multiplicative shocks are mean-preserving (`exp(xi - sigma^2/2)`), so amplifying
dispersion does not move the level. The previous procedure used `mu * exp(xi)`,
which inflated the mean by up to 79% in December while claiming to preserve it.

Discreteness enters only through the rounding of `stop`, whose effect on the
realized correlogram is measured in the analysis report.
"""

import numpy as np
import pandas as pd

from src.core.constants import N_PERIODS, REGIME_DROP_EXPONENT, REGIME_STOP_EXPONENT
from src.core.contract import assert_contract
from src.scenarios.fitting.marginals import expected_matrix
from src.scenarios.generation.dependence import DependenceStrategy, HistoricalBootstrap, Independent, MeanShocks, SpatialJoint


class ScenarioGenerator:
    """Draws demand scenarios that respect the scenario contract.

    The contract is narrow: each pixel must carry `id_pixel`, `stop`, `drop` and
    `demand` arrays of length `N_PERIODS`, with `stop >= 1` and `drop > 0`, because
    the CA only writes cost keys for `demand > 0` and the uncapacitated model
    indexes them directly.

    `floor_hits` / `cells_drawn` count how often the `max(1, .)` floor binds. They
    accumulate across draws; a set writer resets them before each set.
    """

    def __init__(
        self,
        pixels: list[str],
        expected_stop: np.ndarray,
        expected_drop: np.ndarray,
        sigma_stop: np.ndarray,
        sigma_drop: np.ndarray,
        sigma_common_stop: float,
        sigma_common_drop: float,
        dependence: DependenceStrategy,
        regime_stop_exponent: float = REGIME_STOP_EXPONENT,
        regime_drop_exponent: float = REGIME_DROP_EXPONENT,
    ):  # pylint: disable=too-many-arguments
        self.pixels = list(pixels)
        self.expected_stop = expected_stop
        self.expected_drop = expected_drop
        self.sigma_stop = sigma_stop
        self.sigma_drop = sigma_drop
        self.sigma_common_stop = sigma_common_stop
        self.sigma_common_drop = sigma_common_drop
        self.dependence = dependence
        if not np.isclose(regime_stop_exponent + regime_drop_exponent, 1.0):
            raise ValueError("Regime stop/drop exponents must add to 1.")
        self.regime_stop_exponent = regime_stop_exponent
        self.regime_drop_exponent = regime_drop_exponent
        self.floor_hits = 0
        self.cells_drawn = 0

    @classmethod
    def from_fit(cls, fitted: dict, pixels: list[str], dependence: DependenceStrategy) -> "ScenarioGenerator":
        """Build a generator from the fitted marginals, aligned on `pixels`."""
        sigma_stop = fitted["stop"]["sigma"]["sigma"].reindex(pixels).to_numpy(dtype=float)
        sigma_drop = fitted["drop"]["sigma"]["sigma"].reindex(pixels).to_numpy(dtype=float)
        if not np.isfinite(sigma_stop).all() or not np.isfinite(sigma_drop).all():
            missing = [p for p, s in zip(pixels, sigma_stop) if not np.isfinite(s)]
            raise ValueError(f"No dispersion fitted for pixels: {missing[:5]}")

        return cls(
            pixels=pixels,
            expected_stop=expected_matrix(fitted, "stop", pixels),
            expected_drop=expected_matrix(fitted, "drop", pixels),
            sigma_stop=sigma_stop,
            sigma_drop=sigma_drop,
            sigma_common_stop=fitted["stop"]["sigma_common"],
            sigma_common_drop=fitted["drop"]["sigma_common"],
            dependence=dependence,
        )

    @classmethod
    def from_params(cls, params: dict, method: str = "spatial_joint", panel: pd.DataFrame | None = None) -> "ScenarioGenerator":
        """Rebuild a generator from persisted shape parameters.

        This is the reproducible path: given the parameter file and the tracked raw
        pixel inputs, the scenarios regenerate bit-for-bit without the raw demand
        file or the panel. The one exception is `method="historical_bootstrap"`,
        which resamples the historical `panel` by construction.
        """
        from src.scenarios.spatial import cholesky_factor, correlation_matrix, haversine_matrix, pixel_centroids

        pixels = list(params["pixels"])
        expectation = {}
        for quantity in ("stop", "drop"):
            part = params[quantity]
            matrix = np.zeros((len(pixels), N_PERIODS))
            for i, id_pixel in enumerate(pixels):
                layer = id_pixel.split("-")[0]
                level = part["level"][id_pixel]
                for t in range(N_PERIODS):
                    matrix[i, t] = np.exp(level + part["season"][f"{layer}|{t + 1}"])
            expectation[quantity] = matrix

        scaling = params.get("regime_scaling")
        if scaling is None:
            raise ValueError("shape_params.json has no regime_scaling; run recalibrate_regimes first.")

        if method == "spatial_joint":
            centroids = pixel_centroids().set_index("id_pixel").reindex(pixels)
            distances = haversine_matrix(centroids["lon"].to_numpy(), centroids["lat"].to_numpy())
            spatial = params["spatial"]
            sigma = correlation_matrix(distances, spatial["rho_km"], spatial["nugget"], spatial.get("plateau", 0.0))
            dependence: DependenceStrategy = SpatialJoint(cholesky_factor(sigma))
        elif method == "independent":
            dependence = Independent()
        elif method == "historical_bootstrap":
            if panel is None:
                raise ValueError("historical_bootstrap resamples the historical panel; pass panel=.")
            dependence = HistoricalBootstrap.from_panel(panel, pixels, expectation["stop"], expectation["drop"])
        else:
            raise ValueError("method must be independent, spatial_joint or historical_bootstrap")

        return cls(
            pixels=pixels,
            expected_stop=expectation["stop"],
            expected_drop=expectation["drop"],
            sigma_stop=np.array([params["stop"]["sigma"][p] for p in pixels], dtype=float),
            sigma_drop=np.array([params["drop"]["sigma"][p] for p in pixels], dtype=float),
            sigma_common_stop=params["stop"]["sigma_common"],
            sigma_common_drop=params["drop"]["sigma_common"],
            dependence=dependence,
            regime_stop_exponent=float(scaling["stop_exponent"]),
            regime_drop_exponent=float(scaling["drop_exponent"]),
        )

    def draw(
        self, rng: np.random.Generator, multiplier: float, shocks: DependenceStrategy | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one scenario, `(stop, drop)` of shape `(n_pixels, N_PERIODS)`.

        `shocks` overrides the generator's own dependence model for this draw, e.g.
        `MeanShocks()` for the deterministic expected scenario.
        """
        shocks = shocks or self.dependence
        n_pixels = len(self.pixels)
        stop = np.zeros((n_pixels, N_PERIODS), dtype=int)
        drop = np.zeros((n_pixels, N_PERIODS), dtype=float)

        for t in range(N_PERIODS):
            period = shocks.period_shocks(self, rng, t)
            raw_stop = self.expected_stop[:, t] * multiplier**self.regime_stop_exponent * period.f_stop * period.dev_stop
            rounded = np.rint(raw_stop).astype(int)
            self.floor_hits += int((rounded < 1).sum())
            self.cells_drawn += n_pixels
            stop[:, t] = np.maximum(1, rounded)
            drop[:, t] = self.expected_drop[:, t] * multiplier**self.regime_drop_exponent * period.f_drop * period.dev_drop

        return stop, drop

    def mean_period_total(self, rng: np.random.Generator, multiplier: float) -> float:
        """Mean per-period model demand of one drawn scenario."""
        stop, drop = self.draw(rng, multiplier)
        return float((stop * drop).sum() / N_PERIODS)

    def base_period_total(self) -> float:
        """Per-period model demand with every shock at its mean and no regime scaling."""
        return float((self.expected_stop * self.expected_drop).sum() / N_PERIODS)

    def to_payload(self, stop: np.ndarray, drop: np.ndarray, id_scenario, scenario_type: str) -> dict:
        """Assemble the scenario JSON payload, enforcing the contract invariants."""
        assert_contract(stop, drop)
        pixels = []
        for i, id_pixel in enumerate(self.pixels):
            stop_row = [int(v) for v in stop[i]]
            drop_row = [round(float(v), 4) for v in drop[i]]
            pixels.append(
                {
                    "id_pixel": id_pixel,
                    "stop": stop_row,
                    "drop": drop_row,
                    "demand": [round(s * d, 4) for s, d in zip(stop_row, drop_row)],
                }
            )
        return {"id_scenario": id_scenario, "type": scenario_type, "pixels": pixels}

    def annual_expected_payload(self, id_scenario: str, multiplier: float) -> dict:
        """One-period annual mean of the deterministic expected scenario, outside the 12-period contract."""
        stop, drop = self.draw(np.random.default_rng(0), multiplier, shocks=MeanShocks())
        annual_stop = stop.mean(axis=1)
        annual_demand = (stop * drop).mean(axis=1)
        annual_drop = annual_demand / annual_stop
        pixels = [
            {
                "id_pixel": id_pixel,
                "stop": [round(float(annual_stop[i]), 6)],
                "drop": [round(float(annual_drop[i]), 6)],
                "demand": [round(float(annual_demand[i]), 6)],
            }
            for i, id_pixel in enumerate(self.pixels)
        ]
        return {
            "id_scenario": id_scenario,
            "type": "annual_expected",
            "periods": 1,
            "source": "mean over the 12 periods of the expected scenario",
            "optimization_compatible": True,
            "pixels": pixels,
        }
