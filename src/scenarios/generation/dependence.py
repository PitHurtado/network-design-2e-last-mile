"""How the per-period shocks of a scenario are drawn: one strategy per dependence model.

Each strategy returns, for one period, the common factors `F_t` / `G_t` and the pixel
deviations `dev_j` that `ScenarioGenerator.draw` multiplies into the expected demand.
All random shocks are mean-preserving (`exp(xi - sigma^2/2)`).

The order in which a strategy consumes the random stream is part of the contract, because
scenarios must regenerate bit-for-bit from their seed:

    SpatialJoint         normal(stop), normal(drop), std_normal(n) stop, std_normal(n) drop
    Independent          std_normal(n) stop, std_normal(n) drop
    HistoricalBootstrap  one integers() draw, shared by stop and drop
    MeanShocks           nothing
    MedianShocks         nothing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from src.core.constants import N_PERIODS

if TYPE_CHECKING:
    from src.scenarios.generation.generator import ScenarioGenerator


@dataclass
class PeriodShocks:
    """Multiplicative shocks of one period: common factors and per-pixel deviations."""

    f_stop: float
    f_drop: float
    dev_stop: np.ndarray
    dev_drop: np.ndarray


class DependenceStrategy(ABC):
    """Draws the shocks of one period for a generator's pixels."""

    name: ClassVar[str]

    @abstractmethod
    def period_shocks(self, generator: "ScenarioGenerator", rng: np.random.Generator, period: int) -> PeriodShocks:
        """Shocks for `period` (0-based), consuming `rng` in the documented order."""


class SpatialJoint(DependenceStrategy):
    """Common period factor plus pixel deviations correlated through a Gaussian copula.

    Because the marginals are lognormal, the copula reduces exactly to a multivariate
    lognormal. The nugget is folded into the correlation matrix as `corr(0+) = 1 - nugget`.
    """

    name = "spatial_joint"

    def __init__(self, cholesky: np.ndarray):
        self.cholesky = cholesky

    def period_shocks(self, generator, rng, period):
        n_pixels = len(generator.pixels)
        sigma_common_stop, sigma_common_drop = generator.sigma_common_stop, generator.sigma_common_drop
        f_stop = float(np.exp(rng.normal(0.0, sigma_common_stop) - 0.5 * sigma_common_stop**2))
        f_drop = float(np.exp(rng.normal(0.0, sigma_common_drop) - 0.5 * sigma_common_drop**2))
        field_stop = self.cholesky @ rng.standard_normal(n_pixels)
        dev_stop = np.exp(generator.sigma_stop * field_stop - 0.5 * generator.sigma_stop**2)
        field_drop = self.cholesky @ rng.standard_normal(n_pixels)
        dev_drop = np.exp(generator.sigma_drop * field_drop - 0.5 * generator.sigma_drop**2)
        return PeriodShocks(f_stop, f_drop, dev_stop, dev_drop)


class Independent(DependenceStrategy):
    """Strict baseline: no common factor and no spatial correlation.

    The common factor's variance is folded into each pixel's own shock, so the marginal
    dispersion stays comparable with the spatial model.
    """

    name = "independent"

    def period_shocks(self, generator, rng, period):
        n_pixels = len(generator.pixels)
        total_sigma_stop = np.sqrt(generator.sigma_stop**2 + generator.sigma_common_stop**2)
        total_sigma_drop = np.sqrt(generator.sigma_drop**2 + generator.sigma_common_drop**2)
        dev_stop = np.exp(total_sigma_stop * rng.standard_normal(n_pixels) - 0.5 * total_sigma_stop**2)
        dev_drop = np.exp(total_sigma_drop * rng.standard_normal(n_pixels) - 0.5 * total_sigma_drop**2)
        return PeriodShocks(1.0, 1.0, dev_stop, dev_drop)


class HistoricalBootstrap(DependenceStrategy):
    """Resample a whole historical month: the joint vector of pixel shocks it produced.

    `stop_shocks[t]` and `drop_shocks[t]` have shape `(n_pixels, n_years_observed)`; one
    column is drawn per period and used for both quantities, so their joint pattern holds.
    """

    name = "historical_bootstrap"

    def __init__(self, stop_shocks: list[np.ndarray], drop_shocks: list[np.ndarray], n_pixels: int):
        if len(stop_shocks) != N_PERIODS or len(drop_shocks) != N_PERIODS:
            raise ValueError(f"Bootstrap shocks must contain {N_PERIODS} calendar periods.")
        for shocks in (stop_shocks, drop_shocks):
            if any(np.asarray(values).ndim != 2 or np.asarray(values).shape[0] != n_pixels for values in shocks):
                raise ValueError("Each bootstrap shock matrix must have shape (n_pixels, n_historical_draws).")
        self.stop_shocks = [np.asarray(values, dtype=float) for values in stop_shocks]
        self.drop_shocks = [np.asarray(values, dtype=float) for values in drop_shocks]

    @classmethod
    def from_panel(
        cls, panel: pd.DataFrame, pixels: list[str], expected_stop: np.ndarray, expected_drop: np.ndarray
    ) -> "HistoricalBootstrap":
        """Month-specific joint shocks `log(observed / expected)` from the historical panel.

        Missing pixel-months receive a neutral shock, while each pixel's observed shocks
        are centred and mean-preserved before bootstrap sampling.
        """
        included = panel[~panel["excluded"]].copy()
        stop_out, drop_out = [], []
        for month in range(1, N_PERIODS + 1):
            month_frame = included[included["month"] == month]
            years = sorted(month_frame["year"].unique())
            stop_shocks = np.zeros((len(pixels), len(years)), dtype=float)
            drop_shocks = np.zeros((len(pixels), len(years)), dtype=float)
            for i, id_pixel in enumerate(pixels):
                rows = month_frame[month_frame["id_pixel"] == id_pixel].set_index("year")
                for j, year in enumerate(years):
                    if year not in rows.index:
                        continue
                    observed_stop = max(float(rows.loc[year, "n_customers"]), 1.0)
                    observed_drop = max(float(rows.loc[year, "drop"]), 1e-9)
                    stop_shocks[i, j] = np.log(observed_stop / expected_stop[i, month - 1])
                    drop_shocks[i, j] = np.log(observed_drop / expected_drop[i, month - 1])
            for shocks in (stop_shocks, drop_shocks):
                centre = shocks.mean(axis=1, keepdims=True)
                variance = shocks.var(axis=1, ddof=1, keepdims=True) if shocks.shape[1] > 1 else np.zeros((len(pixels), 1))
                shocks -= centre
                shocks -= 0.5 * variance
            stop_out.append(stop_shocks)
            drop_out.append(drop_shocks)
        return cls(stop_out, drop_out, n_pixels=len(pixels))

    def period_shocks(self, generator, rng, period):
        stop_values = self.stop_shocks[period]
        drop_values = self.drop_shocks[period]
        index = int(rng.integers(stop_values.shape[1]))
        return PeriodShocks(1.0, 1.0, np.exp(stop_values[:, index]), np.exp(drop_values[:, index]))


class MeanShocks(DependenceStrategy):
    """Every multiplicative factor at its mean (1.0): the deterministic expected scenario."""

    name = "mean"

    def period_shocks(self, generator, rng, period):
        n_pixels = len(generator.pixels)
        return PeriodShocks(1.0, 1.0, np.ones(n_pixels), np.ones(n_pixels))


class MedianShocks(DependenceStrategy):
    """Every multiplicative factor at its median, `exp(-sigma^2/2)`."""

    name = "median"

    def period_shocks(self, generator, rng, period):
        return PeriodShocks(
            np.exp(-0.5 * generator.sigma_common_stop**2),
            np.exp(-0.5 * generator.sigma_common_drop**2),
            np.exp(-0.5 * generator.sigma_stop**2),
            np.exp(-0.5 * generator.sigma_drop**2),
        )


# Random dependence models a scenario set can be generated with, by their on-disk name.
RANDOM_STRATEGIES = {cls.name: cls for cls in (Independent, SpatialJoint, HistoricalBootstrap)}
