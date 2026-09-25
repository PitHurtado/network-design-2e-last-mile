"""Calibration of the demand regimes.

A regime is a scalar multiplier on the customer counts, chosen so the expected
per-period model demand `sum_j(stop_j * drop_j)` matches a target. The targets are
the p10 / p50 / p90 of the per-period totals observed in the panel, so each regime
corresponds to a demand level that actually occurred.

The multiplier is not simply `target / base`, because `stop` is rounded to an
integer and floored at 1. Those two operations bias the realized total upward,
most visibly in small pixels, so the multiplier is refined by simulation.
"""

import numpy as np

from src.core.constants import REGIME_TARGETS
from src.scenarios.generation.generator import ScenarioGenerator
from src.scenarios.generation.seeds import rng_for
from src.tools.logging import get_logger

logger = get_logger("Regimes")


class RegimeCalibrator:
    """Finds, per regime, the multiplier whose mean per-period total over `seeds` hits the target."""

    def __init__(self, generator: ScenarioGenerator, seeds: list, max_iter: int = 12, tol: float = 1e-4):
        self.generator = generator
        self.seeds = seeds
        self.max_iter = max_iter
        self.tol = tol
        self.base_total = generator.base_period_total()

    def calibrate(self, target: float) -> dict:
        """Iterate a multiplicative correction, which converges in a couple of steps because
        the map is nearly linear once the rounding floor stops binding."""
        multiplier = target / self.base_total
        history = []
        realized = float("nan")
        ratio = float("nan")

        for iteration in range(self.max_iter):
            totals = [self.generator.mean_period_total(rng_for(seed), multiplier) for seed in self.seeds]
            realized = float(np.mean(totals))
            ratio = target / realized
            history.append({"iteration": iteration, "multiplier": multiplier, "realized": realized, "ratio": ratio})

            if abs(ratio - 1.0) < self.tol:
                break
            multiplier *= ratio

        logger.info(
            f"target={target:,.0f} -> multiplier={multiplier:.4f} "
            f"(realized {realized:,.0f}, off by {abs(ratio - 1) * 100:.3f}%, {iteration + 1} iterations)"
        )
        return {"multiplier": multiplier, "realized": realized, "target": target, "history": history}

    def calibrate_all(self, targets: dict[str, float] = REGIME_TARGETS) -> dict:
        return {regime: self.calibrate(target) for regime, target in targets.items()}
