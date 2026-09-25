"""Calibration of the demand regimes.

A regime is a scalar multiplier on the customer counts, chosen so the expected
per-period model demand `sum_j(stop_j * drop_j)` matches a target. The targets are
the p10 / p50 / p90 of the per-period totals observed in the panel, so each regime
corresponds to a demand level that actually occurred.

The multiplier is not simply `target / base`, because `stop` is rounded to an
integer and floored at 1. Those two operations bias the realized total upward,
most visibly in small pixels, so the multiplier is refined by simulation.
"""

from typing import Callable

import numpy as np

from src.core.constants import REGIME_TARGETS
from src.tools.logging import get_logger

logger = get_logger("Regimes")


def spawn_seeds(seed_base: int, n_scenarios: int) -> list:
    """The seed sequence used for generation.

    Calibration and generation must consume the *same* seeds, otherwise the
    multiplier is tuned on one Monte Carlo sample and applied to another, leaving
    the written scenarios a couple of percent off their target.
    """
    return np.random.SeedSequence(seed_base).spawn(n_scenarios)


def calibrate_multiplier(
    target: float,
    total_for: Callable[[float, np.random.Generator], float],
    seeds: list,
    base_total: float,
    max_iter: int = 12,
    tol: float = 1e-4,
) -> dict:
    """Find the multiplier whose mean per-period total over `seeds` hits `target`.

    `total_for(multiplier, rng)` must return the mean per-period model demand of
    one generated scenario. Iterates a multiplicative correction, which converges
    in a couple of steps because the map is nearly linear once the rounding floor
    stops binding.
    """
    multiplier = target / base_total
    history = []
    realized = float("nan")
    ratio = float("nan")

    for iteration in range(max_iter):
        totals = [total_for(multiplier, np.random.Generator(np.random.PCG64(seed))) for seed in seeds]
        realized = float(np.mean(totals))
        ratio = target / realized
        history.append({"iteration": iteration, "multiplier": multiplier, "realized": realized, "ratio": ratio})

        if abs(ratio - 1.0) < tol:
            break
        multiplier *= ratio

    logger.info(
        f"target={target:,.0f} -> multiplier={multiplier:.4f} "
        f"(realized {realized:,.0f}, off by {abs(ratio - 1) * 100:.3f}%, {iteration + 1} iterations)"
    )
    return {"multiplier": multiplier, "realized": realized, "target": target, "history": history}


def calibrate_all(
    total_for: Callable[[float, np.random.Generator], float],
    base_total: float,
    seed_base: int,
    n_scenarios: int,
    seeds: list | None = None,
) -> dict:
    """Calibrate every regime in `REGIME_TARGETS` against the generation seeds."""
    seeds = seeds or spawn_seeds(seed_base, n_scenarios)
    return {
        regime: calibrate_multiplier(target=target, total_for=total_for, seeds=seeds, base_total=base_total)
        for regime, target in REGIME_TARGETS.items()
    }
