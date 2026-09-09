"""Marginal structure of the demand panel: level, seasonality and dispersion.

Each quantity (`n_customers` -> `stop`, and `drop`) is decomposed multiplicatively,
which is additive in logs:

    log q[j, y, m] = level[j] + trend[y] + season[layer(j), m] + residual

The residual is then split into the part shared by every pixel in a period and the
part specific to each pixel:

    residual[j, y, m] = common[y, m] + deviation[j, y, m]

`common` is what the scenario generator's period factor reproduces (and what the
demand regimes shift); `deviation` is what the spatial copula correlates. Splitting
them matters — the current procedure has neither, so aggregate demand averages its
noise away across 161 pixels.

Seasonality is pooled by layer and dispersion is pooled by pixel size class,
because each `(pixel, month)` cell holds only three yearly observations.
"""

import numpy as np
import pandas as pd

from src.constants import N_PERIODS
from src.utils.custom_logger import get_logger

logger = get_logger("Marginals")

BASE_YEAR = 2022
N_SIZE_CLASSES = 4


def _decompose_logs(df: pd.DataFrame, value: str, max_iter: int = 200, tol: float = 1e-10) -> dict:
    """Additive decomposition of `log(value)` into pixel, year and layer-month effects.

    Fitted by alternating means (Kruskal's iterative proportional fitting in logs),
    which converges for this balanced-enough panel and needs no extra dependency.
    Normalization: `trend[BASE_YEAR] = 0` and the layer-month effects average to 0,
    so `level[j]` reads as the pixel's base-year log level.
    """
    work = df[["id_pixel", "layer", "year", "month", value]].copy()
    work = work[work[value] > 0]
    y = np.log(work[value].to_numpy(dtype=float))

    pix_codes, pix_index = pd.factorize(work["id_pixel"])
    yr_codes, yr_index = pd.factorize(work["year"])
    lm_key = work["layer"].astype(str) + "|" + work["month"].astype(str)
    lm_codes, lm_index = pd.factorize(lm_key)

    level = np.zeros(len(pix_index))
    trend = np.zeros(len(yr_index))
    season = np.zeros(len(lm_index))

    def group_mean(residual, codes, size):
        total = np.bincount(codes, weights=residual, minlength=size)
        count = np.bincount(codes, minlength=size)
        return np.divide(total, count, out=np.zeros(size), where=count > 0)

    for it in range(max_iter):
        prev = np.concatenate([level, trend, season])
        level = group_mean(y - trend[yr_codes] - season[lm_codes], pix_codes, len(pix_index))
        trend = group_mean(y - level[pix_codes] - season[lm_codes], yr_codes, len(yr_index))
        season = group_mean(y - level[pix_codes] - trend[yr_codes], lm_codes, len(lm_index))
        if np.max(np.abs(np.concatenate([level, trend, season]) - prev)) < tol:
            break
    logger.info(f"{value}: log decomposition converged in {it + 1} iterations.")

    # Normalization 1: the base year carries no trend, so `level` reads as the
    # base-year log level.
    base_pos = list(yr_index).index(BASE_YEAR)
    level = level + trend[base_pos]
    trend = trend - trend[base_pos]

    season_df = pd.DataFrame(
        {
            "layer": [k.split("|")[0] for k in lm_index],
            "month": [int(k.split("|")[1]) for k in lm_index],
            "season": season,
        }
    )

    # Normalization 2: each layer's seasonal effects average to zero, with the
    # removed mean absorbed into that layer's pixel levels.
    layer_centre = season_df.groupby("layer")["season"].mean()
    season_df["season"] -= season_df["layer"].map(layer_centre)
    layer_of_pixel = work.drop_duplicates("id_pixel").set_index("id_pixel")["layer"]
    level = level + np.array([layer_centre.get(layer_of_pixel[p], 0.0) for p in pix_index])

    season_lookup = season_df.set_index(["layer", "month"])["season"]
    season_per_row = season_lookup.reindex(
        pd.MultiIndex.from_arrays([work["layer"].to_numpy(), work["month"].to_numpy()])
    ).to_numpy()
    residual = y - (level[pix_codes] + trend[yr_codes] + season_per_row)

    return {
        "level": pd.Series(level, index=pix_index, name="level"),
        "trend": pd.Series(trend, index=yr_index, name="trend"),
        "season": season_df,
        "residual": pd.Series(residual, index=work.index, name="residual"),
        "frame": work,
    }


def _split_residual(work: pd.DataFrame, residual: pd.Series) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Split the residual into a per-period common shock and pixel deviations.

    Returns the common shock, the full deviation matrix (used for per-pixel
    dispersion, where a pixel with a few missing months is still informative), and
    a balanced version restricted to pixels observed in every period and
    re-centred on that subset.

    The balanced matrix is what the correlogram must use. The panel is unbalanced
    (13 of 161 pixels miss some months), and centring on all pixels while
    correlating only the complete ones leaves the column sums non-zero, which
    inflates the estimated long-range correlation — here from 0.027 to 0.047.
    """
    tmp = work.assign(residual=residual.to_numpy())
    common = tmp.groupby(["year", "month"])["residual"].mean()
    tmp["deviation"] = tmp["residual"] - tmp.set_index(["year", "month"]).index.map(common)
    deviations = tmp.pivot_table(index="id_pixel", columns=["year", "month"], values="deviation")

    balanced = deviations.dropna(axis=0, how="any")
    balanced = balanced.sub(balanced.mean(axis=0), axis=1)
    if len(balanced) < len(deviations):
        logger.info(
            f"Correlogram will use {len(balanced)} of {len(deviations)} pixels "
            f"(dropped {len(deviations) - len(balanced)} with incomplete history) and re-centre on them."
        )
    return common, deviations, balanced


def _size_classes(panel: pd.DataFrame) -> pd.Series:
    """Assign each pixel to a size class by mean active customers."""
    size = panel.groupby("id_pixel")["n_customers"].mean()
    ranks = size.rank(pct=True)
    return pd.Series(np.ceil(ranks * N_SIZE_CLASSES).clip(1, N_SIZE_CLASSES).astype(int), index=size.index, name="size_class")


def _pooled_sigma(deviations: pd.DataFrame, size_class: pd.Series) -> pd.DataFrame:
    """Per-pixel dispersion, pooled to the class median for stability.

    Each pixel has only ~35 monthly observations, so its own standard deviation is
    noisy; the class median is used for generation and the pixel's own value is
    kept for reporting.
    """
    own = deviations.std(axis=1, ddof=1)
    df = pd.DataFrame({"sigma_own": own})
    df["size_class"] = size_class.reindex(df.index)
    df["sigma"] = df.groupby("size_class")["sigma_own"].transform("median")
    return df


def fit_marginals(panel: pd.DataFrame) -> dict:
    """Fit the level/seasonality/dispersion structure for `stop` and `drop`."""
    included = panel[~panel["excluded"]].copy()
    logger.info(f"Fitting on {len(included):,} pixel-months ({included['id_pixel'].nunique()} pixels).")

    size_class = _size_classes(included)
    out: dict = {"size_class": size_class, "base_year": BASE_YEAR}

    for quantity, column in (("stop", "n_customers"), ("drop", "drop")):
        dec = _decompose_logs(included, column)
        common, deviations, balanced = _split_residual(dec["frame"], dec["residual"])
        sigma = _pooled_sigma(deviations, size_class)

        sigma_common = float(common.std(ddof=1))
        logger.info(
            f"{quantity}: sigma_common={sigma_common:.4f} | "
            f"sigma_deviation median={sigma['sigma'].median():.4f} "
            f"(class medians {sorted(sigma.groupby('size_class')['sigma'].first().round(3).tolist())})"
        )

        out[quantity] = {
            "level": dec["level"],
            "trend": dec["trend"],
            "season": dec["season"],
            "sigma_common": sigma_common,
            "sigma": sigma,
            "common": common,
            "deviations": deviations,
            "deviations_balanced": balanced,
        }

    corr_common = float(np.corrcoef(out["stop"]["common"].to_numpy(), out["drop"]["common"].to_numpy())[0, 1])
    out["corr_common_stop_drop"] = corr_common
    logger.info(f"Correlation between the stop and drop common shocks: {corr_common:.3f}")
    return out


def expected_matrix(fitted: dict, quantity: str, pixels: list[str]) -> np.ndarray:
    """Base-year expectation of `quantity` for each pixel and period.

    Shape `(len(pixels), N_PERIODS)`. This is the deterministic backbone the
    generator perturbs; `trend[BASE_YEAR]` is 0 by construction, so the level is
    the base year's.
    """
    part = fitted[quantity]
    season = part["season"].set_index(["layer", "month"])["season"]
    levels = part["level"]

    out = np.zeros((len(pixels), N_PERIODS))
    for i, id_pixel in enumerate(pixels):
        layer = id_pixel.split("-")[0]
        base = levels.get(id_pixel, np.nan)
        for t in range(N_PERIODS):
            out[i, t] = np.exp(base + season.get((layer, t + 1), 0.0))
    return out
