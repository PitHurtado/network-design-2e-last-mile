"""Spatial dependence structure for the demand scenarios.

Pixels are not independent: demand in a satellite's catchment rises as a block, so
sampling pixels independently makes the aggregate variance collapse like ~1/sqrt(P)
and hides exactly the risk the facility-location decision should see.

The dependence is imposed with a Gaussian copula whose latent correlation follows
an exponential decay in distance plus a nugget for the purely local component:

    corr(h) = (1 - nugget) * exp(-h / rho)      for h > 0
    corr(0) = 1

`rho` (km) and `nugget` are calibrated against the empirical correlogram of the
panel's pixel-level log deviations, so the structure is estimated, not assumed.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.core.constants import GRID_DLAT, GRID_DLON, GRID_LAT0, GRID_LON0, GRID_N_COLS, GRID_N_ROWS
from src.scenarios.crosswalk import build_footprints, load_manual_crosswalk
from src.tools.logging import get_logger

logger = get_logger("Spatial")

EARTH_RADIUS_KM = 6371.0


def cell_center(cell: int) -> tuple[float, float]:
    """Geometric centre (lon, lat) of a grid cell."""
    row, col = divmod(int(cell), GRID_N_COLS)
    return GRID_LON0 + (col + 0.5) * GRID_DLON, GRID_LAT0 + (row + 0.5) * GRID_DLAT


def pixel_centroids(crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Geometric centroid of each pixel's merged footprint.

    Uses the grid geometry rather than `raw_pixels.csv`'s lon/lat, which are
    demand-weighted representative service points and can fall outside their own
    cell.
    """
    if crosswalk is None:
        crosswalk = load_manual_crosswalk()
    footprint, _ = build_footprints(crosswalk)

    rows = []
    for (layer, pixel), cells in footprint.items():
        centers = np.array([cell_center(c) for c in sorted(cells)])
        rows.append(
            {
                "id_pixel": f"{layer}-{int(pixel)}",
                "layer": layer,
                "pixel": int(pixel),
                "n_cells": len(cells),
                "lon": centers[:, 0].mean(),
                "lat": centers[:, 1].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("id_pixel").reset_index(drop=True)
    logger.info(f"Centroids for {len(out)} pixels; footprint sizes {out['n_cells'].min()}-{out['n_cells'].max()} cells.")
    return out


def pixel_grid_cells(crosswalk: pd.DataFrame | None = None) -> dict[str, set[int]]:
    """Return the actual grid-cell footprint for each model pixel."""
    crosswalk = load_manual_crosswalk() if crosswalk is None else crosswalk
    footprint, _ = build_footprints(crosswalk)
    return {f"{layer}-{int(pixel)}": set(cells) for (layer, pixel), cells in footprint.items()}


def pixel_neighbor_pairs(
    pixels: list[str] | None = None,
    crosswalk: pd.DataFrame | None = None,
    ring: int = 1,
) -> pd.DataFrame:
    """Return pixel pairs whose grid footprints are within a given edge ring.

    A first-ring neighbor shares a grid-cell edge with another pixel footprint.
    The second ring is the graph distance-two neighborhood.  The construction uses
    the regular grid rather than centroid distance, so merged rectangular pixels and
    cross-layer pixels are handled consistently with the input geometry.
    """
    if ring < 1:
        raise ValueError("ring must be >= 1")
    crosswalk = load_manual_crosswalk() if crosswalk is None else crosswalk
    footprint, _ = build_footprints(crosswalk)
    allowed = set(pixels) if pixels is not None else {f"{layer}-{int(pixel)}" for layer, pixel in footprint}
    cell_owners: dict[int, set[str]] = {}
    for (layer, pixel), cells in footprint.items():
        id_pixel = f"{layer}-{int(pixel)}"
        if id_pixel not in allowed:
            continue
        for cell in cells:
            cell_owners.setdefault(int(cell), set()).add(id_pixel)

    adjacency: dict[str, set[str]] = {id_pixel: set() for id_pixel in allowed}
    for id_pixel in allowed:
        # The grid is row-major: +/-1 is horizontal except at row boundaries;
        # +/- GRID_N_COLS is vertical.
        cells = next(
            (cells for (layer, pixel), cells in footprint.items() if f"{layer}-{int(pixel)}" == id_pixel),
            set(),
        )
        for cell in cells:
            row, col = divmod(int(cell), GRID_N_COLS)
            neighbors = []
            if col > 0:
                neighbors.append(cell - 1)
            if col < GRID_N_COLS - 1:
                neighbors.append(cell + 1)
            if row > 0:
                neighbors.append(cell - GRID_N_COLS)
            if row < GRID_N_ROWS - 1:
                neighbors.append(cell + GRID_N_COLS)
            for other_cell in neighbors:
                for other in cell_owners.get(int(other_cell), set()):
                    if other != id_pixel:
                        adjacency[id_pixel].add(other)

    edges = set()
    for source, neighbors in adjacency.items():
        for target in neighbors:
            edges.add(tuple(sorted((source, target))))

    if ring == 2:
        first_ring = {pair for pair in edges}
        second_edges = set(first_ring)
        for source in allowed:
            reached = set(adjacency[source])
            for neighbor in list(reached):
                reached.update(adjacency.get(neighbor, set()))
            for target in reached:
                if target != source:
                    pair = tuple(sorted((source, target)))
                    if pair not in first_ring:
                        second_edges.add(pair)
        edges = second_edges

    rows = [{"id_pixel": left, "neighbor": right, "ring": ring} for left, right in sorted(edges)]
    return pd.DataFrame(rows, columns=["id_pixel", "neighbor", "ring"])


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in km."""
    phi = np.radians(lat)
    lam = np.radians(lon)
    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]
    a = np.sin(dphi / 2) ** 2 + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def empirical_correlogram(
    deviations: pd.DataFrame,
    distances: np.ndarray,
    n_bins: int = 12,
) -> pd.DataFrame:
    """Correlation of pixel deviations binned by pairwise distance.

    Parameters
    ----------
    deviations : DataFrame
        Rows are pixels (index aligned with `distances`), columns are periods.
    distances : ndarray
        Pairwise distance matrix in km.
    """
    values = deviations.to_numpy(dtype=float)
    corr = np.corrcoef(values)

    iu = np.triu_indices(len(values), k=1)
    h = distances[iu]
    c = corr[iu]
    ok = np.isfinite(c)
    h, c = h[ok], c[ok]

    edges = np.quantile(h, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(h, edges[1:-1]), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not sel.any():
            continue
        rows.append({"h_km": h[sel].mean(), "corr": c[sel].mean(), "n_pairs": int(sel.sum())})
    out = pd.DataFrame(rows)
    logger.info(f"Empirical correlogram over {len(h):,} pixel pairs in {len(out)} distance bins.")
    return out


def _weighted_r2(pred: np.ndarray, c: np.ndarray, w: np.ndarray) -> float:
    ss_res = float(np.sum(w * (pred - c) ** 2))
    ss_tot = float(np.sum(w * (c - np.average(c, weights=w)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def model_correlation(h: np.ndarray, rho_km: float, nugget: float, plateau: float = 0.0) -> np.ndarray:
    """Correlation as a function of distance.

        corr(h) = plateau + (1 - nugget - plateau) * exp(-h / rho)

    `plateau` is a distance-independent floor. It is not cosmetic: pixels load on
    the period's common shock with different intensities, and subtracting the
    cross-sectional mean only removes the uniformly-weighted part, so a genuine
    long-range correlation survives in the deviations.
    """
    return plateau + (1.0 - nugget - plateau) * np.exp(-np.asarray(h) / rho_km)


def fit_correlogram(correlogram: pd.DataFrame) -> dict:
    """Fit the correlation decay, choosing between a pure and a plateau model.

    The empirical correlogram of this panel is not monotone — it decays over the
    first few km and then rises again — so a pure exponential underfits. Both
    models are fitted and the better weighted R2 wins; which one was used is
    recorded so the choice is auditable rather than implicit.
    """
    h = correlogram["h_km"].to_numpy(dtype=float)
    c = correlogram["corr"].to_numpy(dtype=float)
    w = np.sqrt(correlogram["n_pairs"].to_numpy(dtype=float))

    def fit(with_plateau: bool):
        if with_plateau:

            def residual(theta):
                return w * (model_correlation(h, theta[0], theta[1], theta[2]) - c)

            guess = [max(h.mean(), 0.5), 0.3, max(float(c[-3:].mean()), 0.0)]
            bounds = ([1e-3, 0.0, 0.0], [1e3, 1.0, 1.0])
        else:

            def residual(theta):
                return w * (model_correlation(h, theta[0], theta[1], 0.0) - c)

            guess = [max(h.mean(), 0.5), 0.3]
            bounds = ([1e-3, 0.0], [1e3, 1.0])

        solution = least_squares(residual, guess, bounds=bounds)
        rho, nugget = float(solution.x[0]), float(solution.x[1])
        plateau = float(solution.x[2]) if with_plateau else 0.0
        pred = model_correlation(h, rho, nugget, plateau)
        return {
            "model": "exponential+plateau" if with_plateau else "exponential",
            "rho_km": rho,
            "nugget": nugget,
            "plateau": plateau,
            "r2": _weighted_r2(pred, c, w),
        }

    candidates = [fit(False), fit(True)]
    for candidate in candidates:
        logger.info(
            f"  {candidate['model']:20s} rho={candidate['rho_km']:6.2f} km "
            f"nugget={candidate['nugget']:.3f} plateau={candidate['plateau']:.3f} R2={candidate['r2']:.3f}"
        )
    best = dict(max(candidates, key=lambda d: d["r2"]))
    logger.info(f"Correlogram model selected: {best['model']} (weighted R2={best['r2']:.3f})")
    # Copy first: `best` is one of the candidates, so attaching the list to the
    # original dict would make it self-referential and unserializable.
    best["candidates"] = [dict(candidate) for candidate in candidates]
    return best


def correlation_matrix(distances: np.ndarray, rho_km: float, nugget: float, plateau: float = 0.0) -> np.ndarray:
    """Latent Gaussian correlation matrix from the fitted decay."""
    sigma = model_correlation(distances, rho_km, nugget, plateau)
    np.fill_diagonal(sigma, 1.0)
    return sigma


def cholesky_factor(sigma: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Lower Cholesky factor, nudging the diagonal if the matrix is not quite PD."""
    for attempt in range(8):
        try:
            return np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            scale = jitter * (10**attempt)
            sigma = sigma + np.eye(len(sigma)) * scale
            logger.warning(f"Correlation matrix not positive definite; added {scale:.2e} to the diagonal.")
    raise np.linalg.LinAlgError("Correlation matrix could not be factorized.")
