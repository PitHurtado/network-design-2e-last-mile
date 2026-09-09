"""Assignment of customers to the `(layer, pixel)` keys used by the model.

The 161 `(layer, pixel)` keys were built by hand: customers were split by drop
class (`layer`), aggregated onto the ~1 km^2 grid, and sparse cells were merged
with an adjacent cell holding demand to form a rectangular super-pixel. That is
why `area_surface` equals the number of grid cells a pixel spans (verified: 100%
agreement over all 161 keys).

`base_customers_all_years_z7.csv` carries neither `layer` nor `pixel`, so:

* where a customer appears in the manual crosswalk, the crosswalk wins — it
  encodes the hand-made merges, which are not purely geometric (the arithmetic
  grid cell agrees with the crosswalk pixel in only 89% of rows);
* otherwise the customer's grid cell is mapped onto the merged footprint that
  contains it, and `layer` is imputed from the customer's drop size.

Every assignment carries a `layer_source` of `crosswalk` or `imputed` so the cost
of the imputation can be measured downstream.
"""

import numpy as np
import pandas as pd

from src.constants import (
    GRID_DLAT,
    GRID_DLON,
    GRID_LAT0,
    GRID_LON0,
    GRID_N_COLS,
    GRID_N_ROWS,
    LAYER_DROP_THRESHOLD,
    PATH_CUSTOMER_PIXEL_LAYER,
)
from src.utils.custom_logger import get_logger

logger = get_logger("Crosswalk")

OUTSIDE_GRID = -1


def grid_cell(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Map lon/lat onto the regular grid cell id (`row * N_COLS + col`).

    Returns `OUTSIDE_GRID` for coordinates falling outside the grid extent.
    """
    col = np.floor((np.asarray(lon) - GRID_LON0) / GRID_DLON).astype(int)
    row = np.floor((np.asarray(lat) - GRID_LAT0) / GRID_DLAT).astype(int)
    inside = (col >= 0) & (col < GRID_N_COLS) & (row >= 0) & (row < GRID_N_ROWS)
    return np.where(inside, row * GRID_N_COLS + col, OUTSIDE_GRID)


def load_manual_crosswalk() -> pd.DataFrame:
    """Load the hand-made customer -> `(layer, pixel)` assignment.

    The source file has one row per `(customer, year, month)`; `layer` and `pixel`
    are constant per customer (verified: 0 customers with more than one value), so
    it collapses to one row each.
    """
    df = pd.read_csv(
        PATH_CUSTOMER_PIXEL_LAYER,
        usecols=["cod_customer", "lon", "lat", "layer", "pixel", "demand", "n_dates"],
    )
    dropped = df["layer"].isna().sum() + df["pixel"].isna().sum()
    df = df.dropna(subset=["layer", "pixel"])
    if dropped:
        logger.warning(f"{dropped} crosswalk rows without layer/pixel were discarded.")

    df["layer"] = df["layer"].str.upper()
    df["pixel"] = df["pixel"].astype(int)

    per_customer = df.groupby("cod_customer").agg(
        layer=("layer", "first"),
        pixel=("pixel", "first"),
        n_layers=("layer", "nunique"),
        n_pixels=("pixel", "nunique"),
        lon=("lon", "first"),
        lat=("lat", "first"),
    )
    inconsistent = int(((per_customer["n_layers"] > 1) | (per_customer["n_pixels"] > 1)).sum())
    if inconsistent:
        logger.warning(f"{inconsistent} customers have a non-unique (layer, pixel); the first value was kept.")

    per_customer = per_customer.drop(columns=["n_layers", "n_pixels"]).reset_index()
    per_customer["cell"] = grid_cell(per_customer["lon"].values, per_customer["lat"].values)
    logger.info(f"Manual crosswalk: {len(per_customer)} customers over {per_customer['pixel'].nunique()} grid cells.")
    return per_customer


def build_footprints(crosswalk: pd.DataFrame) -> tuple[dict, dict]:
    """Derive each pixel's merged footprint from the crosswalk.

    Returns
    -------
    footprint : dict[(layer, pixel)] -> set[cell]
        Grid cells occupied by the customers of each pixel.
    cell_to_pixel : dict[(layer, cell)] -> pixel
        Reverse lookup used to place a new customer inside an existing pixel.
    """
    footprint: dict[tuple[str, int], set[int]] = {}
    cell_to_pixel: dict[tuple[str, int], int] = {}

    inside = crosswalk[crosswalk["cell"] != OUTSIDE_GRID]
    for (layer, pixel), sub in inside.groupby(["layer", "pixel"]):
        cells = set(sub["cell"].unique())
        footprint[(layer, pixel)] = cells
        for cell in cells:
            # A cell claimed by two pixels of the same layer is assigned to the
            # pixel contributing more customers there.
            key = (layer, cell)
            if key in cell_to_pixel and cell_to_pixel[key] != pixel:
                incumbent = (inside["layer"].eq(layer) & inside["cell"].eq(cell) & inside["pixel"].eq(cell_to_pixel[key])).sum()
                challenger = (sub["cell"] == cell).sum()
                if challenger <= incumbent:
                    continue
            cell_to_pixel[key] = pixel

    logger.info(f"Footprints: {len(footprint)} (layer, pixel) keys covering {len({c for (_, c) in cell_to_pixel})} cells.")
    return footprint, cell_to_pixel


def _pixel_centroids(crosswalk: pd.DataFrame) -> dict[tuple[str, int], tuple[float, float]]:
    """Mean customer position per pixel, used for the nearest-pixel fallback."""
    g = crosswalk.groupby(["layer", "pixel"])[["lon", "lat"]].mean()
    return {key: (row.lon, row.lat) for key, row in g.iterrows()}


def assign_customers(customers: pd.DataFrame, crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    """Assign `(layer, pixel)` to every customer.

    Parameters
    ----------
    customers : DataFrame
        One row per customer with `cod_customer`, `lon`, `lat` and `drop`
        (items per visit, used to impute `layer`).

    Returns
    -------
    DataFrame with `cod_customer`, `layer`, `pixel`, `id_pixel`, `layer_source`.
    Customers that cannot be placed on the grid are returned with a null pixel and
    must be filtered by the caller.
    """
    if crosswalk is None:
        crosswalk = load_manual_crosswalk()

    _, cell_to_pixel = build_footprints(crosswalk)
    centroids = _pixel_centroids(crosswalk)

    out = customers.copy()
    out["cell"] = grid_cell(out["lon"].values, out["lat"].values)

    known = crosswalk.set_index("cod_customer")[["layer", "pixel"]]
    joined = out.join(known, on="cod_customer")
    from_crosswalk = joined["layer"].notna()

    # Impute layer from drop size for the rest.
    imputed_layer = np.where(out["drop"].values >= LAYER_DROP_THRESHOLD, "B", "A")
    layer = np.where(from_crosswalk, joined["layer"].values, imputed_layer)

    pixel = joined["pixel"].values.astype("float64")
    fallbacks = 0
    for idx in np.flatnonzero(~from_crosswalk.values):
        cell = out["cell"].values[idx]
        if cell == OUTSIDE_GRID:
            continue
        key = (layer[idx], cell)
        if key in cell_to_pixel:
            pixel[idx] = cell_to_pixel[key]
            continue
        # The cell holds no pixel of the imputed layer: fall back to the nearest
        # pixel of that layer by centroid distance.
        candidates = [(k, v) for k, v in centroids.items() if k[0] == layer[idx]]
        if not candidates:
            continue
        lon_i, lat_i = out["lon"].values[idx], out["lat"].values[idx]
        nearest = min(candidates, key=lambda kv: (kv[1][0] - lon_i) ** 2 + (kv[1][1] - lat_i) ** 2)
        pixel[idx] = nearest[0][1]
        fallbacks += 1

    out["layer"] = layer
    out["pixel"] = pixel
    out["layer_source"] = np.where(from_crosswalk, "crosswalk", "imputed")
    out["id_pixel"] = [f"{lay}-{int(px)}" if np.isfinite(px) else None for lay, px in zip(out["layer"], out["pixel"])]

    placed = out["id_pixel"].notna()
    logger.info(
        f"Assigned {placed.sum()} of {len(out)} customers "
        f"({(~from_crosswalk & placed).sum()} imputed, {fallbacks} via nearest-pixel fallback); "
        f"{(~placed).sum()} could not be placed on the grid."
    )
    return out.drop(columns=["cell"])
