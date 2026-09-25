"""Constants for the application: paths, grid geometry and enums."""

from enum import Enum

from src.tools.paths import DATA_DIR, RESULTS_DIR, ROOT_DIR  # noqa: F401 - re-exported

# Raw inputs
PATH_RAW_DEMAND = DATA_DIR / "raw_demand/base_customers_all_years_z7.csv"

PATH_DATA_PIXEL = DATA_DIR / "raw_pixel/input_pixels.xlsx"
PATH_RAW_PIXELS = DATA_DIR / "raw_pixel/raw_pixels.csv"
PATH_GRID_PIXELS = DATA_DIR / "raw_pixel/grid_pixels.geojson"
PATH_CUSTOMER_PIXEL_LAYER = DATA_DIR / "raw_pixel/customer_pixel_layer.csv"

PATH_DATA_FACILITY = DATA_DIR / "raw_facility/input_facilities.xlsx"

PATH_DATA_DISTANCES_FACILITY_DELIVERY_ZONE = DATA_DIR / "raw_distance/input_matrix_distance_facilities_pixels.xlsx"
PATH_DATA_DISTANCES_FACILITIES = DATA_DIR / "raw_distance/input_matrix_distance_dc_facilities.xlsx"

# The monthly panel is ~5.8k rows (161 pixels x 36 months), so CSV keeps it
# readable and avoids pulling in a parquet engine. It is a cache derived from the raw
# demand file; each params artifact keeps its own copy of the panel it was fitted on.
PATH_PANEL_MONTHLY = DATA_DIR / "interim" / "panel_monthly.csv"
PATH_PANEL_SOURCE = DATA_DIR / "interim" / "panel_source.json"

# ── Pixel grid geometry ───────────────────────────────────────────────────────
# grid_pixels.geojson is a perfectly regular lon/lat grid of 16 columns x 19 rows
# of ~1 km^2 cells. Cell ids follow `pixel = row * N_COLS + col`, with `col`
# increasing eastward and `row` increasing northward from the SW corner. Verified
# against all 304 features.
GRID_LON0 = -68.1720351
GRID_LAT0 = -16.6100101
GRID_DLON = 0.009371315934402
GRID_DLAT = 0.009036286905257
GRID_N_COLS = 16
GRID_N_ROWS = 19

# Number of periods in the planning horizon. The Continuous Approximation
# hardcodes 12; emitting scenarios with a different length desynchronizes it.
N_PERIODS = 12

# ── Demand panel construction ─────────────────────────────────────────────────
# `base_customers_all_years_z7.csv` was materialized through a join that fanned
# each delivery event out once per active year of the customer. Verified: row
# multiplicity equals the customer's distinct-year count for 100% of customers.
EXPECTED_FANOUT_SHARE = 0.632

# 2021-02 holds 23,401 demand units against a February median of 76,680, with
# normal delivery-date coverage: an extraction gap, not seasonality.
EXCLUDED_YEAR_MONTHS = ((2021, 2),)

# Threshold on customer drop size (items per visit) used to impute `layer` for
# customers absent from the manual crosswalk. Recovers the hand-made labels with
# 92.1% accuracy; the optimal per-pixel threshold only reaches 95.2%, so `layer`
# is not a deterministic function of drop.
LAYER_DROP_THRESHOLD = 11.62

# Demand regimes, anchored to the raw quantiles (p10 / p50 / p90) of the per-period
# totals across the 35 usable months (2020-01..2022-12 minus the excluded 2021-02).
#
# IMPORTANT — units. These are in *model demand*, i.e. sum_j(stop_j * drop_j), which
# represents one representative delivery round in the period. That is NOT the raw
# monthly item count: the model's per-period demand averages ~45k in 2022 while raw
# monthly items average ~149k, because a customer is served several times a month.
# Mixing the two scales silently inflates every routing cost by ~3x.
#
# These targets embed the 2020-2022 growth trend by construction, so "normal"
# (32,678) sits below the 2022 per-period average of 45,103 — see
# data/scenarios/ESCENARIOS_DOCUMENTACION.md for the detrended alternative.
REGIME_TARGETS = {
    "low": 20_465.0,
    "normal": 32_678.0,
    "high": 48_320.0,
}
REGIMES = tuple(REGIME_TARGETS)

# A regime moves the delivery frequency and the order size together.  The exponents
# add to one so, before stop rounding, their product remains the regime multiplier.
REGIME_STOP_EXPONENT = 0.70
REGIME_DROP_EXPONENT = 0.30

SEED_BASE = 20260908


class TypeOfFlexibility(Enum):
    """Operational flexibility after the satellite capacity is installed."""

    FIXED_OPERATION = "fixed_operation"
    ON_OFF_INSTALLED = "on_off_installed"
    UP_TO_INSTALLED = "up_to_installed"
